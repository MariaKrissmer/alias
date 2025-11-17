
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Literal
from datasets import Dataset, concatenate_datasets
import pandas as pd
import torch, random
from tqdm import tqdm

from util.hard_negative_mining import mine_hard_negatives
from util.load_hf_model import load_model, upload_dataset_to_hf


@dataclass
class TripletGenerationConfig:
    annotation_column: str
    eval_split: float = 0.1
    seed: int = 42
    loss: Literal['MNR', 'Triplet', 'Contrastiv']  = 'MNR' #Loss type: MNR, Triplet, Contrastive
    testrun: bool = False
    random_negative_mining: bool = False
    hard_negative_mining: bool = False
    model: Optional[str] = "neuml/pubmedbert-base-embeddings" # Model reference for hard negative mining
    batch_size: Optional[int] = 128
    subset_size: Optional[int] = 10000
    
    hf_upload: bool = False
    hf_name: str = None


def generate_triplets(
    train_data=None, 
    type=None,
    triplet_config: TripletGenerationConfig = None, 
    **kwargs) -> Tuple[Dataset, Dataset]:
    """Generate triplets from data using config, overridable with kwargs."""
    
    cfg = asdict(triplet_config)
    cfg.update(kwargs)
    
    if isinstance(train_data, Dataset):
        data = train_data.to_pandas()
    else:
        data = train_data
        
    random.seed(cfg["seed"])

    # Choose the correct column names depending on training mode
    if type not in ("ncbi", "scrna"):
        raise ValueError(f"Unknown dataset type: {type}")
    
    # Make sure labels are strings
    annotation_column = cfg["annotation_column"]
    if annotation_column in data.columns:
        data[annotation_column] = data[annotation_column].astype(str)
        
    if 'label' in data.columns:
        data['label'] = data['label'].astype(str)
        
    # Load datset with label column and set up training data so that it fits the requirements needed for loss    
    if cfg["loss"] == 'MNR' or 'Contrastive' or 'Triplet':
        
        print("Generating semantic training pairs for training with MultipleNgeativesRanking loss...")

        if type == 'ncbi':
            
            print("Processing ncbi data...")
            
            label_groups = data.groupby("label")
            pairs = []

            for label, group in label_groups:
                titles = group[group['type'] == 'title']['sentence1'].tolist()
                abstracts = group[group['type'] == 'abstract']['sentence1'].tolist()

                # Title ↔ Title
                for i in range(len(titles)):
                    current = titles[i]
                    others = [t for t in titles if t != current]
                    sample_size = min(5, len(others))
                    sampled = random.sample(others, sample_size)
                    for other in sampled:
                        pairs.append((current, other, label))

                # Abstract ↔ Abstract
                for i in range(len(abstracts)):
                    current = abstracts[i]
                    others = [a for a in abstracts if a != current]
                    sample_size = min(5, len(others))
                    sampled = random.sample(others, sample_size)
                    for other in sampled:
                        pairs.append((current, other, label))

            df = pd.DataFrame(pairs, columns=["sentence1", "sentence2", "label"])
            if cfg["testrun"]:
                df = df[:500]
            
            print(f"Created {len(df)} total semantic pairs.")

        
        elif type == 'scrna':
            
            print("Processing scrna data...")
            
            label_groups = data.groupby("label")
            pairs = []
            
            for label, group in tqdm(label_groups, desc="Generating pairs by label"):
                texts = group['sentence1'].tolist()

                # Generate pairs based on the same label
                for i in range(len(texts)):
                    current = texts[i]

                    # Create a list of indices for all other sentences, excluding the current one
                    other_indices = list(range(len(texts)))
                    other_indices.remove(i)

                    sample_size = min(5, len(other_indices))
                    sampled_indices = random.sample(other_indices, sample_size)
                    
                    for idx in sampled_indices:
                        pairs.append((current, texts[idx], label))

            df = pd.DataFrame(pairs, columns=["sentence1", "sentence2", "label"])
            
            if cfg["testrun"]:
                df = df[:500]
            
            print(f"✅ Created {len(df)} total semantic pairs.")         
            
    
    if cfg["random_negative_mining"]:
        # Create a dictionary to store sentences from different labels for faster lookup
        negative_pool_by_label = {}
        
        # Fill the dictionary with sentences from each label (excluding the label itself)
        for label in df['label'].unique():
            negative_pool_by_label[label] = df[df['label'] != label]['sentence1'].tolist()
        
        # Function to sample a negative sentence
        def sample_negative(row):
            # Get the negative pool (sentences from a different label)
            negative_pool = negative_pool_by_label[row['label']]
            # Sample a random sentence from the negative pool
            return random.choice(negative_pool)

        # Apply the sampling function to create the 'negative' column
        df['negative'] = df.apply(sample_negative, axis=1)
        
        del df['label']
        
        print(df.head(20))
        
        dataset = Dataset.from_pandas(df)
    
    if cfg["hard_negative_mining"]:
        
        dataset = Dataset.from_pandas(df)
    
        # Shuffle the dataset first
        dataset = dataset.shuffle(seed=cfg["seed"])
        
        model = load_model(cfg["model"])
    
        # Move model to appropriate device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model.to(device)
        
        if cfg["loss"] == 'MNR':
            output_format = 'paired-triplet' 
        
        elif cfg["loss"] == 'Triplet':
            output_format = 'paired-triplet'
        
        elif cfg["loss"] == 'Contrastiv':
            output_format = 'labeled-pair'
        
        # Break into batches manually
        subset_size = cfg["subset_size"]
        num_batches = len(dataset) // subset_size + (1 if len(dataset) % subset_size != 0 else 0)

        batched_datasets = []

        for i in tqdm(range(num_batches), desc="Mining hard negatives..."):
            start = i * subset_size
            end = min((i + 1) * subset_size, len(dataset))
            batch = dataset.select(range(start, end))

            mined_batch = mine_hard_negatives(
                dataset=batch,
                model=model,
                num_negatives=5,
                sampling_strategy="top",
                batch_size=cfg["batch_size"],
                use_faiss=False,
                use_gpu=False,
                anchor_column_name='sentence1',
                positive_column_name='sentence2',
                output_format=output_format,
                verbose=False,
                label_column_name='label',
                relative_margin=0,
                range_max=cfg["subset_size"]
            )
            
            batched_datasets.append(mined_batch)

        # Combine all mined batches back into one dataset
        dataset = concatenate_datasets(batched_datasets)
        
        # Split dataset into train and eval (e.g., 90% train, 10% eval)
        split = dataset.train_test_split(test_size=cfg["eval_split"], seed=cfg["seed"])  # fix seed for reproducibility
        train_dataset = split['train']
        eval_dataset = split['test']
        
        if cfg.get("hf_upload", False):
            train_data = {
                f"{type}_train": train_dataset,
                f"{type}_eval": eval_dataset
            }

            if cfg.get("hf_name"):
                upload_dataset_to_hf(dataset_dict=train_data, dataset_name=type, name=cfg["hf_name"])
            else:
                upload_dataset_to_hf(dataset_dict=train_data, dataset_name=type)


    return train_dataset, eval_dataset