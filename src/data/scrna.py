
from dataclasses import dataclass, asdict
from datasets import Dataset
from typing import Optional, List, Tuple
from tqdm import tqdm
from scipy import sparse
import numpy as np
import pandas as pd
import scanpy as sc
import random
from pathlib import Path

from util.cell_sentence_templates import TEMPLATES


@dataclass
class DatascRNAConfig:
    """Configuration for scRNA dataset generation."""
    annotation_column: str = "celltype"
    preprocessing: bool = False
    random_state: int = 42
    
    # --- CELL SENTENCE PARAMETERS ---
    cs_length: List[int] = (10,)
    disease_column: Optional[str] = None
    time_column: Optional[str] = None
    highly_variable_genes: bool = True
    housekeeping_genes: bool = True
    semantic: bool = True
    
    # --- SEMANTIC TEMPLATE WEIGHTS ---
    template_weights_default: Optional[dict] = None
    template_weights_disease: Optional[dict] = None
    template_weights_time: Optional[dict] = None
    
    # --- DATASET PARAMETERS ---
    test_size: float = 0.1
    
    # --- PREPROCESSING PARAMETERS ---
    hvg_number: int = 3000
    min_genes: int = 100
    mt_threshold: float = 15
    min_cells: int = 5
    min_batch_cells: int = 5
    verbose: bool = True
    
def run_preprocessing(adata, batch_key, scrna_config: DatascRNAConfig, **kwargs):
    """Optional preprocessing of AnnData object using config parameters, overridable with kwargs."""
    cfg = asdict(scrna_config)
    cfg.update(kwargs) 
    
    verbose = cfg['verbose']

    if verbose:
        print("Preprocessing AnnData object...")

    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    genes_upper = adata.var_names.str.upper()
    adata.var["mt"] = genes_upper.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
        
    # Filter cells and genes
    if verbose:
        print(f"Filtering cells with fewer than {cfg['min_genes']} expressed genes...")
    sc.pp.filter_cells(adata, min_genes=cfg['min_genes'])

    if verbose:
        print(f"Filtering cells with a percentage of mitochondrial genes expressed over {cfg['mt_threshold']}...")
    adata = adata[adata.obs["pct_counts_mt"] < cfg['mt_threshold']].copy()
    
    if verbose:
        print(f"Filtering genes expressed in fewer than {cfg['min_cells']} cells...")
    sc.pp.filter_genes(adata, min_cells=cfg['min_cells'])

    # Remove small batches
    if verbose:
        print("Analyzing cell type sizes...")
        print(f"Initial number of celtypes: {adata.obs[batch_key].nunique()}")
    batch_sizes = adata.obs[batch_key].value_counts()
    valid_batches = batch_sizes[batch_sizes >= cfg['min_batch_cells']].index
    if verbose:
        print(f"Batches with fewer than {cfg['min_batch_cells']} cells: {len(batch_sizes) - len(valid_batches)}")
        print("Filtering small batches...")
    adata = adata[adata.obs[batch_key].isin(valid_batches)].copy()

    # Save counts to separate layer
    adata.layers["counts"] = adata.X.copy()

    # Normalize, log transform, and identify highly variable genes
    if not adata.uns.get("normalized", False):
        if verbose:
            print("Normalizing total counts per cell...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        adata.uns["normalized"] = True

    if not adata.uns.get("logged", False):
        if verbose:
            print("Performing log transformation...")
        sc.pp.log1p(adata)
        adata.uns["logged"] = True
    
    if verbose:
        print(f"Identifying {cfg['hvg_number']} highly variable genes, accounting for batches using '{batch_key}'...")
    sc.pp.highly_variable_genes(adata, n_top_genes=cfg['hvg_number'], batch_key=batch_key)
    
    if verbose:
        print("Subsetting to highly variable genes...")
    adata = adata[:, adata.var.highly_variable].copy()

    if verbose:
        print("Preprocessing complete. Your AnnData object is ready for further analysis.")
        print(adata)
    return adata

def generate_semantic_sentence(
    gene_list, cell_type,
    scrna_config=None, time=None, disease_status=None, **kwargs
):
    """Create a semantic sentence for each cell, with user-controllable template weights."""
    cfg = asdict(scrna_config) if scrna_config else {}
    cfg.update(kwargs)  # override with kwargs

    gene_str = ", ".join(gene_list)

    # Determine weights
    default_disease = {
        'genes_celltype_disease': 0.5,
        'genes_disease': 0.2,
        'genes_celltype': 0.2,
        'genes': 0.1
    }
    default_time = {
        'genes_celltype_time': 0.5,
        'genes_time': 0.2,
        'genes_celltype': 0.2,
        'genes': 0.1
    }
    default_default = {
        'genes_celltype': 0.7 if cell_type else 0.0,
        'genes': 0.3 if cell_type else 1.0
    }

    # pick which template_weights to use
    if cfg.get('disease_column'):
        template_weights = cfg.get('template_weights_disease') or default_disease
    elif cfg.get('time_column'):
        template_weights = cfg.get('template_weights_time') or default_time
    else:
        template_weights = cfg.get('template_weights_default') or default_default

    categories, weights = zip(*template_weights.items())
    selected_category = random.choices(categories, weights=weights, k=1)[0]
    template = random.choice(TEMPLATES[selected_category])

    return template.format(
        gene_str=gene_str,
        cell_type=cell_type,
        time=time,
        disease_status=disease_status
    )


def process_split(ds, annotation_column, scrna_config, semantic, **kwargs):
    """Add sentence1 column to HF dataset, overridable with kwargs."""
    if semantic:
        ds = ds.map(lambda row: {
            "sentence1": generate_semantic_sentence(
                row["gene_list"],
                row.get(annotation_column),
                scrna_config=scrna_config,
                time=row.get("time"),
                disease_status=row.get("disease_status"),
                **kwargs
            )
        })
    else:
        ds = ds.map(lambda row: {"sentence1": " ".join(row["gene_list"])})
    return ds


def gen_scrna_dataset(adata, scrna_config: DatascRNAConfig, **kwargs) -> Tuple[Dataset, Dataset]:
    """Generate train/test HuggingFace Datasets from AnnData object, overridable with kwargs."""
    cfg = asdict(scrna_config)
    cfg.update(kwargs)  # override any config with kwargs

    annotation_column = cfg['annotation_column']

    if cfg['preprocessing']:
        adata = run_preprocessing(adata, annotation_column, scrna_config, **kwargs)

    # Subset to highly variable genes if requested
    if cfg['highly_variable_genes']:
        if 'highly_variable' not in adata.var:
            raise ValueError("Missing 'highly_variable' column in adata.var.")
        adata_subset = adata[:, adata.var['highly_variable']]
    else:
        adata_subset = adata

    X = adata_subset.X.toarray() if sparse.issparse(adata_subset.X) else adata_subset.X
    genes_upper = adata_subset.var_names.str.upper()
    genes = adata_subset.var.index


    # Filter housekeeping genes
    if cfg['housekeeping_genes']:
        mask = ~(
            genes_upper.str.startswith("MT-") |
            genes_upper.str.startswith("RPS") |
            genes_upper.str.startswith("RPL")
        )
        genes = genes[mask]
        X = X[:, mask]

    all_rows = []
    for i in tqdm(range(X.shape[0]), desc="Building dataset"):
        expr_values = X[i, :]
        ranked_genes = genes[np.argsort(-expr_values, kind="stable")[:cfg['cs_length'][0]]].tolist()

        cell_type = adata_subset.obs[annotation_column].iloc[i]
        row = {
            "index": adata_subset.obs.index[i],
            "gene_list": ranked_genes,
            annotation_column: cell_type,
            "label": cell_type,
        }

        if cfg['disease_column']:
            disease_status = adata_subset.obs[cfg['disease_column']].iloc[i]
            row["disease_status"] = disease_status
            row["label"] = f"{cell_type}_{disease_status}" if pd.notnull(disease_status) else cell_type

        if cfg['time_column']:
            time = adata_subset.obs[cfg['time_column']].iloc[i]
            row["time"] = time
            row["label"] = f"{cell_type}_{time}" if pd.notnull(time) else cell_type

        all_rows.append(row)

    df = pd.DataFrame(all_rows).set_index("index")
    ds = Dataset.from_pandas(df.reset_index())

    # Split into train/test
    ds_split = ds.train_test_split(test_size=cfg['test_size'], seed=cfg['random_state'])
    train_ds = process_split(ds_split["train"], annotation_column, scrna_config, semantic=cfg['semantic'], **kwargs)
    test_ds = process_split(ds_split["test"], annotation_column, scrna_config, semantic=False, **kwargs)
    
    test_indices = test_ds["index"]
    adata_test = adata_subset[test_indices]
    
    return train_ds, test_ds, adata_test
