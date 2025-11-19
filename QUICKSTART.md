# Quick Start Guide

## Complete Pipeline Workflow

The ALIAS workflow consists of 6 main steps:

```
1. Load .h5ad file    →    2. Configure    →    3. Build Datasets
        ↓                                              ↓
6. Evaluate Model     ←    5. Train Model    ←    4. Generate Triplets
```

## Running the Demo Notebook

### Step 1: Prepare Your Data

Place your scRNA-seq data file at:
```
data/demo.h5ad
```

**Requirements:**
- AnnData object in `.h5ad` format
- Cell type annotations in `obs` (e.g., column named `celltype`)
- Gene expression matrix in `X`

**Don't have data?** Generate a demo dataset:
```python
import scanpy as sc
adata = sc.datasets.pbmc3k_processed()
adata.write('data/demo.h5ad')
```

### Step 2: Run the Notebook

```bash
# Navigate to project root
cd /Users/mengerj/repos/alias

# Activate environment
source .venv/bin/activate

# Start Jupyter
uv run jupyter notebook notebooks/demo_training_pipeline.ipynb
```

### Step 3: Execute All Cells

The notebook will:
1. ✅ Load your `.h5ad` file
2. ✅ Convert cells to text-based "cell sentences"
3. ✅ Generate training triplets (positive/negative pairs)
4. ✅ Fine-tune a sentence transformer model
5. ✅ Save the trained model to `out/models/`

## Workflow Details

### 1. Data Loading
```python
import scanpy as sc
adata = sc.read_h5ad("data/demo.h5ad")
```

### 2. Configuration
```python
from alias.data import DatascRNAConfig

config = DatascRNAConfig(
    annotation_column="celltype",
    test_size=0.1,
    preprocessing=False
)
```

### 3. Build Datasets
```python
from alias.data import build_datasets

dataset_dict, adata_test = build_datasets(
    adata=adata,
    datasets=['scrna'],
    scrna_config=config
)
```

Creates HuggingFace Dataset with "cell sentences" - text representations of gene expression.

### 4. Generate Triplets
```python
from alias.data import TripletGenerationConfig, build_triplets

triplet_config = TripletGenerationConfig(
    annotation_column="celltype",
    loss='MNR'
)

triplet_dict = build_triplets(
    dataset_dict=dataset_dict,
    triplet_config=triplet_config
)
```

Creates training data: anchor, positive, negative triplets for contrastive learning.

### 5. Train Model
```python
from alias.model import TrainingSTConfig, train_model

training_config = TrainingSTConfig(
    model="neuml/pubmedbert-base-embeddings",
    loss='MNR',
    epochs=2,
    batch_size=32,
    save_to_local=True
)

trained_model = train_model(
    dataset_dict=triplet_dict,
    datasets='scrna',
    train_config=training_config
)
```

Fine-tunes a sentence transformer model on your data.

### 6. Use the Model
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("out/models/alias_demo_model")
embeddings = model.encode(cell_sentences)
```

## Configuration Tips

### Quick Test Run
```python
# Reduce dataset size
scrna_config = DatascRNAConfig(
    test_size=0.5,  # Use more data for testing
    cs_length=(10,)  # Single cell sentence length
)

# Quick training
training_config = TrainingSTConfig(
    model="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model
    epochs=1,
    batch_size=16,
    testrun=True
)
```

### Production Run
```python
# Full dataset
scrna_config = DatascRNAConfig(
    test_size=0.1,
    cs_length=(10, 20, 50),  # Multiple lengths
    preprocessing=True  # Enable preprocessing
)

# Better training
training_config = TrainingSTConfig(
    model="neuml/pubmedbert-base-embeddings",
    epochs=10,
    batch_size=64,
    fp16=True,  # If GPU available
    save_to_hf=True  # Upload to HuggingFace
)
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Use fewer cells (subset adata)
- Set `fp16=False`

### Slow Training
- Check GPU: `torch.cuda.is_available()`
- Reduce `epochs`
- Use smaller model

### Import Errors
- Activate environment: `source .venv/bin/activate`
- Install dependencies: `uv sync`

### Data Issues
- Check annotation column exists: `print(adata.obs.columns)`
- Verify data shape: `print(adata.shape)`
- Check cell types: `print(adata.obs['celltype'].value_counts())`

## Next Steps

1. **Evaluate**: Use embeddings for downstream tasks
2. **Multi-modal**: Integrate with NCBI literature data
3. **Scale up**: Train on larger datasets
4. **Deploy**: Push model to HuggingFace Hub

## File Structure

```
alias/
├── data/
│   └── demo.h5ad          ← Place your data here
├── notebooks/
│   └── demo_training_pipeline.ipynb
├── out/
│   └── models/
│       └── alias_demo_model/   ← Trained model saved here
└── src/alias/
    ├── data/              ← Dataset building
    ├── model/             ← Model training
    └── util/              ← Utilities
```

## Support

- 📚 Full documentation: [README.md](README.md)
- 🔧 Configuration guide: [CONFIGURATION.md](CONFIGURATION.md)
- 🧪 Testing: [tests/README.md](tests/README.md)
- 📓 Notebooks: [notebooks/README.md](notebooks/README.md)

