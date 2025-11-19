# Notebooks

This directory contains Jupyter notebooks demonstrating ALIAS workflows.

## Available Notebooks

### 1. `demo_training_pipeline.ipynb`
Complete end-to-end training pipeline:
- Load scRNA-seq data
- Build datasets
- Generate triplets
- Train sentence transformer model

**Prerequisites**: Place your `.h5ad` file at `data/demo.h5ad`

## Running Notebooks

### Setup

```bash
# Make sure you're in the project root
cd /Users/mengerj/repos/alias

# Activate environment
source .venv/bin/activate  # or your environment activation

# Install Jupyter if not already installed
uv add --optional dev jupyter

# Start Jupyter
jupyter notebook notebooks/
```

### Quick Start with uv

```bash
# Run Jupyter with uv
uv run jupyter notebook notebooks/demo_training_pipeline.ipynb
```

## Notebook Structure

Each notebook follows this pattern:
1. **Setup**: Import libraries and load data
2. **Configuration**: Set up configs for each step
3. **Processing**: Run the pipeline
4. **Verification**: Check outputs
5. **Summary**: Next steps and tips

## Tips

- Start with small datasets for testing
- Reduce `epochs` and `batch_size` for quick tests
- Set `testrun=True` in configs for rapid iteration
- Check GPU availability: `torch.cuda.is_available()`

## Troubleshooting

**Out of Memory?**
- Reduce `batch_size` in TrainingSTConfig
- Use fewer cells (subset your data)
- Set `fp16=False`

**Training too slow?**
- Reduce `epochs`
- Smaller dataset
- Check if GPU is being used

**Import errors?**
- Make sure you're in the virtual environment
- Run `uv sync` to install dependencies

