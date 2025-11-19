# ALIAS (Adding Layers of Information for the analysis of scRNA-seq data)

This is a first version of the codebase corresponding to our PrePrint [(bioRxiv)](https://www.biorxiv.org/content/10.1101/2025.08.23.671699v1), where we show how small encoder-only language models can be used to generate a joint embedding space for scRNA-seq data with corresponding biomedical literature.

![](images/concept.png)


Please note that, as of now, not all evaluation functions shown in the PrePrint are implemented and the tutorials for using the repo are still WIP. We will have a ready to use repo within the next couple of weeks.

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd alias

# Install dependencies using uv
uv sync

# Set up your Hugging Face tokens
cp .env.example .env
# Edit .env and add your actual Hugging Face tokens
```

### Environment Variables

The project requires Hugging Face tokens for accessing and uploading models/datasets.

**Quick setup:**
```bash
cp .env.example .env
# Edit .env and add your actual Hugging Face tokens
```

📖 **For detailed configuration instructions, troubleshooting, and usage examples, see [CONFIGURATION.md](CONFIGURATION.md)**

## Getting Started

### Demo Notebook

Check out the complete training pipeline in our demo notebook:

```bash
# Place your demo data at data/demo.h5ad
# Then run the notebook
uv run jupyter notebook notebooks/demo_training_pipeline.ipynb
```

See [`notebooks/README.md`](notebooks/README.md) for more details.

## Testing

The package is tested on Python 3.11, 3.12, and 3.13.

### Test Suite

- **Import tests** - Verify all modules can be imported
- **Config validation** - Test configuration classes and defaults
- **Dependency integration** - Ensure package dependencies work together

To run tests locally:

```bash
# Run all tests (67 tests, ~3 seconds)
uv run pytest tests/ -v

# Run fast unit tests only (63 tests, ~2 seconds)
uv run pytest tests/ -v -m "not integration"

# Run integration tests with real data (4 tests)
uv run pytest tests/ -v -m integration

# Run specific test files
uv run pytest tests/test_imports.py -v
uv run pytest tests/test_configs.py -v
uv run pytest tests/test_integration_pipeline.py -v
```

Tests run automatically on every push via GitHub Actions. See [`tests/README.md`](tests/README.md) for details. 








