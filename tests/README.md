# ALIAS Test Suite

Comprehensive testing for the ALIAS package covering imports, configurations, dependencies, and full pipeline integration.

## Test Organization

### Fast Unit Tests (~2s)
These tests run quickly without data files:

- **`test_imports.py`** (59 tests) - Smoke tests verifying all modules can be imported
- **`test_configs.py`** (17 tests) - Configuration dataclass validation
- **`test_dependencies.py`** (14 tests) - Dependency integration and compatibility

### Integration Tests (~3s)
Tests with real data marked with `@pytest.mark.integration`:

- **`test_integration_pipeline.py`** (4 tests) - Full pipeline with 6.4 MB test data
  - Dataset building
  - Triplet generation
  - Pipeline validation

## Running Tests

### Run All Tests
```bash
uv run pytest tests/ -v
```

### Run Fast Tests Only
```bash
uv run pytest tests/ -v -m "not integration"
```

### Run Integration Tests Only
```bash
uv run pytest tests/ -v -m integration
```

### Run Specific Test File
```bash
uv run pytest tests/test_configs.py -v
```

## Test Data

Integration tests use real scRNA-seq data:
- **Location**: `tests/data/test_adata.h5ad` (6.4 MB)
- **Size**: 981 cells × 33,538 genes
- **Cell types**: 17 types (filtered to ≥5 cells each)
- **Purpose**: Realistic testing with actual biological data

## Test Fixtures

Defined in `tests/fixtures.py`:

- **`tiny_adata`** - Fast simulated data (10 cells, 20 genes)
- **`real_adata`** - Real test data loaded once per module
- **`sample_dataset_dict`** - HuggingFace Dataset samples

## What's Tested

### Phase 1: Fast Unit Tests
✅ **Imports** - All modules import correctly  
✅ **Configs** - Dataclass validation and defaults  
✅ **Dependencies** - Package compatibility (numpy, scipy, pandas, anndata, scanpy, torch, etc.)  
✅ **API** - Public API consistency  

### Phase 2: Integration Tests
✅ **Data Pipeline** - AnnData → Datasets → Triplets  
✅ **Real Data** - Works with actual scRNA-seq data  
✅ **Configuration** - End-to-end config validation  
✅ **Bug Detection** - Caught real bugs in triplet generation!  

## Bug Fixes from Tests

### Fixed: Triplet Generation Bug
**Issue**: `UnboundLocalError` when `hard_negative_mining=False`  
**Root Cause**: `train_dataset` and `eval_dataset` only created in hard_negative_mining path  
**Fix**: Added train/test split for random_negative_mining path  
**File**: `src/alias/data/triplet_generation.py`  

This bug would have caused runtime failures for any user using random negative mining without hard negative mining!

## CI/CD Integration

Tests run automatically on:
- Push to `main`, `master`, `develop` branches
- Pull requests
- Python versions: 3.11, 3.12, 3.13

See `.github/workflows/test.yml` for CI configuration.

## Best Practices

1. **Write fast tests first** - Use simulated data for unit tests
2. **Use integration tests for realism** - Real data catches real bugs
3. **Mark slow tests** - Use `@pytest.mark.integration` for tests with real data
4. **Keep test data small** - 5-10 MB is ideal for git tracking
5. **Test the happy path** - Focus on common workflows first

## Adding New Tests

### For New Modules
Add import tests to `test_imports.py`:
```python
def test_import_new_module(self):
    import alias.new_module
    assert alias.new_module is not None
```

### For New Configs
Add validation tests to `test_configs.py`:
```python
def test_new_config_creation(self):
    from alias import NewConfig
    config = NewConfig(required_param="value")
    assert config.required_param == "value"
```

### For New Pipeline Steps
Add integration tests to `test_integration_pipeline.py`:
```python
@pytest.mark.integration
def test_new_pipeline_step(self, real_adata):
    result = new_pipeline_function(real_adata)
    assert len(result) > 0
```

## Performance

| Test Suite | Tests | Time | Data |
|------------|-------|------|------|
| Imports | 59 | ~1s | None |
| Configs | 17 | ~0.5s | None |
| Dependencies | 14 | ~0.5s | Synthetic |
| Integration | 4 | ~3s | 6.4 MB |
| **Total** | **67** | **~3s** | **6.4 MB** |

Fast enough to run on every commit! ⚡

