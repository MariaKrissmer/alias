"""
Test fixtures for ALIAS tests.

Provides both simulated (fast) and real (realistic) test data.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from pathlib import Path


@pytest.fixture
def tiny_adata():
    """
    Minimal simulated AnnData for fast unit tests (10 cells, 20 genes).
    
    Use for: Quick unit tests, config validation, basic functionality.
    """
    n_obs, n_vars = 10, 20
    X = sparse.csr_matrix(np.random.poisson(5, (n_obs, n_vars)))
    
    obs = pd.DataFrame({
        'celltype': ['T_cell', 'B_cell', 'T_cell', 'B_cell', 'T_cell', 
                     'Monocyte', 'B_cell', 'T_cell', 'Monocyte', 'T_cell'],
        'batch': ['batch1'] * 5 + ['batch2'] * 5,
    }, index=[f'cell_{i}' for i in range(n_obs)])
    
    var = pd.DataFrame({
        'gene_names': [f'GENE{i}' for i in range(n_vars)],
        'highly_variable': [True] * 10 + [False] * 10
    }, index=[f'gene_{i}' for i in range(n_vars)])
    
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture(scope="module")
def real_adata():
    """
    Real test data (6.4 MB, ~1000 cells) loaded once per test module.
    
    Use for: Integration tests, full pipeline validation, realistic scenarios.
    """
    test_data_path = Path(__file__).parent / "data" / "test_adata.h5ad"
    
    if not test_data_path.exists():
        pytest.skip(f"Real test data not found at {test_data_path}")
    
    adata = sc.read_h5ad(test_data_path)
    
    # Add 'celltype' column if it doesn't exist
    # Try common cell type annotation columns
    if 'celltype' not in adata.obs.columns:
        for col in ['AIFI_L2', 'AIFI_L3', 'AIFI_L1', 'cell_type', 'cellType']:
            if col in adata.obs.columns:
                adata.obs['celltype'] = adata.obs[col]
                break
    
    # Filter out rare cell types (keep only types with >= 5 cells)
    cell_type_counts = adata.obs['celltype'].value_counts()
    valid_celltypes = cell_type_counts[cell_type_counts >= 5].index
    adata = adata[adata.obs['celltype'].isin(valid_celltypes)].copy()
    
    return adata


@pytest.fixture
def sample_dataset_dict():
    """Sample HuggingFace Dataset for testing."""
    from datasets import Dataset
    return {
        'scrna': {
            'data': Dataset.from_dict({
                'sentence1': ['GENE1 GENE2 GENE3'] * 10,
                'label': ['T_cell'] * 5 + ['B_cell'] * 5
            }),
            'test': Dataset.from_dict({
                'sentence1': ['GENE4 GENE5 GENE6'] * 5,
                'label': ['T_cell'] * 3 + ['B_cell'] * 2
            })
        }
    }

