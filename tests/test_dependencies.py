"""
Test that critical dependencies work together correctly.

These tests verify that key package dependencies are compatible with each other
and work as expected with the chosen versions.
"""

import pytest
import numpy as np
import pandas as pd


class TestCoreScientificPackages:
    """Test numpy, scipy, and pandas compatibility."""

    def test_numpy_basic_operations(self):
        """Test numpy basic operations."""
        arr = np.array([1, 2, 3, 4, 5])
        
        assert arr.sum() == 15
        assert arr.mean() == 3.0
        assert arr.shape == (5,)

    def test_scipy_sparse_matrices(self):
        """Test scipy sparse matrix operations."""
        from scipy import sparse
        
        # Create a sparse matrix
        dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
        sparse_csr = sparse.csr_matrix(dense)
        sparse_csc = sparse.csc_matrix(dense)
        
        # Verify conversion back to dense
        np.testing.assert_array_equal(sparse_csr.toarray(), dense)
        np.testing.assert_array_equal(sparse_csc.toarray(), dense)
        
        # Test sparse operations
        assert sparse_csr.sum() == 15
        assert sparse_csr.shape == (3, 3)

    def test_numpy_scipy_compatibility(self):
        """Test numpy and scipy work together."""
        from scipy import sparse
        
        # Create with numpy, convert to sparse
        dense = np.random.poisson(1, (10, 20))
        sparse_mat = sparse.csr_matrix(dense)
        
        # Convert back and verify
        back_to_dense = sparse_mat.toarray()
        np.testing.assert_array_equal(dense, back_to_dense)

    def test_pandas_dataframe_operations(self):
        """Test pandas DataFrame basic operations."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': ['x', 'y', 'z']
        })
        
        assert len(df) == 3
        assert list(df.columns) == ['a', 'b', 'c']
        assert df['a'].sum() == 6


class TestBioinformaticsPackages:
    """Test anndata and scanpy compatibility."""

    def test_anndata_creation(self):
        """Test creating an AnnData object."""
        import anndata as ad
        from scipy import sparse
        
        # Create minimal AnnData
        X = sparse.csr_matrix(np.random.poisson(1, (5, 10)))
        obs = pd.DataFrame({'celltype': ['A', 'B', 'A', 'B', 'A']}, 
                          index=[f'cell_{i}' for i in range(5)])
        var = pd.DataFrame({'gene': [f'gene_{i}' for i in range(10)]},
                          index=[f'gene_{i}' for i in range(10)])
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        assert adata.shape == (5, 10)
        assert 'celltype' in adata.obs.columns
        assert len(adata.var) == 10

    def test_scanpy_preprocessing(self):
        """Test scanpy preprocessing functions."""
        import anndata as ad
        import scanpy as sc
        from scipy import sparse
        
        # Create test data
        X = sparse.csr_matrix(np.random.poisson(5, (10, 20)))
        adata = ad.AnnData(X=X)
        
        # Test normalize_total
        sc.pp.normalize_total(adata, target_sum=1e4)
        assert adata.X.sum() > 0
        
        # Test log transformation
        sc.pp.log1p(adata)
        assert adata.X.max() > 0

    def test_anndata_scanpy_integration(self):
        """Test AnnData and scanpy work together."""
        import anndata as ad
        import scanpy as sc
        from scipy import sparse
        
        # Create realistic-ish data
        n_obs, n_vars = 50, 100
        X = sparse.csr_matrix(np.random.negative_binomial(5, 0.3, (n_obs, n_vars)))
        
        obs = pd.DataFrame({
            'celltype': np.random.choice(['T_cell', 'B_cell', 'Monocyte'], n_obs),
            'batch': np.random.choice(['batch1', 'batch2'], n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)])
        
        var = pd.DataFrame({
            'gene_name': [f'GENE{i}' for i in range(n_vars)]
        }, index=[f'gene_{i}' for i in range(n_vars)])
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # Run scanpy pipeline
        sc.pp.filter_cells(adata, min_genes=5)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Verify it worked
        assert adata.X.shape[0] > 0
        assert adata.X.shape[1] > 0


class TestMLPackages:
    """Test machine learning and NLP packages."""

    def test_sentence_transformers_import(self):
        """Test sentence-transformers can be imported."""
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import losses
        
        # Just test imports, don't load actual models
        assert SentenceTransformer is not None
        assert losses.MultipleNegativesRankingLoss is not None
        assert losses.TripletLoss is not None
        assert losses.ContrastiveLoss is not None

    def test_huggingface_datasets(self):
        """Test HuggingFace datasets library."""
        from datasets import Dataset
        
        # Create a simple dataset
        ds = Dataset.from_dict({
            'text': ['hello world', 'foo bar', 'test sentence'],
            'label': ['positive', 'negative', 'neutral']
        })
        
        assert len(ds) == 3
        assert 'text' in ds.features
        assert 'label' in ds.features
        assert ds[0]['text'] == 'hello world'

    def test_datasets_operations(self):
        """Test common Dataset operations."""
        from datasets import Dataset
        
        ds = Dataset.from_dict({
            'id': list(range(10)),
            'value': list(range(10, 20))
        })
        
        # Test filtering
        filtered = ds.filter(lambda x: x['id'] % 2 == 0)
        assert len(filtered) == 5
        
        # Test mapping
        mapped = ds.map(lambda x: {'id': x['id'], 'value': x['value'] * 2})
        assert mapped[0]['value'] == 20

    def test_torch_basic(self):
        """Test PyTorch basic operations."""
        import torch
        
        # Create tensors
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        
        # Basic operations
        z = x + y
        assert torch.allclose(z, torch.tensor([5.0, 7.0, 9.0]))
        
        # Test device availability
        assert torch.cuda.is_available() or not torch.cuda.is_available()  # Just checking it doesn't error


class TestDataIntegration:
    """Test integration between different data structures."""

    def test_numpy_to_pandas(self):
        """Test converting numpy arrays to pandas DataFrames."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        df = pd.DataFrame(arr, columns=['a', 'b', 'c'])
        
        assert df.shape == (2, 3)
        assert list(df.columns) == ['a', 'b', 'c']

    def test_pandas_to_datasets(self):
        """Test converting pandas DataFrame to HuggingFace Dataset."""
        from datasets import Dataset
        
        df = pd.DataFrame({
            'text': ['hello', 'world', 'test'],
            'label': [0, 1, 0]
        })
        
        ds = Dataset.from_pandas(df)
        
        assert len(ds) == 3
        assert 'text' in ds.features
        assert ds[0]['text'] == 'hello'

    def test_anndata_to_dataframe(self):
        """Test extracting data from AnnData to DataFrame."""
        import anndata as ad
        from scipy import sparse
        
        X = sparse.csr_matrix(np.array([[1, 2], [3, 4], [5, 6]]))
        obs = pd.DataFrame({'celltype': ['A', 'B', 'A']})
        
        adata = ad.AnnData(X=X, obs=obs)
        
        # Extract obs as DataFrame
        df = adata.obs
        assert isinstance(df, pd.DataFrame)
        assert 'celltype' in df.columns
        
        # Convert X to dense array
        X_dense = adata.X.toarray()
        assert X_dense.shape == (3, 2)


class TestEnvironmentConfiguration:
    """Test environment and configuration setup."""

    def test_hf_config_import(self):
        """Test that hf_config can be imported."""
        from alias.util.hf_config import hf_config
        
        assert hf_config is not None

    def test_hf_config_without_env(self):
        """Test hf_config behavior without .env file."""
        import os
        from alias.util.hf_config import hf_config
        
        # If .env exists, skip this test
        if os.path.exists('.env'):
            pytest.skip(".env file is configured, skipping negative test")
        
        # Should raise informative error when accessing tokens
        with pytest.raises(ValueError, match="not found in environment"):
            _ = hf_config.HF_TOKEN_DOWNLOAD

    def test_dotenv_available(self):
        """Test python-dotenv is available."""
        from dotenv import load_dotenv
        
        # Just test that it's importable
        assert load_dotenv is not None

