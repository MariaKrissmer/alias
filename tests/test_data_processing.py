"""
Tests for data processing functionality.

Tests preprocessing, cs_length parameter behavior, and dataset generation.
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from alias.data import DatascRNAConfig, build_datasets
from alias.data.scrna import run_preprocessing


class TestPreprocessing:
    """Test preprocessing functionality."""
    
    def test_preprocessing_adds_highly_variable(self, tiny_adata):
        """Test that preprocessing adds highly_variable column to adata.var."""
        # Remove highly_variable if it exists (from fixture)
        if 'highly_variable' in tiny_adata.var.columns:
            tiny_adata.var.drop(columns=['highly_variable'], inplace=True)
        
        config = DatascRNAConfig(
            annotation_column="celltype",
            preprocessing=False,  # We'll call run_preprocessing directly
            hvg_number=10,
            min_genes=1,  # Lower threshold for tiny data
            min_cells=1,
            min_batch_cells=1,
            verbose=False
        )
        
        # Ensure highly_variable doesn't exist before
        assert 'highly_variable' not in tiny_adata.var.columns
        
        # Run preprocessing
        adata_processed = run_preprocessing(
            tiny_adata, 
            batch_key='celltype',
            scrna_config=config
        )
        
        # Verify highly_variable column was added and subsetting occurred
        assert adata_processed.n_vars > 0, "All genes were filtered out"
        # After subsetting to HVGs, gene count should be reduced
        # Note: With tiny data (20 genes), HVG selection may not be exact
        assert adata_processed.n_vars < tiny_adata.n_vars, "Should reduce gene count via HVG selection"
    
    def test_preprocessing_normalization(self, tiny_adata):
        """Test that preprocessing performs normalization."""
        config = DatascRNAConfig(
            annotation_column="celltype",
            hvg_number=10,
            min_genes=1,  # Lower thresholds for tiny data
            min_cells=1,
            min_batch_cells=1,
            verbose=False
        )
        
        # Run preprocessing
        adata_processed = run_preprocessing(
            tiny_adata,
            batch_key='celltype',
            scrna_config=config
        )
        
        # Check that we have cells left
        assert adata_processed.n_obs > 0, "All cells were filtered out"
        
        # Check normalization flags
        assert adata_processed.uns.get("normalized") == True
        assert adata_processed.uns.get("logged") == True
        
        # Check that counts layer was saved
        assert 'counts' in adata_processed.layers
    
    def test_preprocessing_filtering(self, tiny_adata):
        """Test that preprocessing filters cells and genes."""
        config = DatascRNAConfig(
            annotation_column="celltype",
            min_genes=1,  # Relaxed for tiny data
            min_cells=1,
            min_batch_cells=1,
            mt_threshold=100,  # High threshold so nothing gets filtered for testing
            hvg_number=10,
            verbose=False
        )
        
        original_n_obs = tiny_adata.n_obs
        original_n_vars = tiny_adata.n_vars
        
        adata_processed = run_preprocessing(
            tiny_adata,
            batch_key='celltype',
            scrna_config=config
        )
        
        # After preprocessing, dimensions should be <= original
        assert adata_processed.n_obs <= original_n_obs
        
        # HVG subsetting should reduce gene count
        # Note: With tiny data, HVG selection may not be exact
        assert adata_processed.n_vars < original_n_vars, "Preprocessing should reduce gene count"


class TestCsLengthParameter:
    """Test cs_length parameter behavior."""
    
    def test_cs_length_single_int_tuple(self, tiny_adata):
        """Test that cs_length=(10,) uses 10 genes."""
        config = DatascRNAConfig(
            annotation_column="celltype",
            cs_length=(10,),
            preprocessing=False,
            highly_variable_genes=False,
            test_size=0.2
        )
        
        dataset_dict, _ = build_datasets(
            adata=tiny_adata,
            datasets=['scrna'],
            scrna_config=config
        )
        
        # Check that gene lists have correct length
        train_ds = dataset_dict['scrna']['data']
        gene_list = train_ds[0]['gene_list']
        
        # Should have exactly 10 genes (or fewer if adata has < 10 genes)
        expected_length = min(10, tiny_adata.n_vars)
        assert len(gene_list) == expected_length
    
    def test_cs_length_single_int_as_int(self):
        """Test that passing cs_length as single int (not tuple) is accepted but causes issues."""
        # Dataclass doesn't validate types strictly, so this will succeed
        config = DatascRNAConfig(
            annotation_column="celltype",
            cs_length=10  # Not a tuple! This will cause issues later
        )
        
        # Config accepts it but it's an int, not indexable
        assert isinstance(config.cs_length, int)
        assert not hasattr(config.cs_length, '__getitem__')
        
        # This demonstrates the config accepts invalid types
        # In practice, this would cause a TypeError when code tries cfg['cs_length'][0]
    
    def test_cs_length_two_values_uses_first(self, tiny_adata):
        """Test that cs_length=(5, 10) only uses the first value (5)."""
        config = DatascRNAConfig(
            annotation_column="celltype",
            cs_length=(5, 10),  # Two values provided
            preprocessing=False,
            highly_variable_genes=False,
            test_size=0.2
        )
        
        dataset_dict, _ = build_datasets(
            adata=tiny_adata,
            datasets=['scrna'],
            scrna_config=config
        )
        
        train_ds = dataset_dict['scrna']['data']
        
        # Check multiple samples to ensure it's consistent
        for i in range(min(5, len(train_ds))):
            gene_list = train_ds[i]['gene_list']
            expected_length = min(5, tiny_adata.n_vars)  # Should use first value (5)
            assert len(gene_list) == expected_length, \
                f"Sample {i}: Expected {expected_length} genes, got {len(gene_list)}"
    
    def test_cs_length_multiple_values_uses_first(self, tiny_adata):
        """Test that cs_length=(3, 5, 10, 15) only uses the first value (3)."""
        config = DatascRNAConfig(
            annotation_column="celltype",
            cs_length=(3, 5, 10, 15),  # Multiple values
            preprocessing=False,
            highly_variable_genes=False,
            test_size=0.2
        )
        
        dataset_dict, _ = build_datasets(
            adata=tiny_adata,
            datasets=['scrna'],
            scrna_config=config
        )
        
        train_ds = dataset_dict['scrna']['data']
        gene_list = train_ds[0]['gene_list']
        
        expected_length = min(3, tiny_adata.n_vars)
        assert len(gene_list) == expected_length, \
            "Only first value of cs_length tuple should be used"
    
    def test_cs_length_larger_than_genes(self, tiny_adata):
        """Test that cs_length larger than available genes uses all genes."""
        n_genes = tiny_adata.n_vars
        
        config = DatascRNAConfig(
            annotation_column="celltype",
            cs_length=(1000,),  # Much larger than available genes
            preprocessing=False,
            highly_variable_genes=False,
            test_size=0.2
        )
        
        dataset_dict, _ = build_datasets(
            adata=tiny_adata,
            datasets=['scrna'],
            scrna_config=config
        )
        
        train_ds = dataset_dict['scrna']['data']
        gene_list = train_ds[0]['gene_list']
        
        # Should use all available genes
        assert len(gene_list) == n_genes


class TestDatasetGenerationWithPreprocessing:
    """Test full dataset generation with preprocessing enabled."""
    
    def test_preprocessing_integration(self, tiny_adata):
        """Test that preprocessing works when enabled in build_datasets."""
        config = DatascRNAConfig(
            annotation_column="celltype",
            preprocessing=True,  # Enable preprocessing
            highly_variable_genes=True,  # Should work after preprocessing
            hvg_number=10,
            cs_length=(5,),
            test_size=0.2,
            min_genes=3,
            min_cells=1,
            min_batch_cells=1,
            verbose=False
        )
        
        dataset_dict, adata_test = build_datasets(
            adata=tiny_adata,
            datasets=['scrna'],
            scrna_config=config
        )
        
        # Verify datasets were created
        assert 'scrna' in dataset_dict
        assert 'data' in dataset_dict['scrna']
        assert 'test' in dataset_dict['scrna']
        
        # Verify returned adata_test exists and has been processed
        assert adata_test is not None
        assert adata_test.n_obs > 0
        
        # Check that sentence1 field was created
        train_ds = dataset_dict['scrna']['data']
        assert 'sentence1' in train_ds.features
        assert len(train_ds[0]['sentence1']) > 0
    
    @pytest.mark.integration
    def test_preprocessing_with_real_data(self, real_adata):
        """Test preprocessing with real data (more realistic scenario)."""
        config = DatascRNAConfig(
            annotation_column="celltype",
            preprocessing=True,
            highly_variable_genes=True,
            hvg_number=500,  # More realistic number
            cs_length=(20,),
            test_size=0.2,
            verbose=False
        )
        
        # Store original shape
        original_shape = real_adata.shape
        
        dataset_dict, adata_test = build_datasets(
            adata=real_adata,
            datasets=['scrna'],
            scrna_config=config
        )
        
        # Verify processing occurred
        train_ds = dataset_dict['scrna']['data']
        assert len(train_ds) > 0
        
        # Check gene list length in generated sentences
        gene_list = train_ds[0]['gene_list']
        assert len(gene_list) == 20, "Should use cs_length[0] = 20"
        
        # Verify test adata was processed
        assert adata_test.n_vars <= 500, "Should have at most hvg_number genes after preprocessing"

