"""
Integration tests using real data.

These tests use a real 6.4 MB adata file to test the complete pipeline
with realistic data structures. Marked as 'integration' to run separately
from fast unit tests.
"""

import pytest
from pathlib import Path

from alias.data import (
    DatascRNAConfig,
    TripletGenerationConfig,
    build_datasets,
    build_triplets
)


@pytest.mark.integration
class TestDataPipelineWithRealData:
    """Test data pipeline with realistic data."""

    def test_build_datasets_real_data(self, real_adata):
        """Test building datasets from real scRNA-seq data."""
        # Check what columns are available
        assert 'celltype' in real_adata.obs.columns, \
            f"Available columns: {list(real_adata.obs.columns)}"
        
        config = DatascRNAConfig(
            annotation_column="celltype",
            test_size=0.2,
            preprocessing=False,  # Skip preprocessing for speed
            cs_length=(10,),  # Single length for speed
            highly_variable_genes=False  # Don't require HVG annotation
        )
        
        dataset_dict, adata_test = build_datasets(
            adata=real_adata,
            datasets=['scrna'],
            scrna_config=config
        )
        
        # Verify structure
        assert 'scrna' in dataset_dict
        assert 'data' in dataset_dict['scrna']
        assert 'test' in dataset_dict['scrna']
        
        # Verify data types
        from datasets import Dataset
        assert isinstance(dataset_dict['scrna']['data'], Dataset)
        assert isinstance(dataset_dict['scrna']['test'], Dataset)
        
        # Verify content
        train_size = len(dataset_dict['scrna']['data'])
        test_size = len(dataset_dict['scrna']['test'])
        
        assert train_size > 0, "Training dataset is empty"
        assert test_size > 0, "Test dataset is empty"
        
        # Check split ratio (should be ~80/20)
        total = train_size + test_size
        test_ratio = test_size / total
        assert 0.15 < test_ratio < 0.25, f"Test ratio {test_ratio} not close to 0.2"
        
        # Verify features
        assert 'sentence1' in dataset_dict['scrna']['data'].features
        assert 'label' in dataset_dict['scrna']['data'].features
        
        print(f"\n✓ Built datasets: {train_size} train, {test_size} test samples")

    def test_generate_triplets_real_data(self, real_adata):
        """Test triplet generation with real data."""
        # Build datasets first
        scrna_config = DatascRNAConfig(
            annotation_column="celltype",
            test_size=0.2,
            preprocessing=False,
            cs_length=(10,),
            highly_variable_genes=False
        )
        
        dataset_dict, _ = build_datasets(
            adata=real_adata,
            datasets=['scrna'],
            scrna_config=scrna_config
        )
        
        # Generate triplets
        triplet_config = TripletGenerationConfig(
            annotation_column="celltype",
            loss='MNR',
            eval_split=0.1,
            random_negative_mining=True,
            hard_negative_mining=False,  # Skip for speed
            testrun=False
        )
        
        triplet_dict = build_triplets(
            dataset_dict=dataset_dict,
            triplet_config=triplet_config
        )
        
        # Verify structure
        assert 'scrna' in triplet_dict
        
        # Find train and eval datasets
        train_key = next(k for k in triplet_dict['scrna'].keys() if 'train' in k)
        eval_key = next(k for k in triplet_dict['scrna'].keys() if 'eval' in k)
        
        train_triplets = triplet_dict['scrna'][train_key]
        eval_triplets = triplet_dict['scrna'][eval_key]
        
        assert len(train_triplets) > 0, "No training triplets generated"
        assert len(eval_triplets) > 0, "No evaluation triplets generated"
        
        # Verify triplet structure
        assert 'sentence1' in train_triplets.features  # anchor
        assert 'sentence2' in train_triplets.features  # positive
        assert 'negative' in train_triplets.features   # negative
        
        # Verify content
        example = train_triplets[0]
        assert isinstance(example['sentence1'], str)
        assert isinstance(example['sentence2'], str)
        assert isinstance(example['negative'], str)
        assert len(example['sentence1']) > 0
        
        print(f"\n✓ Generated triplets: {len(train_triplets)} train, {len(eval_triplets)} eval")

    def test_data_integrity_real_data(self, real_adata):
        """Test that real data has expected properties."""
        # Check basic structure
        assert real_adata.shape[0] > 100, "Too few cells for realistic test"
        assert real_adata.shape[1] > 100, "Too few genes for realistic test"
        
        # Check required metadata
        assert 'celltype' in real_adata.obs.columns, \
            f"Missing 'celltype' column. Available: {list(real_adata.obs.columns)}"
        
        # Check cell types
        cell_types = real_adata.obs['celltype'].unique()
        assert len(cell_types) >= 2, "Need at least 2 cell types for triplet generation"
        
        # Check for reasonable cell type distribution
        min_cells_per_type = real_adata.obs['celltype'].value_counts().min()
        assert min_cells_per_type >= 5, f"Some cell types have too few cells (min: {min_cells_per_type})"
        
        print(f"\n✓ Data integrity check passed:")
        print(f"  Shape: {real_adata.shape[0]} cells × {real_adata.shape[1]} genes")
        print(f"  Cell types: {len(cell_types)}")
        print(f"  Min cells per type: {min_cells_per_type}")


@pytest.mark.integration
class TestPipelineEndToEnd:
    """Test complete pipeline with minimal training."""

    def test_minimal_training_pipeline(self, real_adata):
        """
        Test the complete pipeline with minimal training (fast).
        
        Note: This doesn't actually train a model fully, just verifies
        the pipeline can be set up correctly.
        """
        from alias.data import build_datasets, build_triplets
        from alias.model import TrainingSTConfig
        
        # Step 1: Build datasets
        scrna_config = DatascRNAConfig(
            annotation_column="celltype",
            test_size=0.2,
            preprocessing=False,
            cs_length=(10,),
            highly_variable_genes=False
        )
        
        dataset_dict, _ = build_datasets(
            adata=real_adata,
            datasets=['scrna'],
            scrna_config=scrna_config
        )
        
        # Step 2: Generate triplets
        triplet_config = TripletGenerationConfig(
            annotation_column="celltype",
            loss='MNR',
            eval_split=0.1,
            random_negative_mining=True,
            testrun=True  # Limit dataset size
        )
        
        triplet_dict = build_triplets(
            dataset_dict=dataset_dict,
            triplet_config=triplet_config
        )
        
        # Step 3: Verify training config can be created
        training_config = TrainingSTConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            loss='MNR',
            new_model_name="test_model",
            batch_size=8,
            epochs=1,
            save_to_local=True,  # Required for validation
            save_to_hf=False,
            testrun=True
        )
        
        assert training_config is not None
        
        # Verify data is ready for training
        train_key = next(k for k in triplet_dict['scrna'].keys() if 'train' in k)
        train_data = triplet_dict['scrna'][train_key]
        
        assert len(train_data) > 0
        assert all(col in train_data.features for col in ['sentence1', 'sentence2', 'negative'])
        
        print("\n✓ Complete pipeline setup successful")
        print(f"  Training samples: {len(train_data)}")

