"""
Tests for hard negative mining functionality.
"""

import pytest
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from alias.util.hard_negative_mining import mine_hard_negatives


class TestMineHardNegatives:
    """Test hard negative mining functionality."""
    
    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset with anchor-positive pairs."""
        return Dataset.from_dict({
            'sentence1': [
                'The T cell is a type of lymphocyte.',
                'B cells produce antibodies.',
                'NK cells are part of innate immunity.',
                'Macrophages engulf pathogens.',
                'Dendritic cells present antigens.',
            ],
            'sentence2': [
                'T lymphocytes play a key role in adaptive immunity.',
                'B lymphocytes secrete immunoglobulins.',
                'Natural killer cells destroy infected cells.',
                'Phagocytes consume foreign particles.',
                'Antigen-presenting cells activate T cells.',
            ],
            'label': ['T cell', 'B cell', 'NK cell', 'Macrophage', 'Dendritic cell'],
        })
    
    @pytest.fixture
    def small_model(self):
        """Load a small sentence transformer model for testing."""
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def test_mine_hard_negatives_column_copy_error(self, simple_dataset, small_model):
        """
        Test that mine_hard_negatives works with HuggingFace Dataset.
        
        This test reproduces the bug where dataset[column_name] returns a Column
        object that doesn't have a .copy() method, causing:
        AttributeError: 'Column' object has no attribute 'copy'
        
        The issue is at line 808 in hard_negative_mining.py:
            all_queries = queries.copy()
        
        Where queries = dataset[anchor_column_name] returns a Column, not a list.
        """
        # This should work but currently fails with:
        # AttributeError: 'Column' object has no attribute 'copy'
        result = mine_hard_negatives(
            dataset=simple_dataset,
            model=small_model,
            anchor_column_name='sentence1',
            positive_column_name='sentence2',
            label_column_name='label',
            num_negatives=2,
            sampling_strategy="top",
            batch_size=32,
            use_faiss=False,
            verbose=False,
            range_max=10,
        )
        
        # If we get here, the bug is fixed
        assert result is not None
        assert len(result) > 0
    
    def test_mine_hard_negatives_with_subset(self, simple_dataset, small_model):
        """
        Test mine_hard_negatives with a subset of the dataset.
        
        This mimics how build_triplets uses it with dataset.select().
        """
        # Select a subset like build_triplets does
        subset = simple_dataset.select(range(3))
        
        result = mine_hard_negatives(
            dataset=subset,
            model=small_model,
            anchor_column_name='sentence1',
            positive_column_name='sentence2',
            label_column_name='label',
            num_negatives=2,
            sampling_strategy="top",
            batch_size=32,
            use_faiss=False,
            verbose=False,
            range_max=10,
        )
        
        assert result is not None
        assert len(result) > 0

