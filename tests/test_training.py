"""
Tests for training functionality.

Tests that training works with both string and list for datasets parameter.
"""

import pytest
from datasets import Dataset


class TestTrainingWithDatasets:
    """Test that training works with both string and list for datasets parameter."""
    
    def test_train_with_datasets_as_string(self):
        """Test that datasets='scrna' works."""
        from alias.model import TrainingSTConfig, train_model
        
        # Create minimal triplet dataset
        triplet_data = Dataset.from_dict({
            'sentence1': ['Cell with genes A B C'] * 10,
            'sentence2': ['Cell expressing A B C'] * 10,
            'negative': ['Different cell with X Y Z'] * 10
        })
        
        dataset_dict = {
            'scrna': {
                'scrna_train': triplet_data,
                'scrna_eval': triplet_data.select(range(5))
            }
        }
        
        config = TrainingSTConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            loss='MNR',
            new_model_name="test_model_string",
            batch_size=2,
            epochs=1,
            save_to_local=True,
            save_to_hf=False,
            testrun=True,
            load_from_hf=False
        )
        
        # Should work with string
        trained_model = train_model(
            dataset_dict=dataset_dict,
            datasets='scrna',  # String
            train_config=config
        )
        assert trained_model is not None
    
    def test_train_with_datasets_as_list(self):
        """Test that datasets=['scrna'] works."""
        from alias.model import TrainingSTConfig, train_model
        
        # Create minimal triplet dataset
        triplet_data = Dataset.from_dict({
            'sentence1': ['Cell with genes A B C'] * 10,
            'sentence2': ['Cell expressing A B C'] * 10,
            'negative': ['Different cell with X Y Z'] * 10
        })
        
        dataset_dict = {
            'scrna': {
                'scrna_train': triplet_data,
                'scrna_eval': triplet_data.select(range(5))
            }
        }
        
        config = TrainingSTConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            loss='MNR',
            new_model_name="test_model_list",
            batch_size=2,
            epochs=1,
            save_to_local=True,
            save_to_hf=False,
            testrun=True,
            load_from_hf=False
        )
        
        # Should work with list
        trained_model = train_model(
            dataset_dict=dataset_dict,
            datasets=['scrna'],  # List
            train_config=config
        )
        assert trained_model is not None

