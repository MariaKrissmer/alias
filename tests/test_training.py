"""
Tests for training functionality.

Tests that training works with both string and list for datasets parameter.
"""

import pytest
from datasets import Dataset

from alias.model.training import (
    TrainingSTConfig,
    _learning_rate_for_outer_epoch,
    _lr_scheduler_type_for_outer_epoch,
    _warmup_steps_for_outer_epoch,
)


def test_outer_linear_learning_rate_schedule_uses_global_epoch_progress():
    config = TrainingSTConfig(
        model="neuml/pubmedbert-base-embeddings",
        loss="MNR",
        epochs=5,
        learning_rate=5e-5,
        outer_learning_rate_schedule="linear",
        min_learning_rate=5e-6,
        warmup_steps=1000,
        warmup_first_epoch_only=True,
    )

    assert _learning_rate_for_outer_epoch(config, epoch_index=0) == pytest.approx(5e-5)
    assert _learning_rate_for_outer_epoch(config, epoch_index=2) == pytest.approx(2.75e-5)
    assert _learning_rate_for_outer_epoch(config, epoch_index=4) == pytest.approx(5e-6)
    assert _warmup_steps_for_outer_epoch(config, epoch_index=0) == 1000
    assert _warmup_steps_for_outer_epoch(config, epoch_index=1) == 0
    assert _lr_scheduler_type_for_outer_epoch(config, epoch_index=0) == "linear"
    assert _lr_scheduler_type_for_outer_epoch(config, epoch_index=1) == "linear"


def test_explicit_outer_learning_rate_schedule_uses_configured_epoch_rates():
    config = TrainingSTConfig(
        model="neuml/pubmedbert-base-embeddings",
        loss="MNR",
        epochs=3,
        learning_rate=5e-5,
        outer_learning_rate_schedule="explicit",
        epoch_learning_rates=[5e-5, 2e-5, 1e-5],
    )

    assert _learning_rate_for_outer_epoch(config, epoch_index=0) == pytest.approx(5e-5)
    assert _learning_rate_for_outer_epoch(config, epoch_index=1) == pytest.approx(2e-5)
    assert _learning_rate_for_outer_epoch(config, epoch_index=2) == pytest.approx(1e-5)


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
