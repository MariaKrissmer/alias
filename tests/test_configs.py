"""
Test configuration dataclasses for validation and defaults.

These tests verify that configuration classes have sensible defaults
and properly validate their inputs.
"""

import pytest
from alias.data import DatascRNAConfig, DataNCBIConfig, TripletGenerationConfig
from alias.model import TrainingSTConfig


class TestDatascRNAConfig:
    """Test DatascRNAConfig dataclass."""

    def test_default_values(self):
        """Test DatascRNAConfig has sensible defaults."""
        config = DatascRNAConfig()
        
        assert config.annotation_column == "celltype"
        assert config.preprocessing is False
        assert config.random_state == 42
        assert config.test_size == 0.1
        assert config.hvg_number == 3000
        assert config.min_genes == 100
        assert config.highly_variable_genes is True

    def test_custom_values(self):
        """Test DatascRNAConfig accepts custom values."""
        config = DatascRNAConfig(
            annotation_column="cell_type",
            test_size=0.2,
            random_state=123,
            hvg_number=2000
        )
        
        assert config.annotation_column == "cell_type"
        assert config.test_size == 0.2
        assert config.random_state == 123
        assert config.hvg_number == 2000

    def test_optional_columns(self):
        """Test optional column configurations."""
        config = DatascRNAConfig(
            disease_column="disease",
            time_column="timepoint"
        )
        
        assert config.disease_column == "disease"
        assert config.time_column == "timepoint"

    def test_template_weights(self):
        """Test template weight parameters."""
        config = DatascRNAConfig(
            template_weights_default={"template1": 0.5, "template2": 0.5}
        )
        
        assert config.template_weights_default is not None
        assert isinstance(config.template_weights_default, dict)


class TestDataNCBIConfig:
    """Test DataNCBIConfig dataclass."""

    def test_creation(self):
        """Test DataNCBIConfig can be instantiated with required email."""
        config = DataNCBIConfig(email="test@example.com")
        assert config is not None
        assert config.email == "test@example.com"

    def test_attributes_exist(self):
        """Test DataNCBIConfig has expected attributes."""
        config = DataNCBIConfig(email="test@example.com")
        
        # Check that config is a dataclass with some attributes
        assert hasattr(config, '__dataclass_fields__')
        assert hasattr(config, 'email')


class TestTripletGenerationConfig:
    """Test TripletGenerationConfig dataclass."""

    def test_default_values(self):
        """Test TripletGenerationConfig has sensible defaults."""
        config = TripletGenerationConfig(annotation_column="celltype")
        
        assert config.annotation_column == "celltype"
        assert config.eval_split == 0.1
        assert config.seed == 42
        assert config.loss in ['MNR', 'Triplet', 'Contrastiv']
        assert config.testrun is False

    def test_loss_types(self):
        """Test different loss type configurations."""
        for loss_type in ['MNR', 'Triplet', 'Contrastiv']:
            config = TripletGenerationConfig(
                annotation_column="celltype",
                loss=loss_type
            )
            assert config.loss == loss_type

    def test_mining_flags(self):
        """Test hard negative and random negative mining flags."""
        config_hnm = TripletGenerationConfig(
            annotation_column="celltype",
            hard_negative_mining=True
        )
        assert config_hnm.hard_negative_mining is True
        
        config_rnm = TripletGenerationConfig(
            annotation_column="celltype",
            random_negative_mining=True
        )
        assert config_rnm.random_negative_mining is True

    def test_hf_upload_config(self):
        """Test HuggingFace upload configuration."""
        config = TripletGenerationConfig(
            annotation_column="celltype",
            hf_upload=True,
            hf_name="test_dataset"
        )
        
        assert config.hf_upload is True
        assert config.hf_name == "test_dataset"


class TestTrainingSTConfig:
    """Test TrainingSTConfig dataclass."""

    def test_valid_config(self):
        """Test creating a valid TrainingSTConfig."""
        config = TrainingSTConfig(
            model="test-model",
            loss="MNR",
            save_to_local=True
        )
        
        assert config.model == "test-model"
        assert config.loss == "MNR"
        assert config.save_to_local is True
        assert config.save_to_hf is False

    def test_default_hyperparameters(self):
        """Test default training hyperparameters."""
        config = TrainingSTConfig(
            model="test-model",
            loss="MNR",
            save_to_local=True
        )
        
        assert config.batch_size == 64
        assert config.epochs == 5
        assert config.warmup_steps == 1000
        assert config.weight_decay == 0.01
        assert config.seed == 73
        assert config.fp16 is False

    def test_validation_save_options(self):
        """Test that at least one save option must be enabled."""
        with pytest.raises(ValueError, match="at least one of"):
            TrainingSTConfig(
                model="test-model",
                loss="MNR",
                save_to_local=False,
                save_to_hf=False
            )

    def test_both_save_options_valid(self):
        """Test that both save options can be enabled."""
        config = TrainingSTConfig(
            model="test-model",
            loss="MNR",
            save_to_local=True,
            save_to_hf=True
        )
        
        assert config.save_to_local is True
        assert config.save_to_hf is True

    def test_loss_types(self):
        """Test different loss types."""
        for loss_type in ['MNR', 'Triplet', 'Contrastive']:
            config = TrainingSTConfig(
                model="test-model",
                loss=loss_type,
                save_to_local=True
            )
            assert config.loss == loss_type

    def test_matryoshka_config(self):
        """Test Matryoshka loss configuration."""
        config = TrainingSTConfig(
            model="test-model",
            loss="MNR",
            save_to_local=True,
            matryoshka=[768, 512, 256, 128, 64]
        )
        
        assert config.matryoshka is not None
        assert len(config.matryoshka) == 5
        assert config.matryoshka[0] == 768

    def test_testrun_flag(self):
        """Test testrun flag for quick testing."""
        config = TrainingSTConfig(
            model="test-model",
            loss="MNR",
            save_to_local=True,
            testrun=True
        )
        
        assert config.testrun is True

