"""
Smoke tests to verify that all modules and public APIs can be imported correctly.

These tests serve several purposes:
1. Verify all dependencies are properly installed
2. Catch circular import issues early
3. Validate the public API is accessible as documented
4. Provide quick sanity checks for CI/CD pipelines
"""

import pytest


class TestDataPackageImports:
    """Test imports from the data subpackage."""

    def test_import_data_package(self):
        """Test that the data package can be imported."""
        from alias import data

    def test_import_data_public_api(self):
        """Test that all public API items from data package are importable."""
        from alias.data import (
            build_datasets,
            build_triplets,
            DatascRNAConfig,
            DataNCBIConfig,
            TripletGenerationConfig,
        )

    def test_import_data_build_datasets(self):
        """Test that build_datasets module can be imported."""
        from alias.data import build_datasets

    def test_import_data_scrna(self):
        """Test that scrna module can be imported."""
        from alias.data import scrna

    def test_import_data_ncbi(self):
        """Test that ncbi module can be imported."""
        from alias.data import ncbi

    def test_import_data_triplet_generation(self):
        """Test that triplet_generation module can be imported."""
        from alias.data import triplet_generation


class TestModelPackageImports:
    """Test imports from the model subpackage."""

    def test_import_model_package(self):
        """Test that the model package can be imported."""
        from alias import model

    def test_import_model_public_api(self):
        """Test that all public API items from model package are importable."""
        from alias.model import train_model, TrainingSTConfig

    def test_import_model_training(self):
        """Test that training module can be imported."""
        from alias.model import training


class TestUtilPackageImports:
    """Test imports from the util subpackage."""

    def test_import_util_package(self):
        """Test that the util package can be imported."""
        from alias import util

    def test_import_util_cell_sentence_templates(self):
        """Test that cell_sentence_templates module can be imported."""
        from alias.util import cell_sentence_templates

    def test_import_util_hard_negative_mining(self):
        """Test that hard_negative_mining module can be imported."""
        from alias.util import hard_negative_mining

    def test_import_util_load_hf_model(self):
        """Test that load_hf_model module can be imported."""
        from alias.util import load_hf_model

    def test_import_util_similarity(self):
        """Test that similarity module can be imported."""
        from alias.util import similarity

    def test_import_util_similarity_cellwhisperer(self):
        """Test that similarity_cellwhisperer module can be imported."""
        from alias.util import similarity_cellwhisperer

    def test_import_util_hf_config(self):
        """Test that hf_config module can be imported."""
        from alias.util import hf_config


class TestUtilPlotsImports:
    """Test imports from the util.plots subpackage."""

    def test_import_util_plots_package(self):
        """Test that the util.plots package can be imported."""
        from alias.util import plots

    def test_import_util_plots_color_definition(self):
        """Test that color_definition module can be imported."""
        from alias.util.plots import color_definition

    def test_import_util_plots_plot_colors(self):
        """Test that plot_colors module can be imported."""
        from alias.util.plots import plot_colors

    def test_import_util_plots_pub_style(self):
        """Test that pub_style module can be imported."""
        from alias.util.plots import pub_style

    def test_import_util_plots_umap_plots(self):
        """Test that umap_plots module can be imported."""
        from alias.util.plots import umap_plots


class TestEvaluationImports:
    """Test imports from the evaluation subpackage."""

    def test_import_evaluation_package(self):
        """Test that the evaluation package can be imported."""
        from alias import evaluation

    def test_import_evaluation_celltype_label_similarity(self):
        """Test that celltype_label_similarity module can be imported."""
        from alias.evaluation import celltype_label_similarity

    def test_import_evaluation_embedding(self):
        """Test that embedding module can be imported."""
        from alias.evaluation import embedding

    def test_import_evaluation_functionality_cell_similarity(self):
        """Test that functionality_cell_similarity module can be imported."""
        from alias.evaluation import functionality_cell_similarity

    def test_import_evaluation_functionality_cell_similarity_cellwhisperer(self):
        """Test that functionality_cell_similarity_cellwhisperer module can be imported."""
        from alias.evaluation import functionality_cell_similarity_cellwhisperer

    # Note: celltype_label.plots.py has a dot in the name which makes direct import tricky
    # This would typically be renamed to celltype_label_plots.py for better Python conventions


class TestNoCircularImports:
    """Test that there are no circular import issues."""

    def test_import_all_packages_sequentially(self):
        """Import all main packages to ensure no circular dependencies."""
        from alias import data, model, util, evaluation

        # Verify they're all imported
        assert data is not None
        assert model is not None
        assert util is not None
        assert evaluation is not None


# Optionally, test that __all__ matches what's actually importable
class TestPublicAPIConsistency:
    """Test that __all__ declarations are consistent with actual exports."""

    def test_data_all_consistency(self):
        """Verify data.__all__ matches actual exports."""
        from alias import data

        if hasattr(data, "__all__"):
            for name in data.__all__:
                assert hasattr(data, name), f"{name} in __all__ but not found in module"

    def test_model_all_consistency(self):
        """Verify model.__all__ matches actual exports."""
        from alias import model

        if hasattr(model, "__all__"):
            for name in model.__all__:
                assert hasattr(model, name), f"{name} in __all__ but not found in module"
