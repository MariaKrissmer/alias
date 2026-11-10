from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/revision1_v1/HIAI_Tcells/embeddings_evaluation/eval_embedding_comparison.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("hiai_scib_release_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScibReleaseSourcesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()

    def test_cached_scores_require_matching_model_source(self) -> None:
        runner = self.runner
        spec = next(item for item in runner.EMBEDDING_SPECS if item.name == "MB")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            old_scores = runner.SCIB_AGGREGATE_SCORES_PATH
            old_mapping = runner.EMBEDDING_MAPPING_PATH
            old_force = runner.FORCE_SCIB_SCORES
            try:
                runner.SCIB_AGGREGATE_SCORES_PATH = temp_path / "scib_aggregate_scores.csv"
                runner.EMBEDDING_MAPPING_PATH = temp_path / "embedding_obsm_keys.csv"
                runner.FORCE_SCIB_SCORES = False
                pd.DataFrame([{
                    "model_name": "MB",
                    "include_ablation": True,
                    "include_benchmark": True,
                    "Batch correction": 0.4,
                    "Bio conservation": 0.5,
                    "Total": 0.45,
                }]).to_csv(runner.SCIB_AGGREGATE_SCORES_PATH, index=False)
                mapping = pd.DataFrame([{
                    "model_name": "MB",
                    "kind": "sentence_transformer",
                    "model": "mariakrissmer/MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15",
                    "external_path": "",
                }])
                mapping.to_csv(runner.EMBEDDING_MAPPING_PATH, index=False)
                self.assertIsNone(runner._load_cached_scib_scores_if_complete([spec]))

                mapping.loc[0, "model"] = spec.model
                mapping.to_csv(runner.EMBEDDING_MAPPING_PATH, index=False)
                self.assertEqual(len(runner._load_cached_scib_scores_if_complete([spec])), 1)
            finally:
                runner.SCIB_AGGREGATE_SCORES_PATH = old_scores
                runner.EMBEDDING_MAPPING_PATH = old_mapping
                runner.FORCE_SCIB_SCORES = old_force

    def test_sentence_transformer_reuses_annotation_cells_in_scib_order(self) -> None:
        runner = self.runner
        spec = next(item for item in runner.EMBEDDING_SPECS if item.name == "MB")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            analysis_embedding_root = root / "analysis_runs/annotation_embedding/current/embeddings"
            run_dir = analysis_embedding_root / "MB/embeddings/scrna/ALIASMA/2026-11-09T09-57-31"
            run_dir.mkdir(parents=True)
            embedding_path = run_dir / "df_cells.parquet"
            pd.DataFrame(
                {"0": [1.0, 3.0], "1": [2.0, 4.0]},
                index=pd.Index(["c1", "c2"], name="cell_id"),
            ).to_parquet(embedding_path)
            (run_dir / "metadata.json").write_text(json.dumps({
                "dataset_name": "scrna",
                "model_name": "ALIASMA",
                "reuse_config": {
                    "dataset_name": "scrna",
                    "model_name": "ALIASMA",
                    "annotation_column": "AIFI_L2",
                    "model_type": "sentence_transformer",
                    "batch_size": runner.BATCH_SIZE,
                    "max_cells": runner.MAX_CELLS,
                    "index": True,
                    "additional_data": [],
                },
                "artifacts": {"df_cells": {"path": str(embedding_path)}},
            }), encoding="utf-8")
            annotation_dir = root / "celltype_annotation/MB"
            annotation_dir.mkdir(parents=True)
            (annotation_dir / "metadata.json").write_text(
                json.dumps({"model": spec.model}), encoding="utf-8"
            )
            adata = ad.AnnData(
                X=np.zeros((2, 1)),
                obs=pd.DataFrame(index=["c2", "c1"]),
            )
            scrna_rows = pd.DataFrame({"index": ["c2", "c1"], "sentence1": ["x", "y"]})
            with (
                patch.object(runner, "DATASET_DIR", root),
                patch.object(runner, "EMBEDDING_ROOT", analysis_embedding_root, create=True),
                patch.object(runner, "ANNOTATION_DIR", root / "celltype_annotation"),
                patch.object(runner, "REQUIRE_CACHED_EMBEDDINGS", True),
                patch.object(runner.ev, "generate_embeddings", side_effect=AssertionError("encoded again")),
            ):
                updated, source_path = runner._append_sentence_transformer_embedding(
                    adata, scrna_rows, spec
                )
            np.testing.assert_array_equal(
                updated.obsm["MB"], np.array([[3.0, 4.0], [1.0, 2.0]])
            )
            self.assertEqual(source_path, embedding_path)

    def test_cache_only_mode_fails_if_annotation_embedding_is_missing(self) -> None:
        runner = self.runner
        spec = next(item for item in runner.EMBEDDING_SPECS if item.name == "MB")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adata = ad.AnnData(X=np.zeros((1, 1)), obs=pd.DataFrame(index=["c1"]))
            rows = pd.DataFrame({"index": ["c1"], "sentence1": ["x"]})
            with (
                patch.object(runner, "DATASET_DIR", root),
                patch.object(runner, "ANNOTATION_DIR", root / "celltype_annotation"),
                patch.object(runner, "REQUIRE_CACHED_EMBEDDINGS", True),
                patch.object(runner.ev, "generate_embeddings", side_effect=AssertionError("encoded again")),
            ):
                with self.assertRaises(FileNotFoundError):
                    runner._append_sentence_transformer_embedding(adata, rows, spec)


if __name__ == "__main__":
    unittest.main()
