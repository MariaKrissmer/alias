from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "alias"
    / "evaluation"
    / "functionality_downstream_assessment.py"
)


def load_downstream_module(monkeypatch):
    calls = {"score_genes": [], "rank_genes_groups": []}

    fake_scanpy = types.ModuleType("scanpy")
    fake_scanpy.read_h5ad = lambda path: None
    fake_scanpy.settings = types.SimpleNamespace(figdir=".")
    fake_scanpy.tl = types.SimpleNamespace(
        score_genes=lambda adata, gene_list, score_name, **kwargs: calls["score_genes"].append(
            (score_name, list(gene_list))
        ),
        rank_genes_groups=lambda adata, groupby, reference, method: calls["rank_genes_groups"].append(
            (groupby, reference, method)
        ),
    )
    fake_scanpy.get = types.SimpleNamespace(
        rank_genes_groups_df=lambda adata, group: pd.DataFrame(
            {
                "names": ["NKG7", "GZMA", "CTSW", "CST7", "CCL5", "GZMB", "PRF1"],
                "logfoldchanges": [1.5, 1.2, 0.9, 0.6, 0.3, 0.2, -0.1],
                "pvals_adj": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1],
            }
        )
    )
    fake_scanpy.pl = types.SimpleNamespace(rank_genes_groups_heatmap=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    spec = importlib.util.spec_from_file_location(
        "alias.evaluation.functionality_downstream_assessment",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module._scanpy_calls = calls
    return module


class FakeAdata:
    def __init__(self, obs: pd.DataFrame, var_names: list[str] | None = None):
        self.obs = obs
        self.var_names = pd.Index(var_names or ["NKG7", "GZMA", "CTSW", "CST7", "CCL5", "GZMB", "PRF1"])
        self.X = np.tile(np.arange(1, len(self.var_names) + 1, dtype=float), (len(obs), 1))
        self.obsm = {"X_umap": np.arange(len(obs) * 2, dtype=float).reshape(len(obs), 2)}
        self.uns = {}
        self.obs_names = obs.index

    def copy(self):
        copied = FakeAdata(self.obs.copy(), self.var_names.astype(str).tolist())
        copied.X = self.X.copy()
        copied.obsm = {key: value.copy() for key, value in self.obsm.items()}
        copied.uns = dict(self.uns)
        return copied

    def __getitem__(self, mask):
        if isinstance(mask, tuple):
            mask = mask[0]
        if hasattr(mask, "values"):
            mask = mask.values
        subset = FakeAdata(self.obs.loc[self.obs.index[mask]].copy(), self.var_names.astype(str).tolist())
        subset.X = self.X[mask].copy()
        subset.obsm = {key: value[mask].copy() for key, value in self.obsm.items()}
        return subset


def test_youden_j_requires_ground_truth(monkeypatch):
    module = load_downstream_module(monkeypatch)

    with pytest.raises(ValueError, match="ground_truth"):
        module.compute_cutoff(
            np.array([0.1, 0.9]),
            method="youden_j",
            functionality_label="cytotoxic",
        )


def test_youden_j_separates_labeled_distribution(monkeypatch):
    module = load_downstream_module(monkeypatch)

    result = module.compute_cutoff(
        np.array([0.1, 0.2, 0.8, 0.9]),
        method="youden_j",
        functionality_label="cytotoxic",
        ground_truth=np.array([False, False, True, True]),
    )

    assert 0.2 <= result.cutoff <= 0.8
    assert result.method == "youden_j"


def test_manual_requires_and_uses_configured_cutoff(monkeypatch):
    module = load_downstream_module(monkeypatch)

    with pytest.raises(ValueError, match="manual"):
        module.compute_cutoff(np.array([0.1, 0.9]), method="manual", functionality_label="cytotoxic")

    result = module.compute_cutoff(
        np.array([0.1, 0.9]),
        method="manual",
        functionality_label="cytotoxic",
        manual_cutoffs={"cytotoxic": 0.42},
    )
    assert result.cutoff == 0.42


def test_otsu_and_gmm_separate_bimodal_distribution(monkeypatch):
    module = load_downstream_module(monkeypatch)
    scores = np.array([0.05, 0.1, 0.15, 0.8, 0.85, 0.9])

    otsu = module.compute_cutoff(scores, method="otsu", functionality_label="cytotoxic")
    gmm = module.compute_cutoff(scores, method="gmm", functionality_label="cytotoxic")

    assert 0.15 < otsu.cutoff < 0.8
    assert 0.15 < gmm.cutoff < 0.8


def test_cutoff_sensitivity_can_use_configured_threshold_bounds(monkeypatch):
    module = load_downstream_module(monkeypatch)
    df_sim = pd.DataFrame({"score": [0.1, 0.2, 0.4, 0.8]})

    sensitivity = module._cutoff_sensitivity_table(
        df_sim,
        cutoff=0.4,
        window_fraction=0.1,
        steps=5,
        threshold_min=0.0,
        threshold_max=0.6,
    )

    assert sensitivity["threshold"].min() == 0.0
    assert sensitivity["threshold"].max() == 0.6
    assert 0.4 in sensitivity["threshold"].round(6).tolist()


def test_cutoff_sensitivity_threshold_max_below_cutoff_extends_downward(monkeypatch):
    module = load_downstream_module(monkeypatch)
    df_sim = pd.DataFrame({"score": [0.4, 0.6, 0.7, 0.8, 0.9]})

    sensitivity = module._cutoff_sensitivity_table(
        df_sim,
        cutoff=0.75,
        window_fraction=0.1,
        steps=5,
        threshold_max=0.6,
    )

    rounded_thresholds = sensitivity["threshold"].round(6).tolist()
    assert 0.6 in rounded_thresholds
    assert 0.75 in rounded_thresholds
    assert sensitivity["threshold"].min() <= 0.6
    assert sensitivity["threshold"].max() >= 0.75


def test_run_downstream_assessment_writes_artifacts_and_calls_gene_set_and_deg(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)

    obs = pd.DataFrame(
        {"celltype": ["CD8", "CD8", "CD8", "CD8"]},
        index=["c1", "c2", "c3", "c4"],
    )
    adata = FakeAdata(obs)
    cell_df = pd.DataFrame(
        {
            "celltype": ["CD8", "CD8", "CD8", "CD8"],
            "x": [1.0, 0.9, 0.1, 0.0],
            "y": [0.0, 0.1, 0.9, 1.0],
        },
        index=obs.index,
    )
    additional_df = pd.DataFrame({"data": ["cytotoxic"], "x": [1.0], "y": [0.0]})

    outputs = module.run_functionality_downstream_assessment(
        embeddings_dict={
            "MB": {
                "scrna": {
                    "df_cells": {"dataframe": cell_df, "umap": None, "metadata": {"run_dir": "demo"}},
                    "df_additional": {"dataframe": additional_df},
                }
            }
        },
        eval_data={"adata": adata},
        config=module.FunctionalityDownstreamConfig(
            functionality_labels=["cytotoxic"],
            output_dir=tmp_path,
            annotation_column="celltype",
            cutoff_method="manual",
            manual_cutoffs={"cytotoxic": 0.5},
            gene_sets={"cytotoxic": ["NKG7", "GZMA", "MISSING"]},
            min_group_cells=1,
            plot=False,
            timestamp="2026-07-28T10-00-00",
        ),
    )

    run_dir = outputs["run_dir"]
    label_dir = run_dir / "MB" / "cytotoxic" / "CD8"
    assert (label_dir / "df_sim.csv").exists()
    assert (label_dir / "cutoff_metadata.json").exists()
    assert (label_dir / "gene_set_score_summary.csv").exists()
    assert (label_dir / "deg_summary.csv").exists()
    assert (label_dir / "cutoff_sensitivity.csv").exists()
    assert (label_dir / "top_deg_cutoff_sensitivity.csv").exists()
    assert (label_dir / "gene_cutoff_sensitivity_summary.csv").exists()
    assert (label_dir / "gene_cutoff_sensitivity_top_increasing.csv").exists()
    assert not (label_dir / "rank_tail_enrichment.csv").exists()
    assert not (outputs["run_dir"] / "rank_tail_enrichment_model_summary.csv").exists()
    assert (run_dir / "downstream_summary.csv").exists()
    assert (run_dir / "metadata.json").exists()

    with (label_dir / "cutoff_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["method"] == "manual"
    assert metadata["cutoff"] == 0.5
    assert metadata["deterministic"] is True

    assert module._scanpy_calls["score_genes"] == [("cytotoxic_gene_set_score", ["NKG7", "GZMA"])]
    assert module._scanpy_calls["rank_genes_groups"]

    sensitivity = pd.read_csv(label_dir / "cutoff_sensitivity.csv")
    assert {"threshold", "threshold_delta", "n_associated", "fraction_associated"}.issubset(
        sensitivity.columns
    )
    assert 0.5 in sensitivity["threshold"].round(6).tolist()
    deg_sensitivity = pd.read_csv(label_dir / "top_deg_cutoff_sensitivity.csv")
    assert {"gene", "threshold", "mean_expression_difference"}.issubset(deg_sensitivity.columns)
    assert set(deg_sensitivity["gene"]) == {
        "NKG7",
        "GZMA",
        "CTSW",
        "CST7",
        "CCL5",
        "GZMB",
        "PRF1",
    }
    gene_summary = pd.read_csv(label_dir / "gene_cutoff_sensitivity_summary.csv")
    top_increasing = pd.read_csv(label_dir / "gene_cutoff_sensitivity_top_increasing.csv")
    assert {"gene", "slope", "delta_low_to_high", "spearman_with_cutoff"}.issubset(
        gene_summary.columns
    )
    assert len(top_increasing) <= 5
    assert top_increasing["slope"].is_monotonic_decreasing


def test_all_cells_scope_uses_one_cutoff_and_one_deg_across_cell_types(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)

    obs = pd.DataFrame(
        {"celltype": ["CD8", "CD8", "CD4", "CD4"]},
        index=["c1", "c2", "c3", "c4"],
    )
    adata = FakeAdata(obs)
    adata.obs["cytotoxic_gene_set_score"] = [0.9, 0.8, 0.1, 0.0]
    cell_df = pd.DataFrame(
        {
            "celltype": ["CD8", "CD8", "CD4", "CD4"],
            "x": [1.0, 0.9, 0.1, 0.0],
            "y": [0.0, 0.1, 0.9, 1.0],
        },
        index=obs.index,
    )
    model_umap = pd.DataFrame(
        {
            "UMAP1": [10.0, 11.0, 12.0, 13.0],
            "UMAP2": [20.0, 21.0, 22.0, 23.0],
        },
        index=obs.index,
    )
    additional_df = pd.DataFrame({"data": ["cytotoxic"], "x": [1.0], "y": [0.0]})

    outputs = module.run_functionality_downstream_assessment(
        embeddings_dict={
            "MB": {
                "scrna": {
                    "df_cells": {"dataframe": cell_df, "umap": model_umap, "metadata": {"run_dir": "demo"}},
                    "df_additional": {"dataframe": additional_df},
                }
            }
        },
        eval_data={"adata": adata},
        config=module.FunctionalityDownstreamConfig(
            functionality_labels=["cytotoxic"],
            output_dir=tmp_path,
            annotation_column="celltype",
            assessment_scope="all_cells",
            cutoff_method="manual",
            manual_cutoffs={"cytotoxic": 0.5},
            gene_sets={"cytotoxic": ["NKG7", "GZMA"]},
            min_group_cells=1,
            plot=True,
            timestamp="2026-07-28T11-00-00",
        ),
    )

    result_dir = outputs["run_dir"] / "MB" / "cytotoxic" / "all_cells"
    assert (result_dir / "df_sim.csv").exists()
    assert (result_dir / "score_histogram_by_cell_type.pdf").exists()
    assert (result_dir / "score_histogram_by_cell_type_no_cutoff.pdf").exists()
    assert (outputs["run_dir"] / "model_score_vs_gene_set_score_by_model.pdf").exists()
    assert (outputs["run_dir"] / "gene_set_spearman_ablation_summary.pdf").exists()
    assert (outputs["run_dir"] / "gene_set_pearson_ablation_summary.pdf").exists()
    assert (result_dir / "rank_tail_enrichment.csv").exists()
    assert (result_dir / "rank_tail_summary.csv").exists()
    assert (result_dir / "rank_decile_overlap.csv").exists()
    assert (result_dir / "rank_decile_overlap_heatmap.pdf").exists()
    assert (result_dir / "rank_cell_type_tail_summary.csv").exists()
    assert (outputs["run_dir"] / "rank_tail_enrichment_metrics.csv").exists()
    assert (outputs["run_dir"] / "rank_tail_enrichment_model_summary.csv").exists()
    assert (outputs["run_dir"] / "rank_tail_lift_ablation_line_cytotoxic.pdf").exists()
    assert (outputs["run_dir"] / "rank_tail_median_gene_set_percentile_ablation_line_cytotoxic.pdf").exists()
    assert (outputs["run_dir"] / "rank_tail_lift_auc_ablation_bar_cytotoxic.pdf").exists()
    assert (outputs["run_dir"] / "rank_tail_median_gene_set_percentile_auc_ablation_bar_cytotoxic.pdf").exists()
    assert not (outputs["run_dir"] / "rank_tail_lift_ablation_line.pdf").exists()
    assert not (outputs["run_dir"] / "rank_tail_lift_auc_ablation_bar.pdf").exists()
    assert (result_dir / "umap_model_similarity_score_rank.pdf").exists()
    assert (result_dir / "umap_model_gene_set_score_rank.pdf").exists()
    assert (result_dir / "umap_adata_similarity_score_rank.pdf").exists()
    assert (result_dir / "umap_adata_gene_set_score_rank.pdf").exists()

    df_sim = pd.read_csv(result_dir / "df_sim.csv", index_col=0)
    summary = pd.read_csv(outputs["run_dir"] / "downstream_summary.csv")
    rank_tail = pd.read_csv(result_dir / "rank_tail_enrichment.csv")
    rank_summary = pd.read_csv(result_dir / "rank_tail_summary.csv")
    decile_overlap = pd.read_csv(result_dir / "rank_decile_overlap.csv")
    cell_type_tail = pd.read_csv(result_dir / "rank_cell_type_tail_summary.csv")
    model_summary = pd.read_csv(outputs["run_dir"] / "rank_tail_enrichment_model_summary.csv")
    assert set(df_sim["celltype"]) == {"CD8", "CD4"}
    assert {"similarity_score_rank", "gene_set_score_rank"}.issubset(df_sim.columns)
    assert {"model_UMAP1", "model_UMAP2", "adata_UMAP1", "adata_UMAP2"}.issubset(df_sim.columns)
    assert len(summary) == 1
    assert summary.loc[0, "cell_type"] == "all_cells"
    assert summary.loc[0, "assessment_scope"] == "all_cells"
    assert {"top_fraction", "lift", "median_gene_set_percentile_rank"}.issubset(rank_tail.columns)
    assert {"area_under_lift_curve", "area_under_median_gene_set_percentile_curve"}.issubset(
        rank_summary.columns
    )
    assert {"embedding_decile", "gene_set_decile", "fraction_of_embedding_decile"}.issubset(
        decile_overlap.columns
    )
    assert {"celltype", "fraction_of_embedding_tail", "median_gene_set_percentile_rank"}.issubset(
        cell_type_tail.columns
    )
    assert model_summary.loc[0, "model_name"] == "MB"
    assert model_summary.loc[0, "functionality"] == "cytotoxic"
    assert module._scanpy_calls["rank_genes_groups"] == [("associated", "False", "wilcoxon")]


def test_donor_level_youden_condition_assigns_one_model_label_per_donor(monkeypatch):
    module = load_downstream_module(monkeypatch)
    obs = pd.DataFrame(
        {"subject": ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4"]},
        index=[f"c{i}" for i in range(8)],
    )
    adata = FakeAdata(obs)
    df_sim = pd.DataFrame(
        {
            "score": [0.92, 0.82, 0.74, 0.70, 0.25, 0.20, 0.15, 0.10],
            "ground_truth_positive": [True, True, True, True, False, False, False, False],
        },
        index=obs.index,
    )

    assigned, audit, summary = module._assign_donor_level_youden_condition(
        df_sim,
        adata,
        subject_column="subject",
        score_column="score",
        ground_truth_column="ground_truth_positive",
        output_column="donor_model_associated",
    )

    assert summary["status"] == "ok"
    assert summary["donor_score_aggregation"] == "median"
    assert summary["n_donors_positive"] == 2
    assert summary["n_donors_negative"] == 2
    assert 0.0 <= summary["donor_youden_threshold"] <= 1.0
    assert audit["donor_model_associated"].tolist() == [True, True, False, False]
    assert assigned.groupby(obs["subject"])["donor_model_associated"].nunique().max() == 1
    assert assigned.loc[["c1", "c4"], "donor_model_associated"].tolist() == [True, False]


def test_donor_level_youden_condition_skips_without_two_ground_truth_classes(monkeypatch):
    module = load_downstream_module(monkeypatch)
    obs = pd.DataFrame({"subject": ["d1", "d1", "d2", "d2"]}, index=["c1", "c2", "c3", "c4"])
    adata = FakeAdata(obs)
    df_sim = pd.DataFrame(
        {
            "score": [0.9, 0.8, 0.7, 0.6],
            "ground_truth_positive": [True, True, True, True],
        },
        index=obs.index,
    )

    assigned, audit, summary = module._assign_donor_level_youden_condition(
        df_sim,
        adata,
        subject_column="subject",
        score_column="score",
        ground_truth_column="ground_truth_positive",
        output_column="donor_model_associated",
    )

    assert assigned.equals(df_sim)
    assert set(audit["subject"]) == {"d1", "d2"}
    assert audit["donor_ground_truth_positive"].all()
    assert summary["status"] == "skipped"
    assert summary["reason"] == "missing_two_ground_truth_classes"


def test_subject_level_model_deg_uses_donor_level_youden_condition(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)
    calls = []

    def fake_subject_level_deseq2(*args, **kwargs):
        calls.append(kwargs)
        output_dir = Path(args[2])
        output_prefix = kwargs.get("output_prefix", "")
        pd.DataFrame({"gene": ["NKG7"], "log2FoldChange": [1.0], "padj": [0.01]}).to_csv(
            output_dir / f"{output_prefix}subject_level_deseq2_results.csv",
            index=False,
        )
        return pd.DataFrame(
            [
                {
                    "status": "ok",
                    "design": "~condition",
                    "condition_column": kwargs["condition_column"],
                    "split_mixed_subjects": kwargs["split_mixed_subjects"],
                }
            ]
        )

    monkeypatch.setattr(module, "_subject_level_deseq2", fake_subject_level_deseq2)
    monkeypatch.setattr(module, "_run_deg", lambda *args, **kwargs: pd.DataFrame([{"status": "skipped"}]))
    monkeypatch.setattr(module, "_run_ground_truth_deg_comparison", lambda *args, **kwargs: pd.DataFrame([{"status": "skipped"}]))
    monkeypatch.setattr(module, "_write_pydeseq2_deg_comparison", lambda *args, **kwargs: pd.DataFrame([{"status": "ok"}]))

    obs = pd.DataFrame(
        {
            "celltype": ["CD8"] * 8,
            "subject": ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4"],
            "cmv": ["Positive", "Positive", "Positive", "Positive", "Negative", "Negative", "Negative", "Negative"],
        },
        index=[f"c{i}" for i in range(8)],
    )
    adata = FakeAdata(obs)
    cell_df = pd.DataFrame(
        {
            "celltype": ["CD8"] * 8,
            "x": [1.0, 0.9, 0.8, 0.75, 0.2, 0.15, 0.1, 0.05],
            "y": [0.0, 0.1, 0.2, 0.25, 0.8, 0.85, 0.9, 0.95],
        },
        index=obs.index,
    )
    additional_df = pd.DataFrame({"data": ["cmv prompt"], "x": [1.0], "y": [0.0]})

    outputs = module.run_functionality_downstream_assessment(
        embeddings_dict={
            "MB": {
                "scrna": {
                    "df_cells": {"dataframe": cell_df, "umap": None, "metadata": {"run_dir": "demo"}},
                    "df_additional": {"dataframe": additional_df},
                }
            }
        },
        eval_data={"adata": adata},
        config=module.FunctionalityDownstreamConfig(
            functionality_labels=["cmv prompt"],
            output_dir=tmp_path,
            annotation_column="celltype",
            ground_truth_column="cmv",
            positive_values=["Positive"],
            cutoff_method="youden_j",
            min_group_cells=1,
            subject_level_deg=True,
            subject_column="subject",
            pseudobulk_layer="",
            min_subjects_per_group=1,
            plot=False,
            timestamp="2026-07-28T13-00-00",
        ),
    )

    associated_call = [call for call in calls if call["output_prefix"] == "associated_"][0]
    assert associated_call["condition_column"] == "donor_model_associated"
    assert associated_call["split_mixed_subjects"] is False
    result_dir = outputs["run_dir"] / "MB" / "cmv_prompt" / "CD8"
    assert (result_dir / "associated_donor_youden_assignment.csv").exists()
    summary = pd.read_csv(result_dir / "associated_subject_level_deseq2_summary.csv")
    assert summary.loc[0, "donor_condition_strategy"] == "youden_j_median_score"


def test_downstream_requires_exact_functionality_embeddings_by_default(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)

    cell_df = pd.DataFrame(
        {
            "celltype": ["CD8", "CD8"],
            "x": [1.0, 0.0],
            "y": [0.0, 1.0],
        },
        index=["c1", "c2"],
    )
    additional_df = pd.DataFrame({"data": ["not cytotoxic"], "x": [1.0], "y": [0.0]})

    with pytest.raises(ValueError, match="Missing exact functionality embeddings"):
        module.run_functionality_downstream_assessment(
            embeddings_dict={
                "MB": {
                    "scrna": {
                        "df_cells": {"dataframe": cell_df, "umap": None, "metadata": {"run_dir": "demo"}},
                        "df_additional": {"dataframe": additional_df},
                    }
                }
            },
            config=module.FunctionalityDownstreamConfig(
                functionality_labels=["cytotoxic"],
                output_dir=tmp_path,
                annotation_column="celltype",
                plot=False,
                timestamp="2026-07-28T12-00-00",
            ),
        )


def test_attach_umap_coordinates_prefers_named_umap_columns(monkeypatch):
    module = load_downstream_module(monkeypatch)
    work = pd.DataFrame({"celltype": ["CD8", "CD4"]}, index=["c1", "c2"])
    umap_df = pd.DataFrame(
        {
            "0": [100.0, 200.0],
            "1": [300.0, 400.0],
            "UMAP1": [1.0, 2.0],
            "UMAP2": [3.0, 4.0],
        },
        index=["c1", "c2"],
    )

    attached = module._attach_umap_coordinates(
        work,
        cell_artifact={"umap": umap_df},
        adata=None,
    )

    assert attached["UMAP1"].tolist() == [1.0, 2.0]
    assert attached["UMAP2"].tolist() == [3.0, 4.0]


def test_deg_comparison_barplots_use_only_top5_overlap(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)
    captured = {"genes": None}

    associated_deg = pd.DataFrame(
        {
            "names": ["A", "B", "C", "D", "E", "MODEL_ONLY"],
            "logfoldchanges": [1, 1, 1, 1, 1, 1],
            "pvals_adj": [0.01] * 6,
        }
    )
    ground_truth_deg = pd.DataFrame(
        {
            "names": ["X", "Y", "C", "D", "E", "GROUND_TRUTH_ONLY"],
            "logfoldchanges": [1, 1, 1, 1, 1, 1],
            "pvals_adj": [0.01] * 6,
        }
    )
    results = [associated_deg, ground_truth_deg]
    monkeypatch.setattr(
        module._scanpy().get,
        "rank_genes_groups_df",
        lambda adata, group: results.pop(0),
    )
    monkeypatch.setattr(
        module,
        "_plot_deg_comparison_bars",
        lambda comparison, output_dir: captured.update(
            {"genes": comparison["names"].astype(str).tolist()}
        ),
    )

    obs = pd.DataFrame({"celltype": ["CD8"] * 4}, index=["c1", "c2", "c3", "c4"])
    adata = FakeAdata(obs)
    df_sim = pd.DataFrame(
        {
            "associated": [True, True, False, False],
            "ground_truth_positive": [True, True, False, False],
        },
        index=obs.index,
    )

    summary = module._run_ground_truth_deg_comparison(
        adata,
        df_sim,
        tmp_path,
        min_group_cells=1,
    )

    assert captured["genes"] == ["C", "D", "E"]
    assert summary.loc[0, "compared_genes"] == "C;D;E"


def test_cell_type_histogram_uses_configured_repo_palette(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)
    captured_colors = []

    class FakeAxis:
        spines = {"top": types.SimpleNamespace(set_visible=lambda value: None), "right": types.SimpleNamespace(set_visible=lambda value: None)}

        def hist(self, *args, **kwargs):
            captured_colors.append(kwargs["color"])

        def axvline(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def tick_params(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

    class FakeFig:
        def savefig(self, *args, **kwargs):
            return None

    fake_axis = FakeAxis()
    monkeypatch.setattr(module.plt, "subplots", lambda *args, **kwargs: (FakeFig(), fake_axis))
    monkeypatch.setattr(module.plt, "close", lambda *args, **kwargs: None)

    df_sim = pd.DataFrame(
        {
            "celltype": ["CD4", "CD8"],
            "score": [0.1, 0.9],
        }
    )
    module._plot_histogram_by_cell_type(
        df_sim,
        annotation_column="celltype",
        cutoff=0.5,
        output_path=tmp_path / "hist.pdf",
        bins=5,
        palette_name="tab20_new",
    )

    assert captured_colors[:2] == module.CATEGORICAL_PALETTES["tab20_new"][:2]


def test_cell_type_histogram_handles_nearly_identical_scores(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)
    df_sim = pd.DataFrame(
        {
            "celltype": ["CD4", "CD8", "CD8"],
            "score": [0.1, 0.1 + 1e-320, 0.1 + 2e-320],
        }
    )

    module._plot_histogram_by_cell_type(
        df_sim,
        annotation_column="celltype",
        cutoff=0.1,
        output_path=tmp_path / "hist.pdf",
        bins=60,
        palette_name="tab20_new",
    )

    assert (tmp_path / "hist.pdf").exists()


def test_cell_type_histogram_can_hide_cutoff_line(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)
    captured = {"axvline_calls": 0}

    class FakeAxis:
        spines = {"top": types.SimpleNamespace(set_visible=lambda value: None), "right": types.SimpleNamespace(set_visible=lambda value: None)}

        def hist(self, *args, **kwargs):
            return None

        def axvline(self, *args, **kwargs):
            captured["axvline_calls"] += 1

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def tick_params(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

    class FakeFig:
        def savefig(self, *args, **kwargs):
            return None

    monkeypatch.setattr(module.plt, "subplots", lambda *args, **kwargs: (FakeFig(), FakeAxis()))
    monkeypatch.setattr(module.plt, "close", lambda *args, **kwargs: None)

    df_sim = pd.DataFrame(
        {
            "celltype": ["CD4", "CD8", "CD8"],
            "score": [0.1, 0.5, 0.9],
        }
    )
    module._plot_histogram_by_cell_type(
        df_sim,
        annotation_column="celltype",
        cutoff=0.5,
        output_path=tmp_path / "hist.pdf",
        bins=5,
        palette_name="tab20_new",
        show_cutoff=False,
    )

    assert captured["axvline_calls"] == 0


def test_gene_set_correlation_barplot_uses_annotation_ablation_order(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)
    captured = {"x": None, "colors": None, "labels": None}

    class FakeAxis:
        spines = {"top": types.SimpleNamespace(set_visible=lambda value: None), "right": types.SimpleNamespace(set_visible=lambda value: None)}

        def bar(self, x, height, **kwargs):
            captured["x"] = list(x)
            captured["colors"] = list(kwargs["color"])

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
            return None

        def set_xticks(self, *args, **kwargs):
            return None

        def set_xticklabels(self, labels, *args, **kwargs):
            captured["labels"] = list(labels)

        def tick_params(self, *args, **kwargs):
            return None

        def set_xlim(self, *args, **kwargs):
            return None

    class FakeFig:
        def savefig(self, *args, **kwargs):
            return None

    monkeypatch.setattr(module.plt, "subplots", lambda *args, **kwargs: (FakeFig(), FakeAxis()))
    monkeypatch.setattr(module.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.sns, "despine", lambda *args, **kwargs: None)

    summary = pd.DataFrame(
        {
            "model_name": ["MH", "MB", "MI", "Base", "MF", "MG", "MJ"],
            "functionality": ["cytotoxic"] * 7,
            "spearman_gene_set": [0.1, 0.5, 0.2, 0.25, 0.3, 0.4, 0.6],
            "pearson_gene_set": [0.15, 0.55, 0.25, 0.30, 0.35, 0.45, 0.65],
        }
    )

    module._plot_gene_set_correlation_barplot(
        summary,
        tmp_path / "bar.pdf",
        metric_column="pearson_gene_set",
        y_label="Pearson correlation",
    )

    assert captured["labels"] == ["MA", "MA*", "MB", "MC", "MC*", "MD", "Base"]
    assert captured["x"] == [0.0, 0.62, 1.24, 1.8599999999999999, 2.48, 3.1, 3.7199999999999998]
    assert len(set(captured["colors"])) == 7


def test_gene_set_correlation_dotplot_uses_shapes_and_model_colors(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)
    captured = {"points": [], "labels": None, "legend_labels": None}

    class FakeAxis:
        spines = {"top": types.SimpleNamespace(set_visible=lambda value: None), "right": types.SimpleNamespace(set_visible=lambda value: None)}

        def scatter(self, x, y, **kwargs):
            captured["points"].append(
                {
                    "x": x,
                    "y": y,
                    "marker": kwargs["marker"],
                    "facecolor": kwargs["facecolor"],
                }
            )

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
            return None

        def axhline(self, *args, **kwargs):
            return None

        def set_xticks(self, *args, **kwargs):
            return None

        def set_xticklabels(self, labels, *args, **kwargs):
            captured["labels"] = list(labels)

        def tick_params(self, *args, **kwargs):
            return None

        def set_xlim(self, *args, **kwargs):
            return None

        def legend(self, handles, *args, **kwargs):
            captured["legend_labels"] = [handle.get_label() for handle in handles]

    class FakeFig:
        def savefig(self, *args, **kwargs):
            return None

    monkeypatch.setattr(module.plt, "subplots", lambda *args, **kwargs: (FakeFig(), FakeAxis()))
    monkeypatch.setattr(module.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.sns, "despine", lambda *args, **kwargs: None)

    summary = pd.DataFrame(
        {
            "model_name": ["MG", "MB", "MG", "MB", "MG", "MB"],
            "functionality": ["cytotoxic", "cytotoxic", "exhausted", "exhausted", "activated", "activated"],
            "spearman_gene_set": [0.1, 0.5, 0.2, 0.6, 0.3, 0.7],
        }
    )

    module._plot_gene_set_correlation_dotplot(
        summary,
        tmp_path / "dot.pdf",
        metric_column="spearman_gene_set",
        y_label="Spearman correlation",
    )

    assert captured["labels"] == ["MA", "MB"]
    assert captured["legend_labels"] == ["cytotoxic", "exhausted", "activated"]
    assert {point["marker"] for point in captured["points"]} == {"o", "s", "^"}
    mb_colors = {
        point["facecolor"]
        for point in captured["points"]
        if point["x"] < 0.31
    }
    mg_colors = {
        point["facecolor"]
        for point in captured["points"]
        if point["x"] > 0.31
    }
    assert len(mb_colors) == 1
    assert len(mg_colors) == 1
    assert mb_colors != mg_colors


def test_top_deg_cutoff_sensitivity_uses_red_gene_palette_and_line_labels(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)
    captured = {
        "cmap": None,
        "colors": [],
        "labels": [],
        "saved": [],
        "legend": None,
        "xticks": None,
        "formatter": None,
    }

    class FakeCmap:
        def __call__(self, value):
            captured["cmap"] = "Reds"
            return ("reds", round(float(value), 3))

    class FakeAxis:
        spines = {"top": types.SimpleNamespace(set_visible=lambda value: None), "right": types.SimpleNamespace(set_visible=lambda value: None)}

        def __init__(self):
            self.xaxis = types.SimpleNamespace(set_major_formatter=lambda formatter: captured.update({"formatter": formatter}))

        def plot(self, *args, **kwargs):
            captured["colors"].append(kwargs["color"])

        def scatter(self, *args, **kwargs):
            return None

        def axvline(self, *args, **kwargs):
            return None

        def axhline(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def tick_params(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            captured["legend"] = kwargs
            return None

        def text(self, x, y, text, **kwargs):
            captured["labels"].append((x, y, text, kwargs.get("color")))

        def set_xticks(self, ticks, *args, **kwargs):
            captured["xticks"] = list(ticks)
            return None

        def set_xlim(self, *args, **kwargs):
            return None

    class FakeFig:
        def savefig(self, *args, **kwargs):
            captured["saved"].append(args[0])
            return None

    monkeypatch.setitem(module.COLORMAPS, "Reds", FakeCmap())
    monkeypatch.setattr(module.plt, "subplots", lambda *args, **kwargs: (FakeFig(), FakeAxis()))
    monkeypatch.setattr(module.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.sns, "despine", lambda *args, **kwargs: None)

    sensitivity = pd.DataFrame(
        {
            "gene": ["NKG7", "NKG7", "GZMA", "GZMA"],
            "threshold": [0.375, 0.6, 0.375, 0.6],
            "mean_expression_difference": [1.0, 1.2, 0.5, 0.6],
        }
    )
    module._plot_top_deg_cutoff_sensitivity(sensitivity, tmp_path / "deg.pdf", cutoff=0.48)

    assert captured["cmap"] == "Reds"
    assert set(captured["colors"]) == {("reds", 0.35), ("reds", 0.9)}
    assert {label[2] for label in captured["labels"]} == {"NKG7", "GZMA"}
    assert captured["legend"]["loc"] == "center left"
    assert captured["legend"]["bbox_to_anchor"] == (1.02, 0.5)
    assert captured["xticks"] == [0.4, 0.5, 0.6]
    assert captured["formatter"].fmt == "%.1f"
    assert tmp_path / "deg.pdf" in captured["saved"]
    assert tmp_path / "deg.svg" in captured["saved"]


def test_downstream_plots_apply_celltype_annotation_style(tmp_path, monkeypatch):
    module = load_downstream_module(monkeypatch)
    captured = {}

    monkeypatch.setattr(module.sns, "set_theme", lambda **kwargs: captured.update(kwargs))
    module._set_annotation_plot_style()

    assert captured["context"] == "paper"
    assert captured["style"] == "white"
    assert captured["rc"]["font.size"] == 9
    assert captured["rc"]["axes.labelsize"] == 9
    assert captured["rc"]["xtick.labelsize"] == 8
    assert captured["rc"]["ytick.labelsize"] == 8
