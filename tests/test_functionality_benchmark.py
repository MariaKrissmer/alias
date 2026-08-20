from pathlib import Path

import pandas as pd

from alias.evaluation.functionality_benchmark import (
    AXIS_LABEL_SIZE,
    BENCHMARK_BAR_X_SPACING,
    BENCHMARK_BOTTOM_MARGIN,
    BENCHMARK_LEFT_MARGIN,
    BENCHMARK_PLOT_HEIGHT,
    BAR_WIDTH,
    BAR_X_SPACING,
    FunctionalityBenchmarkConfig,
    FunctionalityBenchmarkSource,
    PLOT_HEIGHT,
    RANK_YTICKS,
    TICK_LABEL_SIZE,
    _ablation_model_palette,
    _benchmark_assignment_x_order,
    _plot_benchmark_assignment_group_rank_plot,
    _filter_benchmark_comparison_summary,
    compute_ground_truth_ranks,
    load_source_scores,
    run_functionality_benchmark,
    summarize_ground_truth_ranks,
)
from alias.util.plots.color_definition import (
    PUBLICATION_ABLATION_MODEL_LABELS,
    PUBLICATION_ABLATION_MODEL_PALETTE,
)


def test_ground_truth_ranks_use_best_rank_for_multiple_accepted_labels(tmp_path: Path):
    scores = pd.DataFrame(
        {
            "model_name": ["demo"] * 6,
            "source_name": ["ours_cell"] * 6,
            "assignment_level": ["cell"] * 6,
            "functionality": ["cytotoxic"] * 3 + ["regulatory"] * 3,
            "cell_type": ["CD4", "CD8", "NK", "CD4", "CD8", "Treg"],
            "score": [0.1, 0.7, 0.9, 0.2, 0.5, 0.8],
        }
    )
    mapping = pd.DataFrame(
        {
            "functionality": ["cytotoxic", "cytotoxic", "regulatory"],
            "cell_type": ["CD8", "NK", "Treg"],
        }
    )

    ranks = compute_ground_truth_ranks(scores, mapping, rank_ascending=False)

    cytotoxic = ranks[ranks["functionality"] == "cytotoxic"].iloc[0]
    assert cytotoxic["rank"] == 1
    assert cytotoxic["best_cell_type"] == "NK"
    assert cytotoxic["accepted_cell_types"] == "CD8;NK"
    assert cytotoxic["status"] == "ok"

    regulatory = ranks[ranks["functionality"] == "regulatory"].iloc[0]
    assert regulatory["rank"] == 1
    assert regulatory["best_cell_type"] == "Treg"


def test_llm_matrix_loader_normalizes_old_functionality_row_format(tmp_path: Path):
    path = tmp_path / "llm.csv"
    pd.DataFrame(
        {
            "Functional Description": ["cytotoxic", "regulatory"],
            "CD8": [4, 1],
            "Treg": [0, 5],
        }
    ).to_csv(path, index=False)

    scores = load_source_scores(
        FunctionalityBenchmarkSource(
            source_name="llama",
            path=path,
            source_format="llm_matrix",
            assignment_level="llm_label",
            model_name="Llama",
        )
    )

    assert set(scores.columns) == {
        "model_name",
        "source_name",
        "assignment_level",
        "functionality",
        "cell_type",
        "score",
    }
    assert len(scores) == 4
    row = scores[(scores["functionality"] == "cytotoxic") & (scores["cell_type"] == "CD8")].iloc[0]
    assert row["score"] == 4
    assert row["model_name"] == "Llama"


def test_missing_mapping_score_is_preserved_as_missing_score():
    scores = pd.DataFrame(
        {
            "model_name": ["demo"],
            "source_name": ["ours_cell"],
            "assignment_level": ["cell"],
            "functionality": ["cytotoxic"],
            "cell_type": ["CD8"],
            "score": [0.8],
        }
    )
    mapping = pd.DataFrame(
        {
            "functionality": ["cytotoxic", "unknown functionality"],
            "cell_type": ["CD8", "Treg"],
        }
    )

    ranks = compute_ground_truth_ranks(scores, mapping)

    missing = ranks[ranks["functionality"] == "unknown functionality"].iloc[0]
    assert pd.isna(missing["rank"])
    assert missing["status"] == "missing_score"


def test_summary_topk_metrics_count_missing_scores_as_failures():
    ranks = pd.DataFrame(
        {
            "model_name": ["demo", "demo"],
            "source_name": ["ours_cell", "ours_cell"],
            "assignment_level": ["cell", "cell"],
            "functionality": ["cytotoxic", "unknown"],
            "rank": [1.0, pd.NA],
            "normalized_rank": [0.5, pd.NA],
            "reciprocal_rank": [1.0, pd.NA],
            "status": ["ok", "missing_score"],
        }
    )

    summary = summarize_ground_truth_ranks(ranks)

    assert summary.loc[0, "n_functionalities"] == 2
    assert summary.loc[0, "n_ok"] == 1
    assert summary.loc[0, "top1_accuracy"] == 0.5
    assert summary.loc[0, "missing_score_count"] == 1


def test_ablation_palette_matches_celltype_annotation_blue_ramp():
    palette = _ablation_model_palette(["MB", "MG", "MF", "MH", "MI"])

    assert PUBLICATION_ABLATION_MODEL_LABELS["MF"] == "MC"
    assert PUBLICATION_ABLATION_MODEL_LABELS["MH"] == "MC*"
    assert palette["MB"] == PUBLICATION_ABLATION_MODEL_PALETTE["MB"]
    assert palette["MG"] == PUBLICATION_ABLATION_MODEL_PALETTE["MG"]
    assert palette["MF"] == PUBLICATION_ABLATION_MODEL_PALETTE["MF"]
    assert palette["MH"] == PUBLICATION_ABLATION_MODEL_PALETTE["MH"]
    assert palette["MI"] == PUBLICATION_ABLATION_MODEL_PALETTE["MI"]


def test_benchmark_comparison_summary_keeps_only_mb_alias_model():
    summary = pd.DataFrame(
        {
            "model_name": ["MB", "MG", "MI", "CellWhisperer", "Llama-3.3-70B"],
            "source_name": ["ours_cell", "ours_cell", "ours_label", "cellwhisperer_cell", "llama_label"],
            "assignment_level": ["cell", "cell", "celltype_label", "cellwhisperer_cell", "llm_label"],
            "mrr": [0.9, 0.4, 0.5, 0.6, 0.7],
        }
    )

    filtered = _filter_benchmark_comparison_summary(summary)

    assert filtered["model_name"].tolist() == ["MB", "CellWhisperer", "Llama-3.3-70B"]


def test_benchmark_assignment_order_puts_cellwhisperer_before_mb_cell_based():
    plot_df = pd.DataFrame(
        {
            "assignment_group": [
                "Cell-based assignment",
                "Cell-based assignment",
                "Cell type label-based assignment",
            ],
            "model_name": ["MB", "CellWhisperer", "MB"],
        }
    )

    x_order = _benchmark_assignment_x_order(plot_df)

    assert x_order[:2] == [
        "Cell-based assignment__CellWhisperer",
        "Cell-based assignment__MB",
    ]


def test_functionality_plot_constants_match_celltype_annotation_plots():
    assert PLOT_HEIGHT == 1.75
    assert BENCHMARK_PLOT_HEIGHT == 2.3
    assert BENCHMARK_LEFT_MARGIN == 0.36
    assert BENCHMARK_BOTTOM_MARGIN == 0.36
    assert AXIS_LABEL_SIZE == 9.5
    assert TICK_LABEL_SIZE == 8.5
    assert BAR_X_SPACING == 0.46
    assert BENCHMARK_BAR_X_SPACING == 0.55
    assert BAR_WIDTH == 0.22
    assert RANK_YTICKS == [1, 3, 5, 7, 9]


def test_benchmark_assignment_rank_plot_writes_dot_and_swarm_versions(tmp_path: Path):
    ranks = pd.DataFrame(
        {
            "model_name": ["MB", "MB", "CellWhisperer", "CellWhisperer", "Llama-3.3-70B", "Llama-3.3-70B"],
            "source_name": [
                "ours_cell",
                "ours_label",
                "cellwhisperer_cell",
                "cellwhisperer_cell",
                "llama_label",
                "llama_label",
            ],
            "assignment_level": [
                "cell",
                "celltype_label",
                "cellwhisperer_cell",
                "cellwhisperer_cell",
                "llm_label",
                "llm_label",
            ],
            "functionality": ["f1", "f1", "f1", "f2", "f1", "f2"],
            "rank": [2, 1, 3, 5, 4, 6],
            "status": ["ok"] * 6,
        }
    )

    _plot_benchmark_assignment_group_rank_plot(ranks, tmp_path)

    assert (tmp_path / "benchmark_assignment_group_rank_plot.pdf").exists()
    assert (tmp_path / "benchmark_assignment_group_rank_plot.png").exists()
    assert (tmp_path / "benchmark_assignment_group_rank_swarm.pdf").exists()
    assert (tmp_path / "benchmark_assignment_group_rank_swarm.png").exists()


def test_run_functionality_benchmark_writes_artifacts(tmp_path: Path):
    results_path = tmp_path / "functionality_results.csv"
    pd.DataFrame(
        {
            "functionality": ["cytotoxic", "cytotoxic", "regulatory", "regulatory"],
            "cell_type": ["CD8", "Treg", "CD8", "Treg"],
            "mean_auc": [0.9, 0.1, 0.2, 0.8],
            "label_embedding_similarity": [0.7, 0.3, 0.4, 0.6],
        }
    ).to_csv(results_path, index=False)

    mapping_path = tmp_path / "mapping.csv"
    pd.DataFrame(
        {
            "Definition": ["cytotoxic", "regulatory"],
            "Cell Type": ["CD8", "Treg"],
        }
    ).to_csv(mapping_path, index=False)

    outputs = run_functionality_benchmark(
        FunctionalityBenchmarkConfig(
            output_dir=tmp_path / "benchmark",
            functionality_mapping_path=mapping_path,
            sources=[
                FunctionalityBenchmarkSource(
                    source_name="ours_cell",
                    path=results_path,
                    source_format="functionality_similarity",
                    assignment_level="cell",
                    score_column="mean_auc",
                    model_name="MB",
                ),
                FunctionalityBenchmarkSource(
                    source_name="ours_label",
                    path=results_path,
                    source_format="functionality_similarity",
                    assignment_level="celltype_label",
                    score_column="label_embedding_similarity",
                    model_name="MB",
                ),
            ],
            timestamp="2026-07-28T10-00-00",
        )
    )

    assert outputs["run_dir"].exists()
    assert (outputs["run_dir"] / "all_scores_long.csv").exists()
    assert (outputs["run_dir"] / "ground_truth_ranks.csv").exists()
    assert (outputs["run_dir"] / "benchmark_summary.csv").exists()
    assert (outputs["run_dir"] / "benchmark_metadata.json").exists()
    assert (outputs["run_dir"] / "benchmark_rank_distribution.pdf").exists()
    assert (outputs["run_dir"] / "ablation_cell_rank_swarm.pdf").exists()
    assert (outputs["run_dir"] / "ablation_celltype_label_rank_swarm.pdf").exists()
    assert (outputs["run_dir"] / "ablation_cell_mrr_bar.pdf").exists()
    assert (outputs["run_dir"] / "ablation_celltype_label_mrr_bar.pdf").exists()
    assert (outputs["run_dir"] / "ablation_cell_vs_celltype_label_mrr_bar.pdf").exists()
    assert (outputs["run_dir"] / "benchmark_assignment_group_mrr_bar.pdf").exists()
    assert (outputs["run_dir"] / "benchmark_assignment_group_rank_plot.pdf").exists()
    assert (outputs["run_dir"] / "benchmark_assignment_group_rank_swarm.pdf").exists()
    assert (outputs["run_dir"] / "benchmark_true_label_rank_heatmap.png").exists()

    summary = pd.read_csv(outputs["summary_csv"])
    assert set(summary["source_name"]) == {"ours_cell", "ours_label"}
    assert summary["top1_accuracy"].eq(1.0).all()
