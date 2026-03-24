import importlib.util
from pathlib import Path

import pandas as pd

from alias.util.artifacts import create_run_directory, save_embedding_frame


PLOTS_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "alias" / "evaluation" / "celltype_label_plots.py"
)
PLOTS_SPEC = importlib.util.spec_from_file_location(
    "alias.evaluation.celltype_label_plots",
    PLOTS_MODULE_PATH,
)
plots_module = importlib.util.module_from_spec(PLOTS_SPEC)
assert PLOTS_SPEC.loader is not None
PLOTS_SPEC.loader.exec_module(plots_module)

EvaluationConfig = plots_module.EvaluationConfig
umap_plots = plots_module.umap_plots


class DummyPlotter:
    def __init__(self):
        self.annotate_centroids = False

    def plot_cells(
        self,
        df,
        annotation_column=None,
        continuous_color_column=None,
        output_path=None,
        annotate_centroids_df=None,
        title=None,
        **kwargs,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(title or "plot", encoding="utf-8")


def test_umap_plots_writes_timestamped_figures_and_umap_artifacts(tmp_path: Path, monkeypatch):
    run_dir = create_run_directory(
        root_dir=tmp_path,
        category="embeddings",
        dataset_name="pbmc_3k",
        model_name="demo_model",
        timestamp="2026-03-23T12-34-56",
    )

    cell_df = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=pd.Index(["0", "1"], name="cell_id"))
    celltype_df = pd.DataFrame([[0.5, 0.6], [0.7, 0.8]], index=pd.Index(["0", "1"], name="cell_id"))

    cell_meta = save_embedding_frame(
        run_dir,
        "df_cells",
        cell_df,
        annotation_map={"celltype": {"0": "T_cell", "1": "B_cell"}},
    )
    cell_meta["run_dir"] = str(run_dir)
    celltype_meta = save_embedding_frame(
        run_dir,
        "df_celltypes",
        celltype_df,
        annotation_map={"0": "T_cell", "1": "B_cell"},
    )
    celltype_meta["run_dir"] = str(run_dir)

    embeddings_dict = {
        "demo_model": {
            "pbmc_3k": {
                "df_cells": cell_meta,
                "df_celltypes": celltype_meta,
            }
        }
    }

    monkeypatch.setattr(
        plots_module,
        "compute_umap",
        lambda embeddings, evaluation_config: [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    )
    monkeypatch.setattr(plots_module, "get_umap_plotter_class", lambda: DummyPlotter)

    updated = umap_plots(
        embeddings_dict=embeddings_dict,
        annotation_column="celltype",
        output_dir=str(tmp_path),
        evaluation_config=EvaluationConfig(n_neighbors=5),
    )

    cell_umap_path = Path(updated["demo_model"]["pbmc_3k"]["df_cells"]["umap"]["path"])
    celltype_umap_path = Path(updated["demo_model"]["pbmc_3k"]["df_celltypes"]["umap"]["path"])

    assert cell_umap_path.exists()
    assert celltype_umap_path.exists()
    assert run_dir in cell_umap_path.parents
    assert cell_umap_path.parent.name == "2026-03-23T12-34-56"
    assert cell_umap_path.parent.parent.name == "celltype_label_plots"
    assert cell_umap_path.parent.parent.parent.name == "umap"

    figure_dir = (
        tmp_path
        / "demo_model"
        / "pbmc_3k"
        / "celltype_label_plots"
        / "2026-03-23T12-34-56"
    )
    assert (figure_dir / "cells_colored_by_annotation.svg").exists()
    assert (figure_dir / "cells_with_celltype_labels.svg").exists()
    assert (figure_dir / "metadata.json").exists()


def test_umap_plots_does_not_duplicate_figures_directory(tmp_path: Path, monkeypatch):
    run_dir = create_run_directory(
        root_dir=tmp_path,
        category="embeddings",
        dataset_name="pbmc_3k",
        model_name="demo_model",
        timestamp="2026-03-23T12-34-56",
    )
    cell_df = pd.DataFrame([[0.1, 0.2]], index=pd.Index(["0"], name="cell_id"))
    cell_meta = save_embedding_frame(
        run_dir,
        "df_cells",
        cell_df,
        annotation_map={"celltype": {"0": "T_cell"}},
    )

    monkeypatch.setattr(
        plots_module,
        "compute_umap",
        lambda embeddings, evaluation_config: [[0.0, 0.0]],
    )
    monkeypatch.setattr(plots_module, "get_umap_plotter_class", lambda: DummyPlotter)

    updated = umap_plots(
        embeddings_dict={"demo_model": {"pbmc_3k": {"df_cells": cell_meta}}},
        annotation_column="celltype",
        output_dir=str(tmp_path / "figures"),
        evaluation_config=EvaluationConfig(n_neighbors=5),
    )

    figure_path = (
        tmp_path
        / "figures"
        / "demo_model"
        / "pbmc_3k"
        / "celltype_label_plots"
        / "2026-03-23T12-34-56"
        / "cells_colored_by_annotation.svg"
    )
    assert figure_path.exists()
    assert "celltype_label_plots" in str(figure_path)
    assert updated["demo_model"]["pbmc_3k"]["df_cells"]["umap"]["path"].endswith("df_cells_umap.parquet")


def test_umap_plots_uses_unique_umap_artifact_directory_on_rerun(tmp_path: Path, monkeypatch):
    run_dir = create_run_directory(
        root_dir=tmp_path,
        category="embeddings",
        dataset_name="pbmc_3k",
        model_name="demo_model",
        timestamp="2026-03-23T12-34-56",
    )
    cell_df = pd.DataFrame([[0.1, 0.2]], index=pd.Index(["0"], name="cell_id"))
    cell_meta = save_embedding_frame(
        run_dir,
        "df_cells",
        cell_df,
        annotation_map={"celltype": {"0": "T_cell"}},
    )

    monkeypatch.setattr(
        plots_module,
        "compute_umap",
        lambda embeddings, evaluation_config: [[0.0, 0.0]],
    )
    monkeypatch.setattr(plots_module, "get_umap_plotter_class", lambda: DummyPlotter)

    first = umap_plots(
        embeddings_dict={"demo_model": {"pbmc_3k": {"df_cells": dict(cell_meta)}}},
        annotation_column="celltype",
        output_dir=str(tmp_path / "figures"),
        evaluation_config=EvaluationConfig(n_neighbors=5),
    )
    second = umap_plots(
        embeddings_dict={"demo_model": {"pbmc_3k": {"df_cells": dict(cell_meta)}}},
        annotation_column="celltype",
        output_dir=str(tmp_path / "figures"),
        evaluation_config=EvaluationConfig(n_neighbors=5),
    )

    first_umap_path = Path(first["demo_model"]["pbmc_3k"]["df_cells"]["umap"]["path"])
    second_umap_path = Path(second["demo_model"]["pbmc_3k"]["df_cells"]["umap"]["path"])

    assert first_umap_path != second_umap_path
    assert first_umap_path.exists()
    assert second_umap_path.exists()
