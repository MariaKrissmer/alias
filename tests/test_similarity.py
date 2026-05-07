from pathlib import Path

import numpy as np
import pandas as pd

from alias.util import similarity as similarity_module


class DummyPlotter:
    def __init__(self, *args, **kwargs):
        self.annotate_centroids = False

    def _touch(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("plot", encoding="utf-8")

    def plot_roc(self, df, output_path, title):
        self._touch(output_path)

    def plot_cells(self, df, output_path, title, continuous_color_column=None, annotate_centroids_df=None):
        self._touch(output_path)

    def plot_similarity_histogram(self, df, label, output_path, bins):
        self._touch(output_path)


def test_evaluate_similarity_meta_is_standalone_and_writes_outputs(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(similarity_module, "UMAPCellPlotter", DummyPlotter)
    monkeypatch.setattr(
        similarity_module.util,
        "cos_sim",
        lambda a, b: np.array([[0.9], [0.8], [0.2], [0.1]], dtype=np.float32),
    )

    df_cells = pd.DataFrame(
        {
            "embedding": [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.9, 0.1], dtype=np.float32),
                np.array([0.1, 0.9], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ],
            "UMAP1": [0.0, 1.0, 2.0, 3.0],
            "UMAP2": [0.0, 1.0, 2.0, 3.0],
            "disease_positive": [True, True, False, False],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    df_centroids = pd.DataFrame({"cell_type": ["T_cell"], "UMAP1": [0.5], "UMAP2": [0.5]})

    result = similarity_module.evaluate_similarity_meta(
        df_cells=df_cells,
        df_centroids=df_centroids,
        out_dir=tmp_path,
        disease_emb=np.array([[1.0, 0.0]], dtype=np.float32),
        label_key="disease_positive",
        bins=30,
        annotation_column_value="T_cell",
    )

    assert result["cell_type"] == "T_cell"
    assert "associated" in result["df_sim"].columns
    assert (tmp_path / "disease_positive" / "roc_curves" / "disease_positive.pdf").exists()
    assert (tmp_path / "disease_positive" / "umap" / "disease_positive.pdf").exists()
    assert (tmp_path / "disease_positive" / "histograms" / "disease_positive.pdf").exists()
