import anndata as ad
import numpy as np
import pandas as pd

from alias.evaluation.heldout_timepoint_quantification import (
    HeldoutTimepointNeighborConfig,
    _compute_neighbor_metrics,
)
from alias.evaluation.heldout_timepoint_umap import _align_extra_obs


def test_heldout_umap_aligns_time_and_sample_by_cell_id():
    cells = pd.DataFrame({0: [1.0, 2.0]}, index=["cell_2", "cell_1"])
    adata = ad.AnnData(
        X=np.zeros((2, 1)),
        obs=pd.DataFrame(
            {"time": [7.0, 8.0], "sample": ["a", "b"]},
            index=["cell_1", "cell_2"],
        ),
    )
    aligned = _align_extra_obs(cells, adata, ["time", "sample"])
    assert aligned["time"].tolist() == [8.0, 7.0]
    assert aligned["sample"].tolist() == ["b", "a"]


def test_heldout_neighbor_metrics_use_adjacent_timepoints():
    cells = pd.DataFrame(
        {
            "celltype": ["Forebrain"] * 3,
            "time": [7.0, 8.0, 9.0],
            "_timepoint_label": ["7.0", "8.0", "9.0"],
        },
        index=["cell_1", "cell_2", "cell_3"],
    )
    metrics = _compute_neighbor_metrics(
        np.array([[0.0], [1.0], [2.0]]),
        cells,
        {"8.0"},
        representation="alias",
        model_name="ALIASMG",
        dataset_name="S9",
        annotation_column="celltype",
        config=HeldoutTimepointNeighborConfig(k_neighbors=2),
    )
    assert metrics["cell_id"].tolist() == ["cell_2"]
    assert metrics["fraction_adjacent_timepoint_neighbors"].tolist() == [1.0]
    assert metrics["fraction_same_timepoint_neighbors"].tolist() == [0.0]
