import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alias.evaluation.lamanno_timepoint_similarity import (
    _align_cells_to_adata,
    _lineage_scores,
    _timepoint_embeddings_from_additional,
)


def test_timepoint_scores_follow_saved_cell_and_text_embeddings():
    cells = pd.DataFrame(
        {0: [1.0, 0.0], 1: [0.0, 1.0]},
        index=pd.Index(["cell_1", "cell_2"]),
    )
    adata = ad.AnnData(
        X=np.zeros((2, 1)),
        obs=pd.DataFrame(
            {"celltype": ["Forebrain", "Forebrain"], "time": [7.0, 8.0]},
            index=["cell_1", "cell_2"],
        ),
    )
    aligned = _align_cells_to_adata(
        cells, adata, annotation_column="celltype", time_column="time", dataset_name="S9"
    )
    scores = _lineage_scores(
        aligned,
        embedding_columns=[0, 1],
        timepoint_labels=["7.0", "8.0"],
        time_embeddings=np.eye(2, dtype=np.float32),
        annotation_column="celltype",
        time_column="time",
        lineage_name="forebrain",
        saved_model_name="ALIASMF",
        dataset_name="S9",
        metric="cosine",
    )
    observed = scores[scores["is_observed_timepoint"]].set_index("cell_id")
    assert observed["true_timepoint_rank"].to_dict() == {"cell_1": 1, "cell_2": 1}
    assert observed["sim_score"].to_dict() == {"cell_1": 1.0, "cell_2": 1.0}


def test_timepoint_scores_require_embeddings_for_all_observed_times():
    loaded_artifacts = {
        "df_additional": {
            "dataframe": pd.DataFrame({"data": ["7.0"], 0: [1.0], 1: [0.0]}),
            "metadata": {"path": "run/df_additional.parquet"},
        }
    }
    with pytest.raises(ValueError, match="missing timepoint labels"):
        _timepoint_embeddings_from_additional(
            loaded_artifacts,
            timepoint_labels=["7.0", "8.0"],
            saved_model_name="ALIASMF",
            dataset_name="S9",
        )
