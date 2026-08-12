from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from alias.util.append_adata import (
    add_embedding_artifacts_to_adata,
    add_embeddings_to_adata,
    add_umap_to_adata,
)
from alias.util.artifacts import save_embedding_frame


def _make_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {"celltype": ["T", "B", "NK"]},
        index=["cell_c", "cell_a", "cell_b"],
    )
    return ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["GENE1", "GENE2"]),
    )


def _make_embeddings_dict(tmp_path: Path) -> dict:
    run_dir = tmp_path / "embeddings" / "scrna" / "demo_model" / "run"
    embedding_df = pd.DataFrame(
        {
            "0": [1.0, 2.0, 3.0],
            "1": [4.0, 5.0, 6.0],
        },
        index=["cell_a", "cell_b", "cell_c"],
    )
    df_cells_meta = save_embedding_frame(run_dir, "df_cells", embedding_df)

    umap_df = pd.DataFrame(
        {"UMAP1": [10.0, 20.0, 30.0], "UMAP2": [40.0, 50.0, 60.0]},
        index=["cell_a", "cell_b", "cell_c"],
    )
    umap_path = run_dir / "df_cells_umap.parquet"
    umap_df.to_parquet(umap_path)
    df_cells_meta["umap"] = {"path": str(umap_path)}

    return {"demo_model": {"scrna": {"df_cells": df_cells_meta}}}


def test_add_embeddings_to_adata_aligns_embeddings_to_adata_obs_order(tmp_path: Path):
    adata = _make_adata()
    embeddings_dict = _make_embeddings_dict(tmp_path)

    updated = add_embeddings_to_adata(adata, embeddings_dict, model_key="demo_model")

    assert list(updated.obs_names) == ["cell_c", "cell_a", "cell_b"]
    np.testing.assert_array_equal(
        updated.obsm["X_demo_model"],
        np.array([[3.0, 6.0], [1.0, 4.0], [2.0, 5.0]], dtype=np.float32),
    )
    assert "X_demo_model" not in adata.obsm


def test_add_umap_to_adata_adds_saved_umap_coordinates(tmp_path: Path):
    adata = _make_adata()
    embeddings_dict = _make_embeddings_dict(tmp_path)

    updated = add_umap_to_adata(adata, embeddings_dict, model_key="demo_model")

    assert list(updated.obs_names) == ["cell_c", "cell_a", "cell_b"]
    np.testing.assert_array_equal(
        updated.obsm["X_umap_demo_model"],
        np.array([[30.0, 60.0], [10.0, 40.0], [20.0, 50.0]], dtype=np.float32),
    )


def test_add_embedding_artifacts_to_adata_adds_embeddings_and_umap(tmp_path: Path):
    adata = _make_adata()
    embeddings_dict = _make_embeddings_dict(tmp_path)

    updated = add_embedding_artifacts_to_adata(
        adata,
        embeddings_dict,
        model_key="demo_model",
    )

    assert "X_demo_model" in updated.obsm
    assert "X_umap_demo_model" in updated.obsm
