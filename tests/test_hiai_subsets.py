import numpy as np
import pandas as pd
from anndata import AnnData

from alias.util.hiai_subsets import subset_hiai_t_cells


def test_subset_hiai_t_cells_normalizes_gdt_and_samples_by_celltype():
    obs = pd.DataFrame(
        {
            "AIFI_L2": [
                "Naive CD4 T cell",
                "Naive CD4 T cell",
                "Naive CD4 T cell",
                "gdT",
                "Memory B cell",
                "Treg",
            ]
        },
        index=[f"cell_{idx}" for idx in range(6)],
    )
    adata = AnnData(X=np.ones((6, 3)), obs=obs)
    adata.raw = adata

    subset = subset_hiai_t_cells(
        adata,
        annotation_column="AIFI_L2",
        min_keep=2,
        fraction=0.5,
        random_state=42,
    )

    assert subset.n_obs == 4
    assert "Memory B cell" not in set(subset.obs["AIFI_L2"])
    assert "gdT" not in set(subset.obs["AIFI_L2"])
    assert "gamma delta T" in set(subset.obs["AIFI_L2"])
    assert subset.obs["AIFI_L2"].value_counts().to_dict()["Naive CD4 T cell"] == 2
