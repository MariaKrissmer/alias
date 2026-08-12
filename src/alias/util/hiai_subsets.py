from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_HIAI_T_CELL_L2: list[str] = [
    "Naive CD4 T cell",
    "Memory CD4 T cell",
    "Treg",
    "DN T cell",
    "Proliferating T cell",
    "Naive CD8 T cell",
    "Memory CD8 T cell",
    "CD8aa",
    "gamma delta T",
    "MAIT",
]


def subset_hiai_t_cells(
    adata,
    *,
    annotation_column: str = "AIFI_L2",
    t_cell_labels: Sequence[str] | None = None,
    min_keep: int = 2000,
    fraction: float = 0.2,
    random_state: int = 42,
    use_raw: bool = True,
    normalize_gdt: bool = True,
):
    """Create the revision HIAI T-cell subset used in the manuscript scripts."""
    if annotation_column not in adata.obs:
        raise KeyError(f"Missing annotation column {annotation_column!r} in adata.obs.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be > 0 and <= 1.")
    if min_keep < 1:
        raise ValueError("min_keep must be >= 1.")

    if normalize_gdt:
        adata.obs[annotation_column] = adata.obs[annotation_column].replace(
            {"gdT": "gamma delta T"}
        )

    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True requires adata.raw to be present.")
        adata_work = adata.raw.to_adata()
        if normalize_gdt:
            adata_work.obs[annotation_column] = adata_work.obs[annotation_column].replace(
                {"gdT": "gamma delta T"}
            )
    else:
        adata_work = adata

    labels = list(t_cell_labels or DEFAULT_HIAI_T_CELL_L2)
    adata_tcells = adata_work[adata_work.obs[annotation_column].isin(labels)].copy()

    rng = np.random.default_rng(random_state)
    indices_to_keep: list[str] = []
    for _, group in adata_tcells.obs.groupby(annotation_column, observed=False):
        cell_indices = group.index.to_numpy()
        n_cells = len(cell_indices)
        if n_cells < min_keep:
            selected = cell_indices
        else:
            n_keep = int(np.ceil(n_cells * fraction))
            selected = rng.choice(cell_indices, size=n_keep, replace=False)
        indices_to_keep.extend(selected.tolist())

    return adata_tcells[indices_to_keep, :].copy()
