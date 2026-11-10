from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse


GeneWeighting = Literal["inverse_sqrt_rank", "one_hot"]


def _iter_records(rows: Any) -> Iterable[dict[str, Any]]:
    if isinstance(rows, pd.DataFrame):
        yield from rows.to_dict(orient="records")
        return
    yield from rows


def _gene_list_from_record(record: dict[str, Any]) -> list[str]:
    gene_list = record.get("gene_list")
    if isinstance(gene_list, str):
        return [gene for gene in gene_list.split() if gene]
    if gene_list is None:
        sentence = record.get("sentence1", "")
        return [gene for gene in str(sentence).split() if gene]
    return [str(gene) for gene in gene_list if str(gene)]


def build_gene_list_matrix(
    rows: Any,
    *,
    weighting: GeneWeighting,
    index_column: str = "index",
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    """Build a sparse cell x gene matrix from ranked gene-list rows."""
    records = list(_iter_records(rows))
    gene_to_column: dict[str, int] = {}
    cell_ids: list[str] = []
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    for row_index, record in enumerate(records):
        cell_id = record.get(index_column, row_index)
        cell_ids.append(str(cell_id))
        seen_genes: set[str] = set()
        for rank, gene in enumerate(_gene_list_from_record(record), start=1):
            if gene in seen_genes:
                continue
            seen_genes.add(gene)
            if gene not in gene_to_column:
                gene_to_column[gene] = len(gene_to_column)
            row_indices.append(row_index)
            column_indices.append(gene_to_column[gene])
            if weighting == "inverse_sqrt_rank":
                values.append(float(1.0 / np.sqrt(rank)))
            elif weighting == "one_hot":
                values.append(1.0)
            else:
                raise ValueError(f"Unsupported gene-list weighting: {weighting}")

    matrix = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(records), len(gene_to_column)),
        dtype=np.float32,
    )
    genes = [
        gene
        for gene, _ in sorted(gene_to_column.items(), key=lambda item: item[1])
    ]
    return matrix, genes, cell_ids


def compute_gene_list_pca_embeddings(
    rows: Any,
    *,
    weighting: GeneWeighting,
    n_components: int = 50,
    random_state: int = 42,
    index_column: str = "index",
) -> pd.DataFrame:
    """Compute low-dimensional embeddings from ranked gene-list matrices."""
    try:
        from sklearn.decomposition import TruncatedSVD
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for PCA baseline embeddings. Install it with:\n"
            'uv pip install --python .venv_annotation/bin/python "scikit-learn>=1.4"'
        ) from error

    matrix, _, cell_ids = build_gene_list_matrix(
        rows,
        weighting=weighting,
        index_column=index_column,
    )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Cannot compute PCA baseline embeddings from an empty gene matrix.")

    effective_components = min(n_components, max(1, min(matrix.shape) - 1))
    reducer = TruncatedSVD(
        n_components=effective_components,
        random_state=random_state,
    )
    embeddings = reducer.fit_transform(matrix).astype(np.float32)
    return pd.DataFrame(
        embeddings,
        index=pd.Index(cell_ids, name=index_column),
        columns=[str(index) for index in range(embeddings.shape[1])],
    )


def compute_expression_pca_embeddings(
    adata: Any,
    *,
    n_components: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute PCA-style embeddings from the full AnnData expression matrix."""
    try:
        from sklearn.decomposition import TruncatedSVD
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for expression PCA baseline embeddings. "
            "Install it with:\n"
            'uv pip install --python .venv_annotation/bin/python "scikit-learn>=1.4"'
        ) from error

    matrix = adata.X
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Cannot compute PCA baseline embeddings from an empty expression matrix.")

    effective_components = min(n_components, max(1, min(matrix.shape) - 1))
    reducer = TruncatedSVD(
        n_components=effective_components,
        random_state=random_state,
    )
    embeddings = reducer.fit_transform(matrix).astype(np.float32)
    return pd.DataFrame(
        embeddings,
        index=pd.Index(adata.obs_names.astype(str), name="index"),
        columns=[str(index) for index in range(embeddings.shape[1])],
    )
