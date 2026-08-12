from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


MarkerSet = dict[str, list[str]]
MarkerMap = dict[str, MarkerSet]


DEFAULT_HIAI_AIFI_L2_MARKERS: MarkerMap = {
    "ASDC": {"positive": ["AXL", "SIGLEC6", "DAB2", "PPP1R14A"]},
    "CD14 monocyte": {"positive": ["CD14", "LYZ", "LST1", "S100A8", "S100A9", "FCN1", "VCAN"]},
    "CD16 monocyte": {"positive": ["FCGR3A", "MS4A7", "LST1", "IFITM3", "LILRB2"]},
    "CD56bright NK cell": {"positive": ["NCAM1", "XCL1", "XCL2", "KLRC1", "GZMK"]},
    "CD56dim NK cell": {"positive": ["FCGR3A", "GNLY", "NKG7", "PRF1", "GZMB", "KLRD1"]},
    "CD8aa": {"positive": ["CD8A", "TRAC", "CD3D"], "negative": ["CD4"]},
    "DN T cell": {"positive": ["TRAC", "CD3D", "CD3E"], "negative": ["CD4", "CD8A", "CD8B"]},
    "Effector B cell": {"positive": ["MZB1", "XBP1", "JCHAIN", "IGKC", "TNFRSF17"]},
    "Erythrocyte": {"positive": ["HBB", "HBA1", "HBA2", "ALAS2"]},
    "ILC": {"positive": ["IL7R", "KIT", "KLRB1", "RORA"]},
    "Intermediate monocyte": {"positive": ["CD14", "FCGR3A", "LYZ", "LST1", "MS4A7", "FCN1"]},
    "MAIT": {"positive": ["SLC4A10", "KLRB1", "TRAV1-2", "IL7R", "DPP4"]},
    "Memory B cell": {"positive": ["MS4A1", "CD79A", "CD27", "BANK1", "TNFRSF13B"]},
    "Memory CD4 T cell": {"positive": ["CD3D", "CD4", "IL7R", "CCR7", "LTB"]},
    "Memory CD8 T cell": {"positive": ["CD3D", "CD8A", "CD8B", "GZMK", "CCL5"]},
    "Naive B cell": {"positive": ["MS4A1", "CD79A", "TCL1A", "IGHD", "IL4R"]},
    "Naive CD4 T cell": {"positive": ["CD3D", "CD4", "CCR7", "LEF1", "TCF7", "SELL"]},
    "Naive CD8 T cell": {"positive": ["CD3D", "CD8A", "CD8B", "CCR7", "LEF1", "TCF7"]},
    "Plasma cell": {"positive": ["MZB1", "XBP1", "JCHAIN", "SDC1", "PRDM1"]},
    "Platelet": {"positive": ["PPBP", "PF4", "NRGN", "GP9"]},
    "Progenitor cell": {"positive": ["CD34", "SPINK2", "PRSS57", "GATA2", "KIT"]},
    "Proliferating NK": {"positive": ["MKI67", "TOP2A", "NKG7", "GNLY"]},
    "Proliferating T": {"positive": ["MKI67", "TOP2A", "CD3D", "TRAC"]},
    "Transitional B": {"positive": ["TCL1A", "IGHD", "CD79A", "MS4A1", "CD24"]},
    "Treg": {"positive": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2"]},
    "cDC1": {"positive": ["CLEC9A", "XCR1", "BATF3", "CADM1"]},
    "cDC2": {"positive": ["CD1C", "FCER1A", "CLEC10A", "CST3"]},
    "gamma delta T": {"positive": ["TRDC", "TRGC1", "TRGC2", "TRDV2", "KLRD1"]},
    "pDC": {"positive": ["GZMB", "IRF7", "TCF4", "IL3RA", "LILRA4", "CLEC4C"]},
}


@dataclass
class ScTypeAnnotationConfig:
    adata_path: Path | str
    annotation_column: str
    model_name: str
    marker_map: MarkerMap | None = None
    marker_path: Path | str | None = None
    layer: str | None = None
    use_raw: bool = False
    scale_expression: bool = True
    min_positive_markers: int = 1
    unknown_label: str = "unknown"
    min_score_margin: float | None = None


def _split_marker_string(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [
        marker.strip()
        for marker in re.split(r"[;,|]", str(value))
        if marker.strip()
    ]


def _normalize_marker_map(marker_map: dict[str, Any]) -> MarkerMap:
    normalized: MarkerMap = {}
    for cell_type, markers in marker_map.items():
        if isinstance(markers, dict):
            positive = _split_marker_string(
                markers.get("positive", markers.get("positive_markers", markers.get("markers")))
            )
            negative = _split_marker_string(markers.get("negative", markers.get("negative_markers")))
        else:
            positive = _split_marker_string(markers)
            negative = []

        if positive or negative:
            normalized[str(cell_type)] = {
                "positive": positive,
                "negative": negative,
            }
    return normalized


def load_sctype_marker_map(marker_path: Path | str) -> MarkerMap:
    """Load scType marker sets from JSON or CSV."""
    path = Path(marker_path)
    if not path.exists():
        raise FileNotFoundError(f"scType marker file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            data = {
                row["cell_type"]: {
                    "positive": row.get("positive", row.get("positive_markers", row.get("markers", []))),
                    "negative": row.get("negative", row.get("negative_markers", [])),
                }
                for row in data
            }
        return _normalize_marker_map(data)

    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if "cell_type" not in df.columns:
            raise ValueError("scType marker CSV must contain a 'cell_type' column.")
        positive_column = next(
            (column for column in ["positive_markers", "positive", "markers"] if column in df.columns),
            None,
        )
        if positive_column is None:
            raise ValueError(
                "scType marker CSV must contain one of: positive_markers, positive, markers."
            )
        negative_column = next(
            (column for column in ["negative_markers", "negative"] if column in df.columns),
            None,
        )
        return _normalize_marker_map(
            {
                row["cell_type"]: {
                    "positive": row[positive_column],
                    "negative": row[negative_column] if negative_column else [],
                }
                for _, row in df.iterrows()
            }
        )

    raise ValueError(f"Unsupported scType marker file extension: {path.suffix}")


def _resolve_marker_map(config: ScTypeAnnotationConfig) -> MarkerMap:
    if config.marker_path is not None:
        return load_sctype_marker_map(config.marker_path)
    if config.marker_map is not None:
        return _normalize_marker_map(config.marker_map)
    return _normalize_marker_map(DEFAULT_HIAI_AIFI_L2_MARKERS)


def _expression_source(adata, *, use_raw: bool, layer: str | None):
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True was requested, but adata.raw is not available.")
        return adata.raw.to_adata()
    if layer is None:
        return adata
    if layer not in adata.layers:
        raise ValueError(f"Layer {layer!r} not found in AnnData layers.")
    layer_adata = ad.AnnData(
        X=adata.layers[layer],
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    layer_adata.obs_names = adata.obs_names.copy()
    layer_adata.var_names = adata.var_names.copy()
    return layer_adata


def _case_insensitive_gene_lookup(var_names: pd.Index) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for gene in var_names.astype(str):
        lookup.setdefault(gene.upper(), gene)
    return lookup


def _present_markers(markers: list[str], gene_lookup: dict[str, str]) -> list[str]:
    present = []
    seen = set()
    for marker in markers:
        gene = gene_lookup.get(str(marker).upper())
        if gene is not None and gene not in seen:
            present.append(gene)
            seen.add(gene)
    return present


def _dense_marker_matrix(adata, genes: list[str]) -> np.ndarray:
    matrix = adata[:, genes].X
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def _score_margin(score_matrix: np.ndarray, top_indices: np.ndarray) -> np.ndarray:
    margins = np.full(score_matrix.shape[0], np.nan, dtype=np.float32)
    for row_index, top_index in enumerate(top_indices):
        row = score_matrix[row_index]
        finite = row[np.isfinite(row)]
        if len(finite) < 2 or not np.isfinite(row[top_index]):
            continue
        sorted_scores = np.sort(finite)
        margins[row_index] = sorted_scores[-1] - sorted_scores[-2]
    return margins


def run_sctype_annotation(config: ScTypeAnnotationConfig) -> pd.DataFrame:
    """Run a training-free scType-style marker score annotation on query cells."""
    adata_path = Path(config.adata_path)
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData test file not found: {adata_path}")

    query = ad.read_h5ad(adata_path)
    if config.annotation_column not in query.obs:
        raise ValueError(
            f"Annotation column {config.annotation_column!r} not found in query adata.obs."
        )

    expr_adata = _expression_source(query, use_raw=config.use_raw, layer=config.layer)
    marker_map = _resolve_marker_map(config)
    if not marker_map:
        raise ValueError("No scType marker sets were provided.")

    gene_lookup = _case_insensitive_gene_lookup(pd.Index(expr_adata.var_names))
    present_by_label = {
        label: {
            "positive": _present_markers(markers.get("positive", []), gene_lookup),
            "negative": _present_markers(markers.get("negative", []), gene_lookup),
        }
        for label, markers in marker_map.items()
    }
    all_marker_genes = sorted(
        {
            gene
            for markers in present_by_label.values()
            for gene in [*markers["positive"], *markers["negative"]]
        }
    )
    if not all_marker_genes:
        raise ValueError("None of the configured scType marker genes are present in the query AnnData.")

    marker_matrix = _dense_marker_matrix(expr_adata, all_marker_genes)
    if config.scale_expression:
        marker_matrix = _zscore_columns(marker_matrix)
    gene_to_column = {gene: index for index, gene in enumerate(all_marker_genes)}

    labels = list(present_by_label)
    score_matrix = np.full((expr_adata.n_obs, len(labels)), -np.inf, dtype=np.float32)
    positive_counts = np.zeros(len(labels), dtype=np.int32)
    negative_counts = np.zeros(len(labels), dtype=np.int32)

    for label_index, label in enumerate(labels):
        positive_genes = present_by_label[label]["positive"]
        negative_genes = present_by_label[label]["negative"]
        positive_counts[label_index] = len(positive_genes)
        negative_counts[label_index] = len(negative_genes)
        if len(positive_genes) < config.min_positive_markers:
            continue

        positive_indices = [gene_to_column[gene] for gene in positive_genes]
        score = marker_matrix[:, positive_indices].mean(axis=1)
        if negative_genes:
            negative_indices = [gene_to_column[gene] for gene in negative_genes]
            score = score - marker_matrix[:, negative_indices].mean(axis=1)
        score_matrix[:, label_index] = score

    top_indices = np.argmax(score_matrix, axis=1)
    top_scores = score_matrix[np.arange(score_matrix.shape[0]), top_indices]
    margins = _score_margin(score_matrix, top_indices)

    predicted_labels = np.array(labels, dtype=object)[top_indices]
    missing_prediction = ~np.isfinite(top_scores)
    if config.min_score_margin is not None:
        missing_prediction |= margins < float(config.min_score_margin)
    predicted_labels[missing_prediction] = config.unknown_label

    predictions = pd.DataFrame(index=pd.Index(query.obs_names.astype(str), name="cell_id"))
    predictions["true_label"] = query.obs[config.annotation_column].astype(str).to_numpy()
    predictions["predicted_label"] = predicted_labels.astype(str)
    predictions["score"] = np.where(np.isfinite(top_scores), top_scores, np.nan).astype(float)
    predictions["score_margin"] = margins.astype(float)
    predictions["n_present_positive_markers"] = positive_counts[top_indices]
    predictions["n_present_negative_markers"] = negative_counts[top_indices]
    predictions["model_name"] = config.model_name
    return predictions
