from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from Bio import Entrez
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer

from .cl import DataCLConfig, collect_cl_terms


@dataclass
class DataNCBIConfig:
    email: str
    dataset_id: Optional[str] = None
    output_dir: Optional[str | Path] = None
    save_artifacts: bool = False
    raw_articles_path: Optional[str | Path] = None
    fetch_if_missing: bool = True
    random_state: int = 42
    organism: str = "homo sapiens"
    query_mode: str = "broad_mesh"
    query_field: str = "Title/Abstract"
    query_include_plural: bool = True
    tissue: Optional[str] = None
    max_articles: int = 100
    batch_size: int = 250
    max_retries: int = 3
    model: str = "neuml/pubmedbert-base-embeddings"
    max_tokens: int = 512
    overlap: int = 20
    diseases: Optional[List[str]] = None
    semantic: bool = False
    celltypes_from_adata: bool = True
    celltypes_list: Optional[List[str]] = None
    annotation_column: str = "celltype"
    test_split: Optional[float] = None
    heldout_values: Optional[List[str]] = None
    heldout_key: str = "cell_types"
    remove_multilabel_pmids: bool = False
    shuffle_labels: bool = False
    label_shuffle_seed: Optional[int] = None
    label_shuffle_original_column: str = "original_label"
    collect_cl_terms: bool = False
    cl_description_path: Optional[str | Path] = None
    cl_raw_path: Optional[str | Path] = None
    cl_split_descriptions: bool = True
    cl_min_sentence_words: int = 8
    cl_max_description_sentences_per_label: Optional[int] = None
    cl_marker_map: Optional[dict[str, Any]] = None
    cl_infusion_mode: str = "native"


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _safe_config(config: DataNCBIConfig) -> dict[str, Any]:
    return {key: _safe_json_value(value) for key, value in asdict(config).items()}


def _read_raw_articles(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), dtype={"PMID": str})


def _write_raw_articles(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _celltypes_from_source(adata, cfg: Dict) -> list[str]:
    annotation_column = cfg["annotation_column"]
    if cfg["celltypes_from_adata"] and adata is not None:
        return sorted(adata.obs[annotation_column].dropna().astype(str).unique().tolist())
    return sorted(map(str, cfg.get("celltypes_list") or []))


def _extract_abstract_text(abstract_value) -> str:
    if isinstance(abstract_value, list):
        return " ".join(map(str, abstract_value))
    return str(abstract_value)


def _quote_pubmed_phrase(value: str) -> str:
    return str(value).replace('"', r"\"")


def _phrase_variants(celltype: str, *, include_plural: bool = True) -> list[str]:
    singular = str(celltype).strip()
    variants = [singular]
    if include_plural and singular and not singular.lower().endswith("s"):
        variants.append(f"{singular}s")
    return variants


def _mesh_constraint(term: str) -> str:
    clean_term = str(term).strip()
    if not clean_term:
        return ""
    if "[" in clean_term and "]" in clean_term:
        return clean_term
    return f"{clean_term}[Mesh]"


def _query_prefix(cfg: Dict) -> str:
    constraints = [f"{cfg['organism']}[Mesh]"]
    tissue = cfg.get("tissue")
    if tissue:
        constraints.append(_mesh_constraint(tissue))
    return " AND ".join(f"({constraint})" for constraint in constraints)


def _celltype_query(celltype: str, cfg: Dict) -> str:
    query_mode = cfg.get("query_mode", "broad_mesh")
    query_prefix = _query_prefix(cfg)

    if query_mode in {"broad_mesh", "mesh_broad"}:
        return f"{query_prefix} AND {celltype}"

    if query_mode in {"exact_title_abstract", "exact_phrase_title_abstract"}:
        field = cfg.get("query_field", "Title/Abstract")
        phrases = _phrase_variants(
            celltype,
            include_plural=bool(cfg.get("query_include_plural", True)),
        )
        phrase_query = " OR ".join(
            f'"{_quote_pubmed_phrase(phrase)}"[{field}]' for phrase in phrases
        )
        return f"{query_prefix} AND ({phrase_query})"

    raise ValueError(
        "Unsupported NCBI query_mode. Expected 'broad_mesh' or "
        f"'exact_title_abstract', got {query_mode!r}."
    )


def build_pubmed_queries(
    celltype: str,
    config: DataNCBIConfig | Dict,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    cfg = asdict(config) if isinstance(config, DataNCBIConfig) else dict(config)
    base_query = _celltype_query(celltype, cfg)

    queries, query_to_label, query_to_disease = [], {}, {}
    if cfg.get("diseases"):
        for disease in cfg["diseases"]:
            q_and = f"({base_query} AND {disease})"
            q_not = f"({base_query} NOT {disease})"
            queries += [q_and, q_not]
            query_to_label[q_and], query_to_label[q_not] = f"{celltype}_{disease}", celltype
            query_to_disease[q_and], query_to_disease[q_not] = disease, ""
    else:
        queries = [base_query]
        query_to_label[base_query] = celltype
        query_to_disease[base_query] = ""

    return queries, query_to_label, query_to_disease


def fetch_articles(celltype: str, cfg: Dict) -> pd.DataFrame:
    """Fetch PubMed titles and abstracts for a given cell type."""
    Entrez.email = cfg["email"]
    queries, query_to_label, query_to_disease = build_pubmed_queries(celltype, cfg)

    all_abstracts = {}

    for query_index, query in enumerate(queries):
        handle = Entrez.esearch(db="pubmed", term=query, retmax=cfg["max_articles"])
        record = Entrez.read(handle)
        handle.close()
        pmids = record.get("IdList", [])
        n_pmids_found = int(record.get("Count", len(pmids)))
        n_pmids_returned = len(pmids)
        print(f"{n_pmids_returned} PMIDs for query: {query}")

        for i in tqdm(range(0, len(pmids), cfg["batch_size"])):
            batch = pmids[i : i + cfg["batch_size"]]
            for attempt in range(cfg["max_retries"]):
                try:
                    handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml", retmode="text")
                    records = Entrez.read(handle)
                    handle.close()

                    for article in records.get("PubmedArticle", []):
                        pmid = str(article["MedlineCitation"]["PMID"])
                        article_key = (pmid, query)
                        if article_key in all_abstracts:
                            continue
                        art = article["MedlineCitation"]["Article"]
                        title = art.get("ArticleTitle", "No Title Available")
                        abstract = _extract_abstract_text(
                            art.get("Abstract", {}).get("AbstractText", ["No Abstract"])
                        )
                        all_abstracts[article_key] = {
                            "PMID": pmid,
                            "Title": title,
                            "Abstract": abstract,
                            "Query": query,
                            "label": query_to_label[query],
                            "disease": query_to_disease[query],
                            "n_pmids_found": n_pmids_found,
                            "n_pmids_returned": n_pmids_returned,
                            "query_index": query_index,
                        }

                    time.sleep(random.uniform(1, 3))
                    break
                except Exception as e:
                    print(f"Retry {attempt+1}/{cfg['max_retries']} for batch {i}: {e}")
                    time.sleep(2 ** attempt)

    return pd.DataFrame(list(all_abstracts.values()))


def build_ncbi_raw_articles(
    adata=None,
    ncbi_config: DataNCBIConfig | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Collect or load reusable raw NCBI article rows before variant processing."""
    if ncbi_config is None:
        raise ValueError("ncbi_config is required.")

    cfg = asdict(ncbi_config)
    cfg.update(kwargs)
    annotation_column = cfg["annotation_column"]
    raw_articles_path = cfg.get("raw_articles_path")

    if raw_articles_path and Path(raw_articles_path).exists():
        return _read_raw_articles(raw_articles_path)

    if raw_articles_path and not cfg.get("fetch_if_missing", True):
        raise FileNotFoundError(
            f"raw_articles_path does not exist and fetch_if_missing=False: {raw_articles_path}"
        )

    all_dfs = []
    for ct in _celltypes_from_source(adata, cfg):
        df = fetch_articles(ct, cfg)
        if not df.empty:
            df[annotation_column] = ct
            all_dfs.append(df)
        else:
            print(f"No NCBI articles found for cell type '{ct}'")

    if not all_dfs:
        raw_df = pd.DataFrame(
            columns=[
                "PMID",
                "Title",
                "Abstract",
                "Query",
                "label",
                "disease",
                annotation_column,
                "n_pmids_found",
                "n_pmids_returned",
                "query_index",
            ]
        )
    else:
        raw_df = pd.concat(all_dfs, ignore_index=True)
        subset = [col for col in ["PMID", "label", "Query"] if col in raw_df.columns]
        if subset:
            raw_df = raw_df.drop_duplicates(subset=subset).reset_index(drop=True)

    if raw_articles_path:
        _write_raw_articles(raw_df, raw_articles_path)

    return raw_df


def apply_ncbi_heldout_filter(
    raw_df: pd.DataFrame,
    *,
    annotation_column: str,
    heldout_values: Optional[List[str]],
) -> pd.DataFrame:
    if not heldout_values:
        return raw_df.copy()

    heldout = set(map(str, heldout_values))
    filter_column = annotation_column if annotation_column in raw_df.columns else "label"
    mask = ~raw_df[filter_column].astype(str).isin(heldout)
    return raw_df.loc[mask].reset_index(drop=True)


def apply_ncbi_multilabel_pmid_filter(
    raw_df: pd.DataFrame,
    *,
    pmid_column: str = "PMID",
    label_column: str = "label",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove PMIDs that are associated with more than one distinct label."""
    if raw_df.empty or pmid_column not in raw_df.columns or label_column not in raw_df.columns:
        return raw_df.copy(), {
            "n_removed_by_multilabel_filter": 0,
            "n_removed_multilabel_pmids": 0,
        }

    n_labels_by_pmid = raw_df.groupby(pmid_column)[label_column].nunique()
    multilabel_pmids = set(n_labels_by_pmid[n_labels_by_pmid > 1].index)
    if not multilabel_pmids:
        return raw_df.copy(), {
            "n_removed_by_multilabel_filter": 0,
            "n_removed_multilabel_pmids": 0,
        }

    filtered = raw_df.loc[~raw_df[pmid_column].isin(multilabel_pmids)].reset_index(drop=True)
    return filtered, {
        "n_removed_by_multilabel_filter": int(len(raw_df) - len(filtered)),
        "n_removed_multilabel_pmids": int(len(multilabel_pmids)),
    }


def split_text_by_tokens(tokenizer, text: str, cfg: Dict) -> Tuple[List[str], List[int]]:
    """Split text into overlapping chunks based on max_tokens and overlap."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    step = cfg["max_tokens"] - cfg["overlap"]
    chunks = [tokens[i : i + cfg["max_tokens"]] for i in range(0, len(tokens), step)]
    decoded = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
    return decoded, [len(c) for c in chunks]


def process_ncbi_df(df: pd.DataFrame, annotation_column: str, cfg: Dict) -> pd.DataFrame:
    """Convert abstracts and titles into a training DataFrame."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    records = []

    for _, row in df.iterrows():
        for field in ["Abstract", "Title"]:
            text = row.get(field)
            if not text:
                continue
            substrings, lengths = split_text_by_tokens(tokenizer, text, cfg)
            for s, l in zip(substrings, lengths):
                rec = {
                    "sentence1": s,
                    "token_length": l,
                    annotation_column: row.get(annotation_column, None),
                    "PMID": row.get("PMID"),
                    "Title": row.get("Title"),
                    "Abstract": row.get("Abstract"),
                    "Query": row.get("Query"),
                    "type": field.lower(),
                    "label": row.get("label"),
                    "disease": row.get("disease"),
                    "source": row.get("source", "NCBI"),
                }
                records.append(rec)

    return pd.DataFrame(records)


def _collect_cl_rows_for_ncbi(
    *,
    labels: list[str],
    annotation_column: str,
    cfg: Dict,
) -> pd.DataFrame:
    if not cfg.get("collect_cl_terms", False):
        return pd.DataFrame()

    description_path = cfg.get("cl_description_path")
    if not description_path:
        raise ValueError("collect_cl_terms=True requires cl_description_path.")

    return collect_cl_terms(
        DataCLConfig(
            description_path=description_path,
            raw_cl_path=cfg.get("cl_raw_path"),
            labels=sorted(set(map(str, labels))),
            annotation_column=annotation_column,
            split_descriptions=bool(cfg.get("cl_split_descriptions", True)),
            min_sentence_words=int(cfg.get("cl_min_sentence_words", 8)),
            max_description_sentences_per_label=cfg.get(
                "cl_max_description_sentences_per_label"
            ),
            marker_map=cfg.get("cl_marker_map"),
        )
    )


def apply_cl_infusion_mode(cl_df: pd.DataFrame, *, mode: str = "native") -> pd.DataFrame:
    """Map raw CL knowledge rows into NCBI-compatible text pools."""
    if cl_df.empty:
        return cl_df.copy()

    mode = str(mode or "native")
    if mode == "native":
        return cl_df.copy()

    if mode == "mapped":
        infused = cl_df.copy()
        infused["cl_original_type"] = infused["type"]
        title_like = {"cl_definition", "sctype_positive_marker_genes"}
        abstract_like = {"cl_description", "cl_description_sentence"}
        infused["type"] = infused["type"].map(
            lambda value: "title" if value in title_like else "abstract"
            if value in abstract_like
            else value
        )
        return infused

    if mode == "title_abstract":
        copies = []
        for target_type in ["title", "abstract"]:
            copy = cl_df.copy()
            copy["cl_original_type"] = copy["type"]
            copy["type"] = target_type
            copy["cl_infusion_type"] = target_type
            copies.append(copy)
        return pd.concat(copies, ignore_index=True, sort=False)

    raise ValueError(
        "Unsupported cl_infusion_mode. Expected 'native', 'mapped', or "
        f"'title_abstract', got {mode!r}."
    )


def shuffle_ncbi_labels(
    df: pd.DataFrame,
    *,
    seed: int,
    original_column: str = "original_label",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preserve content/order and deterministically permute only the label column."""
    if "label" not in df.columns:
        raise ValueError("shuffle_labels=True requires a 'label' column.")
    if original_column in df.columns:
        raise ValueError(f"Cannot store original labels in existing column: {original_column}")

    shuffled = df.copy()
    original_labels = shuffled["label"].astype(str).to_numpy()
    shuffled_labels = np.random.default_rng(seed).permutation(original_labels)
    shuffled[original_column] = original_labels
    shuffled["label"] = shuffled_labels

    changed = shuffled["label"].astype(str) != shuffled[original_column].astype(str)
    return shuffled, {
        "shuffle_labels": True,
        "label_shuffle_seed": seed,
        "label_shuffle_original_column": original_column,
        "n_rows": int(len(shuffled)),
        "n_changed_labels": int(changed.sum()),
        "fraction_changed_labels": float(changed.mean()) if len(shuffled) else 0.0,
    }


def _empty_dataset() -> Dataset:
    return Dataset.from_dict({})


def _is_dataset_nonempty(ds) -> bool:
    return isinstance(ds, Dataset) and len(ds) > 0


def _write_article_counts(raw_df: pd.DataFrame, reports_dir: Path, annotation_column: str) -> dict[str, str]:
    raw_df = raw_df.copy()
    for column in ["n_pmids_found", "n_pmids_returned"]:
        if column not in raw_df.columns:
            raw_df[column] = np.nan

    if raw_df.empty:
        counts = pd.DataFrame(columns=[annotation_column, "n_article_rows", "n_unique_pmids"])
    else:
        counts = (
            raw_df.groupby(annotation_column, dropna=False)
            .agg(
                n_article_rows=("PMID", "size"),
                n_unique_pmids=("PMID", "nunique"),
                n_pmids_found=("n_pmids_found", "max"),
                n_pmids_returned=("n_pmids_returned", "max"),
            )
            .reset_index()
            .sort_values(annotation_column)
        )

    counts_path = reports_dir / "article_counts_by_celltype.csv"
    counts.to_csv(counts_path, index=False)

    figure_path = reports_dir / "article_counts_by_celltype.pdf"
    plt.figure(figsize=(max(6, 0.35 * max(len(counts), 1)), 4))
    if not counts.empty:
        plot_counts = counts.sort_values("n_article_rows", ascending=False)
        plt.bar(plot_counts[annotation_column].astype(str), plot_counts["n_article_rows"])
        plt.xticks(rotation=90)
    plt.ylabel("article rows")
    plt.xlabel(annotation_column)
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()

    return {
        "article_counts_by_celltype": str(counts_path),
        "article_counts_by_celltype_pdf": str(figure_path),
    }


def _write_collection_summary(raw_df: pd.DataFrame, reports_dir: Path, annotation_column: str) -> dict[str, str]:
    raw_df = raw_df.copy()
    for column in ["n_pmids_found", "n_pmids_returned"]:
        if column not in raw_df.columns:
            raw_df[column] = np.nan

    if raw_df.empty:
        summary = pd.DataFrame(
            columns=[
                annotation_column,
                "label",
                "Query",
                "n_article_rows",
                "n_unique_pmids",
                "n_pmids_found",
                "n_pmids_returned",
            ]
        )
    else:
        summary = (
            raw_df.groupby([annotation_column, "label", "Query"], dropna=False)
            .agg(
                n_article_rows=("PMID", "size"),
                n_unique_pmids=("PMID", "nunique"),
                n_pmids_found=("n_pmids_found", "max"),
                n_pmids_returned=("n_pmids_returned", "max"),
            )
            .reset_index()
            .sort_values([annotation_column, "label", "Query"])
        )

    summary_path = reports_dir / "article_collection_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {"article_collection_summary": str(summary_path)}


def _write_label_shuffle_report(
    processed_df: pd.DataFrame,
    reports_dir: Path,
    *,
    original_column: str,
) -> dict[str, str]:
    report = (
        processed_df.groupby([original_column, "label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={original_column: "original_label", "label": "shuffled_label"})
    )
    original_totals = report.groupby("original_label")["count"].transform("sum")
    report["proportion_within_original_label"] = report["count"] / original_totals
    report["correct_label"] = (
        report["original_label"].astype(str) == report["shuffled_label"].astype(str)
    )
    report_path = reports_dir / "label_shuffle_report.csv"
    report.to_csv(report_path, index=False)

    matrix = pd.crosstab(
        processed_df[original_column].astype(str),
        processed_df["label"].astype(str),
    )
    figure_path = reports_dir / "label_shuffle_confusion.pdf"
    plt.figure(figsize=(max(6, 0.35 * matrix.shape[1]), max(4, 0.35 * matrix.shape[0])))
    sns.heatmap(matrix, cmap="viridis", cbar_kws={"label": "count"})
    plt.xlabel("Shuffled label")
    plt.ylabel("Original label")
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()

    return {
        "label_shuffle_report": str(report_path),
        "label_shuffle_confusion": str(figure_path),
    }


def save_ncbi_dataset_artifacts(
    *,
    train_ds: Dataset,
    test_ds,
    raw_articles: pd.DataFrame,
    processed_df: pd.DataFrame,
    ncbi_config: DataNCBIConfig,
    variant_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, dict[str, str]]:
    if ncbi_config.output_dir is None:
        raise ValueError("ncbi_config.output_dir is required when save_artifacts=True.")

    output_dir = Path(ncbi_config.output_dir)
    datasets_dir = output_dir / "datasets"
    reports_dir = output_dir / "reports"
    metadata_dir = output_dir / "metadata"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    dataset_artifacts = {"ncbi_data": str(datasets_dir / "ncbi_data")}
    train_ds.save_to_disk(dataset_artifacts["ncbi_data"])
    if _is_dataset_nonempty(test_ds):
        dataset_artifacts["ncbi_test"] = str(datasets_dir / "ncbi_test")
        test_ds.save_to_disk(dataset_artifacts["ncbi_test"])

    train_head_path = reports_dir / "ncbi_train_head.csv"
    train_head_count = min(5, len(train_ds))
    if train_head_count:
        train_ds.select(range(train_head_count)).to_pandas().to_csv(train_head_path, index=False)
    else:
        pd.DataFrame().to_csv(train_head_path, index=False)

    report_artifacts = {"ncbi_train_head": str(train_head_path)}
    report_artifacts.update(
        _write_article_counts(raw_articles, reports_dir, ncbi_config.annotation_column)
    )
    report_artifacts.update(
        _write_collection_summary(raw_articles, reports_dir, ncbi_config.annotation_column)
    )
    if ncbi_config.shuffle_labels and ncbi_config.label_shuffle_original_column in processed_df.columns:
        report_artifacts.update(
            _write_label_shuffle_report(
                processed_df,
                reports_dir,
                original_column=ncbi_config.label_shuffle_original_column,
            )
        )

    raw_path = ncbi_config.raw_articles_path
    metadata = {
        "dataset_id": ncbi_config.dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(raw_path) if raw_path else None,
        "config": _safe_config(ncbi_config),
        "n_raw_article_rows": int(len(raw_articles)),
        "n_processed_rows": int(len(processed_df)),
        "n_train_rows": int(len(train_ds)),
        "n_test_rows": int(len(test_ds)) if _is_dataset_nonempty(test_ds) else 0,
        "heldout_values": {
            ncbi_config.heldout_key: sorted(map(str, ncbi_config.heldout_values or []))
        },
        "shuffle_labels": bool(ncbi_config.shuffle_labels),
        "label_shuffle_seed": (
            ncbi_config.label_shuffle_seed
            if ncbi_config.label_shuffle_seed is not None
            else ncbi_config.random_state
        ),
        "dataset_artifacts": dataset_artifacts,
        "report_artifacts": report_artifacts,
    }
    if variant_metadata:
        metadata.update(_safe_json_value(variant_metadata))

    with (metadata_dir / "generation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return {
        "dataset_artifacts": dataset_artifacts,
        "report_artifacts": report_artifacts,
    }


def gen_ncbi_dataset(
    adata=None, ncbi_config: DataNCBIConfig = None, **kwargs
) -> Tuple[Dataset, Dataset]:
    """Generate train/test HuggingFace Datasets from AnnData or config, overridable with kwargs."""
    if ncbi_config is None:
        raise ValueError("ncbi_config is required.")

    cfg = asdict(ncbi_config)
    cfg.update(kwargs)
    annotation_column = cfg["annotation_column"]
    source_labels = _celltypes_from_source(adata, cfg)

    raw_df = build_ncbi_raw_articles(adata, ncbi_config, **kwargs)
    variant_raw_df = apply_ncbi_heldout_filter(
        raw_df,
        annotation_column=annotation_column,
        heldout_values=cfg.get("heldout_values"),
    )
    n_removed_by_heldout_filter = int(len(raw_df) - len(variant_raw_df))
    multilabel_filter_metadata = {
        "n_removed_by_multilabel_filter": 0,
        "n_removed_multilabel_pmids": 0,
    }
    if cfg.get("remove_multilabel_pmids", False):
        variant_raw_df, multilabel_filter_metadata = apply_ncbi_multilabel_pmid_filter(
            variant_raw_df,
            pmid_column="PMID",
            label_column="label",
        )
    heldout = set(map(str, cfg.get("heldout_values") or []))
    cl_labels = [label for label in source_labels if str(label) not in heldout]
    cl_df = _collect_cl_rows_for_ncbi(
        labels=cl_labels,
        annotation_column=annotation_column,
        cfg=cfg,
    )
    cl_raw_row_count = int(len(cl_df))
    cl_df_for_training = apply_cl_infusion_mode(
        cl_df,
        mode=cfg.get("cl_infusion_mode", "native"),
    )

    if variant_raw_df.empty and cl_df_for_training.empty:
        print("No data collected. Returning empty Datasets.")
        train_ds, test_ds = _empty_dataset(), _empty_dataset()
        if cfg.get("save_artifacts", False):
            save_ncbi_dataset_artifacts(
                train_ds=train_ds,
                test_ds=test_ds,
                raw_articles=variant_raw_df,
                processed_df=pd.DataFrame(),
                ncbi_config=ncbi_config,
                variant_metadata={
                    "n_removed_by_heldout_filter": n_removed_by_heldout_filter,
                    **multilabel_filter_metadata,
                    "n_cl_raw_rows": 0,
                    "n_cl_rows": 0,
                    "cl_infusion_mode": cfg.get("cl_infusion_mode", "native"),
                },
            )
        return train_ds, test_ds

    if variant_raw_df.empty:
        df_processed = pd.DataFrame()
    else:
        df_processed = process_ncbi_df(variant_raw_df, annotation_column, cfg)
    if not cl_df_for_training.empty:
        df_processed = pd.concat(
            [df_processed, cl_df_for_training],
            ignore_index=True,
            sort=False,
        )

    label_shuffle_metadata = None
    if cfg.get("shuffle_labels", False):
        seed = cfg.get("label_shuffle_seed")
        if seed is None:
            seed = cfg["random_state"]
        df_processed, label_shuffle_metadata = shuffle_ncbi_labels(
            df_processed,
            seed=seed,
            original_column=cfg.get("label_shuffle_original_column") or "original_label",
        )

    ds = Dataset.from_pandas(df_processed.reset_index(), preserve_index=False)

    if cfg["test_split"] is not None:
        ds_split = ds.train_test_split(test_size=cfg["test_split"], seed=cfg["random_state"])
        train_ds = ds_split["train"]
        test_ds = ds_split["test"]
    else:
        train_ds = ds
        test_ds = _empty_dataset()

    if cfg.get("save_artifacts", False):
        save_ncbi_dataset_artifacts(
            train_ds=train_ds,
            test_ds=test_ds,
            raw_articles=variant_raw_df,
            processed_df=df_processed,
            ncbi_config=ncbi_config,
            variant_metadata={
                "n_removed_by_heldout_filter": n_removed_by_heldout_filter,
                **multilabel_filter_metadata,
                "label_shuffle": label_shuffle_metadata,
                "n_cl_raw_rows": cl_raw_row_count,
                "n_cl_rows": int(len(cl_df_for_training)),
                "cl_infusion_mode": cfg.get("cl_infusion_mode", "native"),
                "cl_description_path": (
                    str(cfg.get("cl_description_path"))
                    if cfg.get("cl_description_path")
                    else None
                ),
                "cl_raw_path": str(cfg.get("cl_raw_path")) if cfg.get("cl_raw_path") else None,
            },
        )

    return train_ds, test_ds
