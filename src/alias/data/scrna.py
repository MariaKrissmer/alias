
from dataclasses import dataclass, asdict
from datasets import Dataset
from typing import Optional, List, Tuple
from tqdm import tqdm
from scipy import sparse
import numpy as np
import pandas as pd
import scanpy as sc
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from alias.util.cell_sentence_templates import TEMPLATES
from alias.data.scrna_splits import (
    SplitIndices,
    make_heldout_value_split,
    make_proportional_heldout_group_split,
    make_proportional_heldout_donor_and_value_split,
    make_proportional_heldout_donor_split,
    make_random_stratified_split,
    validate_split_indices,
    write_generation_metadata,
    write_split_indices,
    write_split_report,
)


@dataclass
class DatascRNAConfig:
    """Configuration for scRNA dataset generation."""
    dataset_id: Optional[str] = None
    annotation_column: str = "celltype"
    preprocessing: bool = False
    random_state: int = 42
    
    # --- CELL SENTENCE PARAMETERS ---
    cs_length: List[int] = (10,)
    disease_column: Optional[str] = None
    disease_value_map: Optional[dict[str, str]] = None
    disease_output_column: str = "disease_status"
    time_column: Optional[str] = None
    highly_variable_genes: bool = True
    housekeeping_genes: bool = True
    semantic: bool = True
    
    # --- SEMANTIC TEMPLATE WEIGHTS ---
    template_weights_default: Optional[dict] = None
    template_weights_disease: Optional[dict] = None
    template_weights_time: Optional[dict] = None
    
    # --- DATASET PARAMETERS ---
    test_size: float = 0.1
    total_cells: Optional[int] = None
    split_strategy: Optional[str] = None
    donor_column: Optional[str] = None
    donor_test_size: Optional[float] = None
    group_column: Optional[str] = None
    group_test_size: Optional[float] = None
    group_key: str = "groups"
    report_columns: Optional[List[str]] = None
    stratified_subsample: bool = True
    heldout_column: Optional[str] = None
    heldout_values: Optional[List[str]] = None
    heldout_key: str = "values"
    nonsemantic_train_column: Optional[str] = None
    nonsemantic_train_values: Optional[List[str]] = None
    nonsemantic_train_key: str = "values"
    shuffle_train_labels: bool = False
    label_shuffle_seed: Optional[int] = None
    label_shuffle_original_column: str = "original_label"
    output_dir: Optional[str | Path] = None
    source: Optional[str] = None
    save_artifacts: bool = False
    
    # --- PREPROCESSING PARAMETERS ---
    hvg_number: int = 3000
    min_genes: int = 100
    mt_threshold: float = 15
    min_cells: int = 5
    min_batch_cells: int = 5
    verbose: bool = True
    
def run_preprocessing(adata, batch_key, scrna_config: DatascRNAConfig, **kwargs):
    """Optional preprocessing of AnnData object using config parameters, overridable with kwargs."""
    cfg = asdict(scrna_config)
    cfg.update(kwargs) 
    
    verbose = cfg['verbose']

    if verbose:
        print("Preprocessing AnnData object...")

    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    genes_upper = adata.var_names.str.upper()
    adata.var["mt"] = genes_upper.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
        
    # Filter cells and genes
    if verbose:
        print(f"Filtering cells with fewer than {cfg['min_genes']} expressed genes...")
    sc.pp.filter_cells(adata, min_genes=cfg['min_genes'])

    if verbose:
        print(f"Filtering cells with a percentage of mitochondrial genes expressed over {cfg['mt_threshold']}...")
    adata = adata[adata.obs["pct_counts_mt"] < cfg['mt_threshold']].copy()
    
    if verbose:
        print(f"Filtering genes expressed in fewer than {cfg['min_cells']} cells...")
    sc.pp.filter_genes(adata, min_cells=cfg['min_cells'])

    # Remove small batches
    if verbose:
        print("Analyzing cell type sizes...")
        print(f"Initial number of celtypes: {adata.obs[batch_key].nunique()}")
    batch_sizes = adata.obs[batch_key].value_counts()
    valid_batches = batch_sizes[batch_sizes >= cfg['min_batch_cells']].index
    if verbose:
        print(f"Batches with fewer than {cfg['min_batch_cells']} cells: {len(batch_sizes) - len(valid_batches)}")
        print("Filtering small batches...")
    adata = adata[adata.obs[batch_key].isin(valid_batches)].copy()
    sc.pp.filter_genes(adata, min_cells=cfg['min_cells'])

    # Save counts to separate layer
    adata.layers["counts"] = adata.X.copy()

    # Normalize, log transform, and identify highly variable genes
    if not adata.uns.get("normalized", False):
        if verbose:
            print("Normalizing total counts per cell...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        adata.uns["normalized"] = True

    if not adata.uns.get("logged", False):
        if verbose:
            print("Performing log transformation...")
        sc.pp.log1p(adata)
        adata.uns["logged"] = True
    
    if verbose:
        print(f"Identifying {cfg['hvg_number']} highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=cfg['hvg_number'])
    
    if verbose:
        print("Subsetting to highly variable genes...")
    adata = adata[:, adata.var.highly_variable].copy()

    if verbose:
        print("Preprocessing complete. Your AnnData object is ready for further analysis.")
        print(adata)
    return adata

def generate_semantic_sentence(
    gene_list, cell_type,
    scrna_config=None, time=None, disease_status=None, **kwargs
):
    """Create a semantic sentence for each cell, with user-controllable template weights."""
    cfg = asdict(scrna_config) if scrna_config else {}
    cfg.update(kwargs)  # override with kwargs

    gene_str = ", ".join(gene_list)

    # Determine weights
    default_disease = {
        'genes_celltype_disease': 0.5,
        'genes_disease': 0.2,
        'genes_celltype': 0.2,
        'genes': 0.1
    }
    default_time = {
        'genes_celltype_time': 0.5,
        'genes_time': 0.2,
        'genes_celltype': 0.2,
        'genes': 0.1
    }
    default_default = {
        'genes_celltype': 0.7 if cell_type else 0.0,
        'genes': 0.3 if cell_type else 1.0
    }

    # pick which template_weights to use
    if cfg.get('disease_column'):
        template_weights = cfg.get('template_weights_disease') or default_disease
    elif cfg.get('time_column'):
        template_weights = cfg.get('template_weights_time') or default_time
    else:
        template_weights = cfg.get('template_weights_default') or default_default

    categories, weights = zip(*template_weights.items())
    selected_category = random.choices(categories, weights=weights, k=1)[0]
    template = random.choice(TEMPLATES[selected_category])

    return template.format(
        gene_str=gene_str,
        cell_type=cell_type,
        time=time,
        disease_status=disease_status
    )


def process_split(ds, annotation_column, scrna_config, semantic, **kwargs):
    """Add sentence1 column to HF dataset, overridable with kwargs."""
    cfg = asdict(scrna_config)
    cfg.update(kwargs)
    disease_output_column = cfg.get("disease_output_column") or "disease_status"
    nonsemantic_column = cfg.get("nonsemantic_train_column")
    nonsemantic_values = cfg.get("nonsemantic_train_values") or []
    nonsemantic_values = set(map(str, nonsemantic_values))
    mixed_sentence_modes = bool(nonsemantic_column and nonsemantic_values)

    def _gene_sentence(row):
        return " ".join(row["gene_list"])

    def _semantic_sentence(row):
        return generate_semantic_sentence(
            row["gene_list"],
            row.get(annotation_column),
            scrna_config=scrna_config,
            time=row.get("time"),
            disease_status=row.get(disease_output_column),
            **kwargs
        )

    if semantic:
        if mixed_sentence_modes:
            ds = ds.map(
                lambda row: {
                    "sentence1": (
                        _gene_sentence(row)
                        if str(row.get(nonsemantic_column)) in nonsemantic_values
                        else _semantic_sentence(row)
                    ),
                    "sentence_mode": (
                        "nonsemantic"
                        if str(row.get(nonsemantic_column)) in nonsemantic_values
                        else "semantic"
                    ),
                }
            )
        else:
            ds = ds.map(lambda row: {"sentence1": _semantic_sentence(row)})
    else:
        if mixed_sentence_modes:
            ds = ds.map(lambda row: {"sentence1": _gene_sentence(row), "sentence_mode": "nonsemantic"})
        else:
            ds = ds.map(lambda row: {"sentence1": _gene_sentence(row)})
    return ds


def _shuffle_train_labels(
    train_ds: Dataset,
    scrna_config: DatascRNAConfig,
    **kwargs,
) -> Dataset:
    cfg = asdict(scrna_config)
    cfg.update(kwargs)
    if not cfg.get("shuffle_train_labels", False):
        return train_ds

    if "label" not in train_ds.features:
        raise ValueError("shuffle_train_labels=True requires a 'label' column.")

    original_column = cfg.get("label_shuffle_original_column") or "original_label"
    if original_column in train_ds.features:
        raise ValueError(
            f"Cannot store original labels in existing column: {original_column}"
        )

    seed = cfg.get("label_shuffle_seed")
    if seed is None:
        seed = cfg["random_state"]

    train_df = train_ds.to_pandas()
    original_labels = train_df["label"].astype(str).to_numpy()
    shuffled_labels = np.random.default_rng(seed).permutation(original_labels)
    train_df[original_column] = original_labels
    train_df["label"] = shuffled_labels
    return Dataset.from_pandas(train_df, preserve_index=False)


def _label_shuffle_metadata(
    train_ds: Dataset,
    scrna_config: DatascRNAConfig,
) -> dict[str, int | float | str | bool] | None:
    if not scrna_config.shuffle_train_labels:
        return None

    original_column = scrna_config.label_shuffle_original_column
    if original_column not in train_ds.features:
        return None

    train_df = train_ds.to_pandas()
    changed = train_df["label"].astype(str) != train_df[original_column].astype(str)
    n_train_rows = len(train_df)
    return {
        "shuffle_train_labels": True,
        "label_shuffle_seed": (
            scrna_config.label_shuffle_seed
            if scrna_config.label_shuffle_seed is not None
            else scrna_config.random_state
        ),
        "label_shuffle_original_column": original_column,
        "n_train_rows": n_train_rows,
        "n_changed_labels": int(changed.sum()),
        "fraction_changed_labels": float(changed.mean()) if n_train_rows else 0.0,
    }


def _write_label_shuffle_report(
    train_ds: Dataset,
    *,
    output_dir: str | Path,
    original_column: str,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = train_ds.to_pandas()
    report = (
        train_df.groupby([original_column, "label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(
            columns={
                original_column: "original_label",
                "label": "shuffled_label",
            }
        )
    )
    original_totals = report.groupby("original_label")["count"].transform("sum")
    report["proportion_within_original_label"] = report["count"] / original_totals
    report["correct_label"] = report["original_label"].astype(str) == report["shuffled_label"].astype(str)
    report_path = output_path / "label_shuffle_report.csv"
    report.to_csv(report_path, index=False)

    matrix = pd.crosstab(
        train_df[original_column].astype(str),
        train_df["label"].astype(str),
    )
    figure_path = output_path / "label_shuffle_confusion.pdf"
    plt.figure(figsize=(max(6, 0.35 * matrix.shape[1]), max(4, 0.35 * matrix.shape[0])))
    sns.heatmap(matrix, cmap="viridis", cbar_kws={"label": "count"})
    plt.xlabel("Shuffled label")
    plt.ylabel("Original label")
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()

    proportion_matrix = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    proportion_figure_path = output_path / "label_shuffle_correct_label_proportions.pdf"
    plt.figure(
        figsize=(
            max(6, 0.35 * proportion_matrix.shape[1]),
            max(4, 0.35 * proportion_matrix.shape[0]),
        )
    )
    sns.heatmap(
        proportion_matrix,
        cmap="mako",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "proportion within original label"},
    )
    plt.xlabel("Shuffled label")
    plt.ylabel("Original label")
    plt.title("Proportion of cells retaining the correct label after shuffling")
    plt.tight_layout()
    plt.savefig(proportion_figure_path, bbox_inches="tight")
    plt.close()

    return {
        "label_shuffle_report": str(report_path),
        "label_shuffle_confusion": str(figure_path),
        "label_shuffle_correct_label_proportions": str(proportion_figure_path),
    }


def _map_config_value(value, value_map: dict[str, str] | None):
    if pd.isnull(value):
        return value
    if not value_map:
        return value
    return value_map.get(str(value), value)


def _disease_output_column(cfg: dict) -> str:
    return cfg.get("disease_output_column") or "disease_status"


def _obs_with_configured_disease_column(
    obs: pd.DataFrame,
    scrna_config: DatascRNAConfig,
) -> pd.DataFrame:
    cfg = asdict(scrna_config)
    disease_column = cfg.get("disease_column")
    if not disease_column:
        return obs

    output_column = _disease_output_column(cfg)
    if disease_column not in obs.columns:
        raise ValueError(f"Missing disease_column in AnnData obs: {disease_column}")

    mapped_obs = obs.copy()
    mapped_obs[output_column] = mapped_obs[disease_column].map(
        lambda value: _map_config_value(value, cfg.get("disease_value_map"))
    )
    return mapped_obs


def _report_extra_columns(scrna_config: DatascRNAConfig) -> list[str]:
    columns = []
    if scrna_config.disease_column:
        columns.append(scrna_config.disease_output_column or "disease_status")
    if scrna_config.time_column:
        columns.append(scrna_config.time_column)
    if scrna_config.group_column:
        columns.append(scrna_config.group_column)
    columns.extend(scrna_config.report_columns or [])
    columns = list(dict.fromkeys(columns))
    return columns


def _sentence_mode_metadata(
    train_ds: Dataset,
    scrna_config: DatascRNAConfig,
) -> dict[str, int | float | str | list[str]] | None:
    if not scrna_config.nonsemantic_train_column or not scrna_config.nonsemantic_train_values:
        return None
    if "sentence_mode" not in train_ds.features:
        return None

    train_df = train_ds.to_pandas()
    n_train_rows = len(train_df)
    n_nonsemantic = int((train_df["sentence_mode"] == "nonsemantic").sum())
    return {
        "nonsemantic_train_column": scrna_config.nonsemantic_train_column,
        scrna_config.nonsemantic_train_key: list(map(str, scrna_config.nonsemantic_train_values)),
        "n_train_rows": n_train_rows,
        "n_train_nonsemantic_rows": n_nonsemantic,
        "fraction_train_nonsemantic_rows": float(n_nonsemantic / n_train_rows) if n_train_rows else 0.0,
    }


def _write_sentence_mode_report(
    train_ds: Dataset,
    test_ds: Dataset,
    *,
    output_dir: str | Path,
    column: str,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frames = []
    for split_name, dataset in [("train", train_ds), ("test", test_ds)]:
        frame = dataset.to_pandas()
        if "sentence_mode" not in frame.columns:
            continue
        if column not in frame.columns:
            raise ValueError(f"Missing nonsemantic_train_column in dataset rows: {column}")
        counts = (
            frame.groupby(["sentence_mode", column], dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={column: "value"})
        )
        counts["split"] = split_name
        split_total = counts["count"].sum()
        counts["proportion"] = counts["count"] / split_total if split_total else 0.0
        frames.append(counts.loc[:, ["split", "sentence_mode", "value", "count", "proportion"]])

    report = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["split", "sentence_mode", "value", "count", "proportion"]
    )
    report_path = output_path / "sentence_mode_report.csv"
    report.to_csv(report_path, index=False)
    return {"sentence_mode_report": str(report_path)}


def _prepare_adata_for_scrna(adata, scrna_config: DatascRNAConfig, **kwargs):
    cfg = asdict(scrna_config)
    cfg.update(kwargs)

    annotation_column = cfg['annotation_column']

    if cfg['preprocessing']:
        adata = run_preprocessing(adata, annotation_column, scrna_config, **kwargs)

    if cfg['highly_variable_genes']:
        if 'highly_variable' not in adata.var:
            raise ValueError("Missing 'highly_variable' column in adata.var.")
        return adata[:, adata.var['highly_variable']].copy()

    return adata.copy()


def _build_scrna_rows(adata_subset, indices, scrna_config: DatascRNAConfig, **kwargs) -> pd.DataFrame:
    cfg = asdict(scrna_config)
    cfg.update(kwargs)

    annotation_column = cfg['annotation_column']
    selected = adata_subset[list(indices)].copy()
    X = selected.X.toarray() if sparse.issparse(selected.X) else selected.X
    genes_upper = selected.var_names.str.upper()
    genes = selected.var.index

    if cfg['housekeeping_genes']:
        mask = ~(
            genes_upper.str.startswith("MT-") |
            genes_upper.str.startswith("RPS") |
            genes_upper.str.startswith("RPL")
        )
        genes = genes[mask]
        X = X[:, mask]

    all_rows = []
    for i in tqdm(range(X.shape[0]), desc="Building dataset"):
        expr_values = X[i, :]
        ranked_genes = genes[np.argsort(-expr_values, kind="stable")[:cfg['cs_length'][0]]].tolist()

        cell_type = selected.obs[annotation_column].iloc[i]
        row = {
            "index": selected.obs.index[i],
            "gene_list": ranked_genes,
            annotation_column: cell_type,
            "label": cell_type,
        }

        if cfg['disease_column']:
            disease_status = _map_config_value(
                selected.obs[cfg['disease_column']].iloc[i],
                cfg.get("disease_value_map"),
            )
            disease_output_column = _disease_output_column(cfg)
            row[disease_output_column] = disease_status
            row["label"] = f"{cell_type}_{disease_status}" if pd.notnull(disease_status) else cell_type

        if cfg['time_column']:
            time = selected.obs[cfg['time_column']].iloc[i]
            row["time"] = time
            row["label"] = f"{cell_type}_{time}" if pd.notnull(time) else cell_type

        all_rows.append(row)

    return pd.DataFrame(all_rows)


def make_scrna_split_indices(
    adata,
    scrna_config: DatascRNAConfig,
    **kwargs,
) -> SplitIndices | None:
    cfg = asdict(scrna_config)
    cfg.update(kwargs)

    split_strategy = cfg.get("split_strategy")
    if split_strategy is None:
        return None

    if split_strategy == "random_stratified":
        return make_random_stratified_split(
            adata.obs,
            dataset_id=cfg.get("dataset_id") or "scrna",
            annotation_column=cfg["annotation_column"],
            test_size=cfg["test_size"],
            total_cells=cfg.get("total_cells"),
            random_state=cfg["random_state"],
        )

    if split_strategy == "heldout_donor":
        if not cfg.get("donor_column"):
            raise ValueError("donor_column is required when split_strategy='heldout_donor'.")
        return make_proportional_heldout_donor_split(
            adata.obs,
            dataset_id=cfg.get("dataset_id") or "scrna",
            donor_column=cfg["donor_column"],
            annotation_column=cfg["annotation_column"],
            test_size=cfg["test_size"],
            total_cells=cfg.get("total_cells"),
            donor_test_size=cfg.get("donor_test_size"),
            random_state=cfg["random_state"],
            stratified_subsample=cfg.get("stratified_subsample", True),
        )

    if split_strategy == "heldout_donor_and_value":
        if not cfg.get("donor_column"):
            raise ValueError("donor_column is required when split_strategy='heldout_donor_and_value'.")
        if not cfg.get("heldout_column"):
            raise ValueError("heldout_column is required when split_strategy='heldout_donor_and_value'.")
        if not cfg.get("heldout_values"):
            raise ValueError("heldout_values is required when split_strategy='heldout_donor_and_value'.")
        return make_proportional_heldout_donor_and_value_split(
            adata.obs,
            dataset_id=cfg.get("dataset_id") or "scrna",
            donor_column=cfg["donor_column"],
            annotation_column=cfg["annotation_column"],
            heldout_column=cfg["heldout_column"],
            heldout_values=cfg["heldout_values"],
            heldout_key=cfg.get("heldout_key") or "values",
            test_size=cfg["test_size"],
            total_cells=cfg.get("total_cells"),
            donor_test_size=cfg.get("donor_test_size"),
            random_state=cfg["random_state"],
            stratified_subsample=cfg.get("stratified_subsample", True),
        )

    if split_strategy == "heldout_group":
        if not cfg.get("group_column"):
            raise ValueError("group_column is required when split_strategy='heldout_group'.")
        return make_proportional_heldout_group_split(
            adata.obs,
            dataset_id=cfg.get("dataset_id") or "scrna",
            group_column=cfg["group_column"],
            group_key=cfg.get("group_key") or "groups",
            annotation_column=cfg["annotation_column"],
            test_size=cfg["test_size"],
            total_cells=cfg.get("total_cells"),
            group_test_size=cfg.get("group_test_size"),
            random_state=cfg["random_state"],
            stratified_subsample=cfg.get("stratified_subsample", True),
        )

    if split_strategy == "heldout_value":
        if not cfg.get("heldout_column"):
            raise ValueError("heldout_column is required when split_strategy='heldout_value'.")
        if not cfg.get("heldout_values"):
            raise ValueError("heldout_values is required when split_strategy='heldout_value'.")
        return make_heldout_value_split(
            adata.obs,
            dataset_id=cfg.get("dataset_id") or "scrna",
            annotation_column=cfg["annotation_column"],
            heldout_column=cfg["heldout_column"],
            heldout_values=cfg["heldout_values"],
            heldout_key=cfg.get("heldout_key") or "values",
            test_size=cfg["test_size"],
            total_cells=cfg.get("total_cells"),
            random_state=cfg["random_state"],
            stratified_subsample=cfg.get("stratified_subsample", True),
        )

    raise ValueError(f"Unsupported scRNA split_strategy: {split_strategy}")


def _metadata_safe_config(scrna_config: DatascRNAConfig) -> dict:
    payload = asdict(scrna_config)
    if payload["output_dir"] is not None:
        payload["output_dir"] = str(payload["output_dir"])
    return payload


def save_scrna_dataset_artifacts(
    *,
    train_ds: Dataset,
    test_ds: Dataset,
    adata_test,
    split_indices: SplitIndices,
    adata_obs: pd.DataFrame,
    scrna_config: DatascRNAConfig,
) -> dict[str, dict[str, str]]:
    if scrna_config.output_dir is None:
        raise ValueError("scrna_config.output_dir is required when save_artifacts=True.")

    output_dir = Path(scrna_config.output_dir)
    datasets_dir = output_dir / "datasets"
    reports_dir = output_dir / "reports"
    metadata_dir = output_dir / "metadata"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    dataset_artifacts = {
        "scrna_data": str(datasets_dir / "scrna_data"),
        "scrna_test": str(datasets_dir / "scrna_test"),
        "adata_test": str(output_dir / "adata_test.h5ad"),
        "split_indices": str(metadata_dir / "split_indices.json"),
    }
    train_head_path = reports_dir / "scrna_train_head.csv"
    train_ds.save_to_disk(dataset_artifacts["scrna_data"])
    test_ds.save_to_disk(dataset_artifacts["scrna_test"])
    adata_test.write(dataset_artifacts["adata_test"])
    write_split_indices(split_indices, dataset_artifacts["split_indices"])
    train_head_count = min(5, len(train_ds))
    train_head = train_ds.select(range(train_head_count)).to_pandas()
    train_head.to_csv(train_head_path, index=False)

    report_obs = _obs_with_configured_disease_column(adata_obs, scrna_config)
    report_artifacts = write_split_report(
        report_obs,
        split_indices,
        output_dir=reports_dir,
        annotation_column=scrna_config.annotation_column,
        donor_column=(
            scrna_config.donor_column
            if split_indices.strategy.startswith("heldout_donor")
            else None
        ),
        extra_columns=_report_extra_columns(scrna_config),
    )
    label_shuffle_metadata = _label_shuffle_metadata(train_ds, scrna_config)
    if label_shuffle_metadata is not None:
        report_artifacts.update(
            _write_label_shuffle_report(
                train_ds,
                output_dir=reports_dir,
                original_column=scrna_config.label_shuffle_original_column,
            )
        )
    sentence_mode_metadata = _sentence_mode_metadata(train_ds, scrna_config)
    if sentence_mode_metadata is not None:
        report_artifacts.update(
            _write_sentence_mode_report(
                train_ds,
                test_ds,
                output_dir=reports_dir,
                column=scrna_config.nonsemantic_train_column,
            )
        )
    extra_metadata = {}
    if label_shuffle_metadata is not None:
        extra_metadata["label_shuffle"] = label_shuffle_metadata
    if sentence_mode_metadata is not None:
        extra_metadata["sentence_mode"] = sentence_mode_metadata
    report_artifacts["scrna_train_head"] = str(train_head_path)
    write_generation_metadata(
        metadata_dir / "generation_metadata.json",
        split=split_indices,
        source=scrna_config.source or "",
        scrna_config=_metadata_safe_config(scrna_config),
        dataset_artifacts=dataset_artifacts,
        report_artifacts=report_artifacts,
        extra_metadata=extra_metadata or None,
    )
    return {
        "dataset_artifacts": dataset_artifacts,
        "report_artifacts": report_artifacts,
    }


def gen_scrna_dataset_from_indices(
    adata,
    split_indices: SplitIndices,
    scrna_config: DatascRNAConfig,
    *,
    prepared_adata=None,
    train_semantic: bool | None = None,
    test_semantic: bool = False,
    **kwargs,
) -> Tuple[Dataset, Dataset]:
    """Generate train/test HuggingFace Datasets from explicit AnnData indices."""
    adata_subset = (
        prepared_adata.copy()
        if prepared_adata is not None
        else _prepare_adata_for_scrna(adata, scrna_config, **kwargs)
    )
    validate_split_indices(split_indices, adata_subset.obs)

    annotation_column = asdict(scrna_config)["annotation_column"]
    train_df = _build_scrna_rows(adata_subset, split_indices.train_indices, scrna_config, **kwargs)
    test_df = _build_scrna_rows(adata_subset, split_indices.test_indices, scrna_config, **kwargs)

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)
    train_ds = process_split(
        train_ds,
        annotation_column,
        scrna_config,
        semantic=scrna_config.semantic if train_semantic is None else train_semantic,
        **kwargs,
    )
    train_ds = _shuffle_train_labels(train_ds, scrna_config, **kwargs)
    test_ds = process_split(
        test_ds,
        annotation_column,
        scrna_config,
        semantic=test_semantic,
        **kwargs,
    )

    adata_test = adata_subset[split_indices.test_indices].copy()

    return train_ds, test_ds, adata_test


def gen_scrna_dataset(adata, scrna_config: DatascRNAConfig, **kwargs) -> Tuple[Dataset, Dataset]:
    """Generate train/test HuggingFace Datasets from AnnData object, overridable with kwargs."""
    cfg = asdict(scrna_config)
    cfg.update(kwargs)  # override any config with kwargs

    annotation_column = cfg['annotation_column']
    adata_subset = _prepare_adata_for_scrna(adata, scrna_config, **kwargs)
    df = _build_scrna_rows(
        adata_subset,
        adata_subset.obs.index.astype(str).tolist(),
        scrna_config,
        **kwargs,
    ).set_index("index")
    ds = Dataset.from_pandas(df.reset_index())

    # Split into train/test
    ds_split = ds.train_test_split(test_size=cfg['test_size'], seed=cfg['random_state'])
    train_ds = process_split(ds_split["train"], annotation_column, scrna_config, semantic=cfg['semantic'], **kwargs)
    train_ds = _shuffle_train_labels(train_ds, scrna_config, **kwargs)
    test_ds = process_split(ds_split["test"], annotation_column, scrna_config, semantic=False, **kwargs)
    
    test_indices = test_ds["index"]
    adata_test = adata_subset[test_indices]
    
    return train_ds, test_ds, adata_test
