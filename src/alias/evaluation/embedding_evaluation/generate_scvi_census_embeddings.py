from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_external_helpers():
    helper_path = PROJECT_ROOT / "src" / "alias" / "evaluation" / "embedding_evaluation" / "external.py"
    spec = importlib.util.spec_from_file_location("_alias_embedding_external", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load external embedding helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.require_optional_dependency, module.save_external_embedding_artifacts


require_optional_dependency, save_external_embedding_artifacts = _load_external_helpers()


DATASET_ID = "S2_heldout_donor_semantic_200k"
DATASET_DIR = (
    PROJECT_ROOT
    / "out"
    / "data"
    / "revision1_v1"
    / "HIAI_Tcells"
    / DATASET_ID
)
DEFAULT_ADATA_PATH = DATASET_DIR / "adata_test.h5ad"
DEFAULT_OUTPUT_DIR = DATASET_DIR / "embeddings_evaluation" / "external_embeddings"
DEFAULT_PRETRAINED_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "scVI_census_pretrained.parquet"
DEFAULT_PRETRAINED_METADATA_PATH = DEFAULT_OUTPUT_DIR / "scVI_census_pretrained_metadata.json"
DEFAULT_DATASET_ADAPTED_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "scVI_census_dataset_adapted.parquet"
DEFAULT_DATASET_ADAPTED_METADATA_PATH = DEFAULT_OUTPUT_DIR / "scVI_census_dataset_adapted_metadata.json"
DEFAULT_MODEL_CACHE_DIR = DEFAULT_OUTPUT_DIR / "scVI_census_modelhub"
DEFAULT_SCVI_S3_BUCKET = "cellxgene-contrib-public"
DEFAULT_CENSUS_VERSION = "2025-11-08"
DEFAULT_ORGANISM = "homo_sapiens"
SCVI_INSTALL_COMMAND = (
    'uv pip install --python .venv_annotation/bin/python "scvi-tools>=1.3.3" '
    '"boto3>=1.34" "botocore>=1.34" "pyarrow>=15" '
    '"cellxgene-census>=1.15" "mygene>=3.2.2"'
)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def print_dependency_guidance() -> None:
    print("Dependency guidance for scVI Census embeddings:", flush=True)
    print(f"  {SCVI_INSTALL_COMMAND}", flush=True)


def _default_s3_path(*, census_version: str, organism: str) -> str:
    return f"models/scvi/{census_version}/{organism}/model.pt"


def _default_output_path(mode: str) -> Path:
    if mode == "dataset_adapted":
        return DEFAULT_DATASET_ADAPTED_OUTPUT_PATH
    return DEFAULT_PRETRAINED_OUTPUT_PATH


def _default_metadata_path(mode: str) -> Path:
    if mode == "dataset_adapted":
        return DEFAULT_DATASET_ADAPTED_METADATA_PATH
    return DEFAULT_PRETRAINED_METADATA_PATH


def _apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.query_train_epochs is None:
        args.query_train_epochs = 20 if args.mode == "dataset_adapted" else 0
    if args.source_batch_key is None and args.mode == "dataset_adapted":
        args.source_batch_key = "batch_id"
    if args.s3_path is None:
        args.s3_path = _default_s3_path(
            census_version=args.census_version,
            organism=args.organism,
        )
    if args.output_path is None:
        args.output_path = _default_output_path(args.mode)
    if args.metadata_path is None:
        args.metadata_path = _default_metadata_path(args.mode)
    return args


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scVI Census embeddings for the HIAI T-cell S2 test AnnData.",
    )
    parser.add_argument(
        "--mode",
        choices=("pretrained", "dataset_adapted"),
        default=os.environ.get("HIAI_TCELLS_SCVI_CENSUS_MODE", "pretrained"),
        help=(
            "pretrained uses deterministic encoder-only inference. "
            "dataset_adapted performs scVI query adaptation before embedding."
        ),
    )
    parser.add_argument(
        "--census-version",
        default=os.environ.get("HIAI_TCELLS_SCVI_CENSUS_VERSION", DEFAULT_CENSUS_VERSION),
    )
    parser.add_argument(
        "--organism",
        default=os.environ.get("HIAI_TCELLS_SCVI_CENSUS_ORGANISM", DEFAULT_ORGANISM),
        help="CELLxGENE Census organism, for example homo_sapiens or mus_musculus.",
    )
    parser.add_argument("--adata-path", type=Path, default=DEFAULT_ADATA_PATH)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_env_path("HIAI_TCELLS_SCVI_CENSUS_MODEL_DIR"),
        help=(
            "Local scVI Census model directory. If omitted, the CELLxGENE Census "
            "model is pulled from S3. Can also be set with "
            "HIAI_TCELLS_SCVI_CENSUS_MODEL_DIR."
        ),
    )
    parser.add_argument(
        "--no-download-model",
        action="store_false",
        dest="download_model",
        help="Fail instead of pulling the default CELLxGENE Census model from S3.",
    )
    parser.set_defaults(download_model=True)
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=_env_path("HIAI_TCELLS_SCVI_CENSUS_MODEL_CACHE_DIR")
        or DEFAULT_MODEL_CACHE_DIR,
        help="Persistent cache directory for the downloaded scVI Census model.",
    )
    parser.add_argument("--s3-bucket", default=DEFAULT_SCVI_S3_BUCKET)
    parser.add_argument("--s3-path", default=None)
    parser.add_argument(
        "--s3-download-mode",
        choices=("auto", "direct", "hub"),
        default="auto",
        help=(
            "How to interpret --s3-path. 'direct' downloads a model.pt file into "
            "the cache directory; 'hub' uses scvi.hub.HubModel.pull_from_s3; "
            "'auto' uses hub only for paths ending in modelhub."
        ),
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument(
        "--layer",
        default="counts",
        help="Expression layer to use. Use 'X' to keep adata.X.",
    )
    parser.add_argument(
        "--batch-key",
        default="batch",
        help="Batch key required by many scVI reference registries.",
    )
    parser.add_argument(
        "--source-batch-key",
        default=None,
        help="Source obs column copied into --batch-key when --batch-key is absent.",
    )
    parser.add_argument(
        "--constant-batch-value",
        default=os.environ.get("HIAI_TCELLS_SCVI_CONSTANT_BATCH_VALUE", "unassigned"),
        help=(
            "Batch value used when no source batch key is configured. "
            "This is the pretrained-mode default."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--query-train-epochs", type=int, default=None)
    parser.add_argument(
        "--no-early-stopping",
        action="store_false",
        dest="early_stopping",
        help="Disable early stopping during scVI query adaptation.",
    )
    parser.set_defaults(early_stopping=True)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--gene-id-column",
        default=None,
        help=(
            "Optional adata.var column containing reference gene ids. Use this "
            "when adata.var_names are gene symbols but the reference uses "
            "Ensembl ids."
        ),
    )
    parser.add_argument(
        "--gene-symbol-to-ensembl-path",
        type=Path,
        default=_env_path("HIAI_TCELLS_SCVI_GENE_SYMBOL_TO_ENSEMBL_PATH"),
        help=(
            "Optional CSV/TSV mapping file used when query var_names are gene "
            "symbols. Can also be set with "
            "HIAI_TCELLS_SCVI_GENE_SYMBOL_TO_ENSEMBL_PATH."
        ),
    )
    parser.add_argument("--mapping-symbol-column", default="symbol")
    parser.add_argument("--mapping-ensembl-column", default="ensembl_id")
    parser.add_argument(
        "--no-auto-mygene-mapping",
        action="store_false",
        dest="auto_mygene_mapping",
        help=(
            "Disable automatic gene-symbol to Ensembl mapping through mygene "
            "when no direct overlap with the reference genes is found."
        ),
    )
    parser.set_defaults(auto_mygene_mapping=True)
    parser.add_argument(
        "--no-census-var-mapping",
        action="store_false",
        dest="census_var_mapping",
        help="Disable default Census var-table symbol to Ensembl mapping.",
    )
    parser.set_defaults(census_var_mapping=True)
    parser.add_argument(
        "--no-round-counts",
        action="store_false",
        dest="round_counts",
        help="Disable rounding selected count-like expression values before scVI.",
    )
    parser.set_defaults(round_counts=True)
    parser.add_argument("--force", action="store_true")
    return _apply_mode_defaults(parser.parse_args(argv))


def _resolve_model_reference(
    *,
    args: argparse.Namespace,
    HubModel: Any,
    botocore: Any,
    boto3_client_factory: Any,
) -> tuple[str, dict[str, Any]]:
    if args.model_dir is not None:
        if not args.model_dir.exists():
            raise FileNotFoundError(
                f"scVI Census model directory not found: {args.model_dir}"
            )
        return str(args.model_dir), {
            "model_source": "local",
            "model_dir": str(args.model_dir),
        }

    if not args.download_model:
        raise ValueError(
            "Provide --model-dir or allow S3 download by omitting --no-download-model."
        )

    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    download_mode = args.s3_download_mode
    if download_mode == "auto":
        download_mode = "hub" if str(args.s3_path).rstrip("/").endswith("modelhub") else "direct"

    if download_mode == "direct":
        model_s3_path = str(args.s3_path)
        if not model_s3_path.endswith("model.pt"):
            model_s3_path = str(Path(model_s3_path) / "model.pt")
        model_local_path = args.model_cache_dir / "model.pt"
        if not model_local_path.exists():
            print(
                "Pulling scVI Census model.pt from S3 into "
                f"{model_local_path} ({args.s3_bucket}/{model_s3_path})"
            )
            s3 = boto3_client_factory(
                "s3",
                config=botocore.config.Config(signature_version=botocore.UNSIGNED),
            )
            try:
                s3.download_file(args.s3_bucket, model_s3_path, str(model_local_path))
            except Exception as error:
                raise FileNotFoundError(
                    "Could not download scVI Census model.pt from "
                    f"s3://{args.s3_bucket}/{model_s3_path}. "
                    "If you are using an scvi-tools HubModel folder instead, pass "
                    "--s3-download-mode hub and set --s3-path to the modelhub folder."
                ) from error
        else:
            print(f"Reusing cached scVI Census model.pt: {model_local_path}")
        return str(args.model_cache_dir), {
            "model_source": "s3_direct_model_pt",
            "model_dir": str(args.model_cache_dir),
            "model_cache_dir": str(args.model_cache_dir),
            "s3_bucket": args.s3_bucket,
            "s3_path": model_s3_path,
            "s3_download_mode": download_mode,
        }

    if HubModel is None:
        raise ValueError("HubModel is required when --s3-download-mode=hub.")
    print(
        "Pulling scVI Census model from S3 into "
        f"{args.model_cache_dir} ({args.s3_bucket}/{args.s3_path})"
    )
    hub_model = HubModel.pull_from_s3(
        s3_bucket=args.s3_bucket,
        s3_path=args.s3_path,
        pull_anndata=False,
        cache_dir=str(args.model_cache_dir),
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    )
    local_dir = getattr(hub_model, "local_dir", None)
    if local_dir is None:
        raise ValueError("Downloaded scVI HubModel did not expose a local_dir.")
    return str(local_dir), {
        "model_source": "s3_download",
        "model_dir": str(local_dir),
        "model_cache_dir": str(args.model_cache_dir),
        "s3_bucket": args.s3_bucket,
        "s3_path": args.s3_path,
        "s3_download_mode": download_mode,
    }


def _round_expression_counts(X: Any) -> Any:
    """Round count-like expression values while preserving dense/sparse layout."""
    if sp.issparse(X):
        rounded = X.copy()
        rounded.data = np.rint(rounded.data).astype(rounded.data.dtype)
        return rounded
    return np.rint(X).astype(getattr(X, "dtype", np.float32), copy=False)


def _resolve_device(requested: str) -> str:
    req = (requested or "auto").lower()
    try:
        import torch
    except Exception:
        return "cpu"
    if req in {"auto", "gpu", "cuda"} and torch.cuda.is_available():
        return "cuda"
    if req == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_adata(adata_path: Path, layer: str, *, round_counts: bool = True):
    ad = require_optional_dependency(
        "anndata",
        install_command=SCVI_INSTALL_COMMAND,
        package_label="anndata",
    )
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    adata = ad.read_h5ad(adata_path)
    if layer != "X":
        if layer not in adata.layers:
            raise KeyError(
                f"Layer {layer!r} not found in {adata_path}. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        adata.X = adata.layers[layer].copy()
    if round_counts:
        adata.X = _round_expression_counts(adata.X)
    return adata


def _ensure_batch_column(
    adata: Any,
    *,
    batch_key: str,
    source_batch_key: str | None,
    constant_batch_value: str = "unassigned",
) -> str:
    if batch_key in adata.obs:
        adata.obs[batch_key] = adata.obs[batch_key].astype(str)
        return batch_key
    if source_batch_key and source_batch_key in adata.obs:
        adata.obs[batch_key] = adata.obs[source_batch_key].astype(str)
        return source_batch_key
    adata.obs[batch_key] = str(constant_batch_value)
    return "constant_unknown"


def _extract_ensembl_gene_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        gene_id = value.get("gene")
        return [str(gene_id)] if gene_id else []
    if isinstance(value, list):
        gene_ids: list[str] = []
        for item in value:
            gene_ids.extend(_extract_ensembl_gene_ids(item))
        return gene_ids
    return []


def _strip_ensembl_version(gene_id: Any) -> str:
    return str(gene_id).split(".")[0]


def _read_gene_symbol_to_ensembl_mapping(
    path: Path,
    *,
    symbol_column: str,
    ensembl_column: str,
) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Gene symbol to Ensembl mapping file not found: {path}")
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    mapping_df = pd.read_csv(path, sep=separator)
    missing = {symbol_column, ensembl_column}.difference(mapping_df.columns)
    if missing:
        raise ValueError(f"{path} is missing mapping columns: {sorted(missing)}")
    mapping_df = mapping_df[[symbol_column, ensembl_column]].dropna().copy()
    mapping_df[symbol_column] = mapping_df[symbol_column].astype(str)
    mapping_df[ensembl_column] = mapping_df[ensembl_column].map(_strip_ensembl_version)
    mapping_df = mapping_df.drop_duplicates(subset=symbol_column, keep="first")
    return dict(zip(mapping_df[symbol_column], mapping_df[ensembl_column]))


def _query_census_symbol_to_ensembl_mapping(
    *,
    census_version: str,
    organism: str,
) -> dict[str, str]:
    cellxgene_census = require_optional_dependency(
        "cellxgene_census",
        install_command=SCVI_INSTALL_COMMAND,
        package_label="cellxgene-census",
    )
    with cellxgene_census.open_soma(census_version=census_version) as census:
        var_df = (
            census["census_data"][organism]
            .ms["RNA"]
            .var
            .read(column_names=["feature_id", "feature_name"])
            .concat()
            .to_pandas()
        )
    var_df = (
        var_df[["feature_name", "feature_id"]]
        .dropna()
        .drop_duplicates(subset="feature_name", keep="first")
    )
    var_df["feature_id"] = var_df["feature_id"].map(_strip_ensembl_version)
    return dict(zip(var_df["feature_name"].astype(str), var_df["feature_id"]))


def _query_mygene_symbol_to_ensembl_mapping(
    symbols: list[str],
    *,
    reference_var_names: pd.Index,
    organism: str,
) -> dict[str, str]:
    mygene = require_optional_dependency(
        "mygene",
        install_command=SCVI_INSTALL_COMMAND,
        package_label="mygene",
    )
    reference_ids = set(reference_var_names.astype(str))
    client = mygene.MyGeneInfo()
    results = client.querymany(
        symbols,
        scopes="symbol,alias",
        fields="ensembl.gene,symbol",
        species="mouse" if organism == "mus_musculus" else "human",
        as_dataframe=False,
        verbose=False,
    )
    mapping: dict[str, str] = {}
    for result in results:
        query_symbol = str(result.get("query", ""))
        for gene_id in _extract_ensembl_gene_ids(result.get("ensembl")):
            gene_id = _strip_ensembl_version(gene_id)
            if gene_id in reference_ids:
                mapping[query_symbol] = gene_id
                break
    return mapping


def _subset_to_reference_gene_ids(
    adata: Any,
    mapping: dict[str, str],
    *,
    reference_var_names: pd.Index,
    mapping_source: str,
) -> tuple[Any, dict[str, Any]]:
    original_var_names = pd.Index(adata.var_names.astype(str))
    reference_ids = set(reference_var_names.astype(str))
    used_reference_ids: set[str] = set()
    keep_positions: list[int] = []
    ensembl_ids: list[str] = []
    duplicate_reference_mappings = 0
    mapped_reference_genes = 0

    for position, gene_name in enumerate(original_var_names):
        candidate = mapping.get(str(gene_name))
        if candidate is not None:
            candidate = _strip_ensembl_version(candidate)
        if candidate in reference_ids and candidate not in used_reference_ids:
            keep_positions.append(position)
            ensembl_ids.append(candidate)
            used_reference_ids.add(candidate)
            mapped_reference_genes += 1
        elif candidate in reference_ids:
            duplicate_reference_mappings += 1

    adata_query = adata[:, keep_positions].copy()
    adata_query.var["gene_symbol"] = original_var_names[keep_positions].astype(str)
    adata_query.var["original_var_name"] = original_var_names[keep_positions].astype(str)
    adata_query.var["ensembl_id"] = ensembl_ids
    adata_query.var_names = pd.Index(ensembl_ids)

    metadata = {
        "gene_mapping_source": mapping_source,
        "n_query_genes": int(len(original_var_names)),
        "n_input_genes": int(len(original_var_names)),
        "n_mapping_entries": int(len(mapping)),
        "n_mapped_reference_genes": int(mapped_reference_genes),
        "n_mapped_genes": int(mapped_reference_genes),
        "n_dropped_genes": int(len(original_var_names) - mapped_reference_genes),
        "n_duplicate_reference_mappings_skipped": int(duplicate_reference_mappings),
        "n_query_genes_after_reference_subset": int(adata_query.n_vars),
        "reference_overlap_after_mapping": int(
            len(adata_query.var_names.intersection(reference_var_names))
        ),
    }
    return adata_query, metadata


def _prepare_query_var_names(
    *,
    scvi: Any,
    adata: Any,
    model_reference: str,
    gene_id_column: str | None,
    mapping_path: Path | None,
    mapping_symbol_column: str,
    mapping_ensembl_column: str,
    census_var_mapping: bool,
    census_version: str,
    organism: str,
    auto_mygene_mapping: bool,
) -> tuple[Any, dict[str, Any]]:
    reference_var_names = pd.Index(
        scvi.model.SCVI.prepare_query_anndata(
            adata,
            model_reference,
            return_reference_var_names=True,
        ).astype(str)
    )
    overlap_before = int(len(adata.var_names.intersection(reference_var_names)))
    metadata: dict[str, Any] = {
        "reference_var_count": int(len(reference_var_names)),
        "reference_overlap_before_mapping": overlap_before,
        "gene_mapping_source": "none",
        "n_input_genes": int(adata.n_vars),
    }
    if overlap_before > 0:
        print(
            f"Found {overlap_before}/{len(reference_var_names)} reference genes "
            "before gene-id mapping."
        )
        metadata.update(
            {
                "n_mapped_genes": overlap_before,
                "n_dropped_genes": int(adata.n_vars - overlap_before),
                "n_query_genes_after_reference_subset": int(adata.n_vars),
            }
        )
        return adata, metadata

    mapping: dict[str, str] | None = None
    mapping_source = ""
    if gene_id_column:
        if gene_id_column not in adata.var:
            raise KeyError(f"adata.var is missing --gene-id-column {gene_id_column!r}")
        mapping = dict(
            zip(
                adata.var_names.astype(str),
                adata.var[gene_id_column].map(_strip_ensembl_version),
            )
        )
        mapping_source = f"adata.var[{gene_id_column}]"
    elif mapping_path is not None:
        mapping = _read_gene_symbol_to_ensembl_mapping(
            mapping_path,
            symbol_column=mapping_symbol_column,
            ensembl_column=mapping_ensembl_column,
        )
        mapping_source = str(mapping_path)
    elif census_var_mapping:
        print(
            "No overlap between query var_names and scVI Census reference genes. "
            "Mapping symbols with the CELLxGENE Census var table."
        )
        mapping = _query_census_symbol_to_ensembl_mapping(
            census_version=census_version,
            organism=organism,
        )
        mapping_source = f"cellxgene_census_var:{census_version}:{organism}"
    elif auto_mygene_mapping:
        print(
            "No overlap between query var_names and scVI Census reference genes. "
            "Query var_names look like gene symbols while the reference uses "
            "Ensembl ids, so mapping symbols with mygene."
        )
        mapping = _query_mygene_symbol_to_ensembl_mapping(
            adata.var_names.astype(str).tolist(),
            reference_var_names=reference_var_names,
            organism=organism,
        )
        mapping_source = "mygene"

    if mapping is None:
        raise ValueError(
            "No overlap between query var_names and scVI Census reference genes. "
            "The Census model uses Ensembl gene ids. Provide --gene-id-column, "
            "--gene-symbol-to-ensembl-path, or allow automatic mygene mapping."
        )

    adata_query, mapping_metadata = _subset_to_reference_gene_ids(
        adata,
        mapping,
        reference_var_names=reference_var_names,
        mapping_source=mapping_source,
    )
    metadata.update(mapping_metadata)
    if metadata["reference_overlap_after_mapping"] == 0:
        raise ValueError(
            "Gene-id mapping produced zero overlap with the scVI Census reference. "
            "Check that the mapping points to human Ensembl gene ids."
        )
    print(
        "Mapped query gene identifiers to scVI Census reference genes: "
        f"{metadata['reference_overlap_after_mapping']}/{metadata['reference_var_count']} overlap."
    )
    return adata_query, metadata


def _load_query_model(
    *,
    scvi: Any,
    adata: Any,
    model_reference: str,
    query_train_epochs: int,
    early_stopping: bool,
    early_stopping_patience: int,
    weight_decay: float,
    accelerator: str,
    devices: str,
) -> Any:
    scvi.model.SCVI.prepare_query_anndata(adata, model_reference, inplace=True)
    query_model = scvi.model.SCVI.load_query_data(adata, model_reference)

    if query_train_epochs > 0:
        train_kwargs: dict[str, Any] = {
            "max_epochs": query_train_epochs,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "accelerator": accelerator,
            "plan_kwargs": {"weight_decay": weight_decay},
        }
        if devices != "auto":
            train_kwargs["devices"] = devices
        query_model.train(**train_kwargs)
    else:
        query_model.is_trained = True
        device = _resolve_device(accelerator)
        if device != "cpu" and hasattr(query_model, "to_device"):
            try:
                query_model.to_device(device)
            except Exception as error:
                print(f"Could not move scVI query model to {device!r}: {error}. Using default device.")
    return query_model


def _history_to_frame(query_model: Any) -> pd.DataFrame:
    history = getattr(query_model, "history", None)
    if history is None:
        return pd.DataFrame()
    if isinstance(history, pd.DataFrame):
        return history.reset_index()
    if isinstance(history, dict):
        try:
            return pd.DataFrame(history).reset_index()
        except Exception:
            return pd.DataFrame()
    try:
        return pd.DataFrame(history).reset_index()
    except Exception:
        return pd.DataFrame()


def _write_query_training_history(
    query_model: Any,
    *,
    output_path: Path,
    enabled: bool,
) -> Path | None:
    if not enabled:
        return None
    history_df = _history_to_frame(query_model)
    if history_df.empty:
        return None
    history_path = output_path.with_name(f"{output_path.stem}_training_history.csv")
    history_df.to_csv(history_path, index=False)
    return history_path


def generate_scvi_census_embeddings(args: argparse.Namespace) -> tuple[Path, Path]:
    print_dependency_guidance()
    if args.output_path.exists() and not args.force:
        print(f"Reusing existing scVI Census embeddings: {args.output_path}")
        return args.output_path, args.metadata_path

    scvi = require_optional_dependency(
        "scvi",
        install_command=SCVI_INSTALL_COMMAND,
        package_label="scvi-tools",
    )
    HubModel = None
    botocore = None
    boto3_client_factory = None
    if args.model_dir is None:
        botocore = require_optional_dependency(
            "botocore",
            install_command=SCVI_INSTALL_COMMAND,
            package_label="botocore",
        )
        if args.s3_download_mode == "hub" or (
            args.s3_download_mode == "auto"
            and str(args.s3_path).rstrip("/").endswith("modelhub")
        ):
            hub = require_optional_dependency(
                "scvi.hub",
                install_command=SCVI_INSTALL_COMMAND,
                package_label="scvi-tools hub",
            )
            HubModel = hub.HubModel
        else:
            boto3 = require_optional_dependency(
                "boto3",
                install_command=SCVI_INSTALL_COMMAND,
                package_label="boto3",
            )
            boto3_client_factory = boto3.client
    model_reference, model_metadata = _resolve_model_reference(
        args=args,
        HubModel=HubModel,
        botocore=botocore,
        boto3_client_factory=boto3_client_factory,
    )
    adata = _load_adata(args.adata_path, args.layer, round_counts=args.round_counts)
    source_batch_key = _ensure_batch_column(
        adata,
        batch_key=args.batch_key,
        source_batch_key=args.source_batch_key,
        constant_batch_value=args.constant_batch_value,
    )
    adata_query, gene_mapping_metadata = _prepare_query_var_names(
        scvi=scvi,
        adata=adata,
        model_reference=model_reference,
        gene_id_column=args.gene_id_column,
        mapping_path=args.gene_symbol_to_ensembl_path,
        mapping_symbol_column=args.mapping_symbol_column,
        mapping_ensembl_column=args.mapping_ensembl_column,
        census_var_mapping=args.census_var_mapping,
        census_version=args.census_version,
        organism=args.organism,
        auto_mygene_mapping=args.auto_mygene_mapping,
    )
    query_model = _load_query_model(
        scvi=scvi,
        adata=adata_query,
        model_reference=model_reference,
        query_train_epochs=args.query_train_epochs,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        weight_decay=args.weight_decay,
        accelerator=args.accelerator,
        devices=args.devices,
    )
    embeddings = query_model.get_latent_representation(
        adata_query,
        batch_size=args.batch_size,
    )
    training_history_path = _write_query_training_history(
        query_model,
        output_path=args.output_path,
        enabled=args.query_train_epochs > 0,
    )
    if not adata.obs_names.equals(adata_query.obs_names):
        raise ValueError("Query gene preparation changed cell order unexpectedly.")

    metadata = {
        "model_name": "scVI_census",
        "mode": args.mode,
        "inference_mode": (
            "encoder_only_pretrained"
            if args.query_train_epochs == 0
            else "query_adapted"
        ),
        "census_version": args.census_version,
        "organism": args.organism,
        "adata_path": str(args.adata_path),
        "expression_layer": args.layer,
        "round_counts": args.round_counts,
        "batch_key": args.batch_key,
        "source_batch_key": source_batch_key,
        "configured_source_batch_key": args.source_batch_key,
        "constant_batch_value": args.constant_batch_value,
        "query_train_epochs": args.query_train_epochs,
        "query_training_history_path": (
            str(training_history_path) if training_history_path is not None else None
        ),
        "query_train_early_stopping": args.early_stopping,
        "query_train_early_stopping_patience": args.early_stopping_patience,
        "query_train_weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "accelerator": args.accelerator,
        "resolved_encoder_device": _resolve_device(args.accelerator),
        "devices": args.devices,
        "scvi_version": getattr(scvi, "__version__", "unknown"),
        "reference_workflow": (
            "optional Census/query gene symbol to Ensembl mapping and reference-gene subset -> "
            "SCVI.prepare_query_anndata -> SCVI.load_query_data -> "
            "optional query adaptation -> "
            "get_latent_representation"
        ),
        "reference_url": (
            "https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/hub/"
            "cellxgene_census_model.html"
        ),
        **model_metadata,
        **gene_mapping_metadata,
    }
    return save_external_embedding_artifacts(
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        cell_ids=adata.obs_names,
        embeddings=embeddings,
        metadata=metadata,
        force=args.force,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        output_path, metadata_path = generate_scvi_census_embeddings(args)
    except (FileNotFoundError, ImportError, KeyError, ValueError) as error:
        print(error)
        raise SystemExit(1) from error
    print(f"Saved scVI Census embeddings to {output_path}")
    print(f"Saved scVI Census metadata to {metadata_path}")


if __name__ == "__main__":
    main()
