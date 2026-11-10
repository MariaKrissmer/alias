from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
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
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "scGPT.parquet"
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_DIR / "scGPT_metadata.json"
SCGPT_INSTALL_COMMAND = (
    "uv venv --python 3.11 .venv_scgpt\n"
    'uv pip install --python .venv_scgpt/bin/python "scgpt==0.2.4" '
    '"torch==2.3.*" "torchtext==0.18.*" "ipython>=8" "pyarrow>=15"'
)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def print_dependency_guidance() -> None:
    print("Dependency guidance for scGPT embeddings:", flush=True)
    print(f"  {SCGPT_INSTALL_COMMAND}", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scGPT zero-shot embeddings for the HIAI T-cell S2 test AnnData.",
    )
    parser.add_argument("--adata-path", type=Path, default=DEFAULT_ADATA_PATH)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_env_path("HIAI_TCELLS_SCGPT_MODEL_DIR"),
        help=(
            "Local scGPT checkpoint directory. Can also be set with "
            "HIAI_TCELLS_SCGPT_MODEL_DIR."
        ),
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--layer",
        default="counts",
        help="Expression layer to use. Use 'X' to keep adata.X.",
    )
    parser.add_argument("--gene-col", default="gene_name")
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Device for scGPT. 'auto' uses CUDA when available, otherwise CPU. "
            "MPS is avoided by default because scGPT's transformer path uses "
            "nested tensor ops not implemented on MPS."
        ),
    )
    parser.add_argument(
        "--use-fast-transformer",
        action="store_true",
        help="Pass use_fast_transformer=True to scGPT. Off by default for portability.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_adata(adata_path: Path, *, layer: str, gene_col: str):
    ad = require_optional_dependency(
        "anndata",
        install_command=SCGPT_INSTALL_COMMAND,
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
    _coerce_unsigned_expression_to_float32(adata)

    if gene_col not in adata.var:
        adata.var[gene_col] = adata.var_names.astype(str)
    else:
        gene_values = adata.var[gene_col].astype("string")
        adata.var[gene_col] = gene_values.fillna(adata.var_names.astype(str)).astype(str)
    return adata


def _coerce_unsigned_expression_to_float32(adata: Any) -> None:
    matrix = adata.X
    dtype = getattr(matrix, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.unsignedinteger):
        adata.X = matrix.astype(np.float32)


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ensure_sched_getaffinity(os_module: Any = os) -> bool:
    """Provide the Linux-only API expected by scGPT on macOS.

    scGPT uses ``len(os.sched_getaffinity(0))`` as ``num_workers`` for a
    DataLoader with a local Dataset class. Returning an empty set forces
    ``num_workers=0``, avoiding macOS spawn pickling failures.
    """
    if hasattr(os_module, "sched_getaffinity"):
        return False

    def _sched_getaffinity(_pid: int) -> set[int]:
        return set()

    os_module.sched_getaffinity = _sched_getaffinity
    return True


def _extract_scgpt_embeddings(result_adata: Any) -> tuple[np.ndarray, str]:
    if "X_scGPT" in result_adata.obsm:
        return np.asarray(result_adata.obsm["X_scGPT"], dtype=np.float32), "obsm:X_scGPT"
    if "X_scgpt" in result_adata.obsm:
        return np.asarray(result_adata.obsm["X_scgpt"], dtype=np.float32), "obsm:X_scgpt"

    matrix = result_adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32), "X"


def generate_scgpt_embeddings(args: argparse.Namespace) -> tuple[Path, Path]:
    print_dependency_guidance()
    if args.model_dir is None:
        raise ValueError("Provide --model-dir or set HIAI_TCELLS_SCGPT_MODEL_DIR.")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"scGPT model directory not found: {args.model_dir}")
    if args.output_path.exists() and not args.force:
        print(f"Reusing existing scGPT embeddings: {args.output_path}")
        return args.output_path, args.metadata_path

    scg = require_optional_dependency(
        "scgpt",
        install_command=SCGPT_INSTALL_COMMAND,
        package_label="scGPT",
    )
    torch = require_optional_dependency(
        "torch",
        install_command=SCGPT_INSTALL_COMMAND,
        package_label="torch",
    )
    added_sched_getaffinity_fallback = _ensure_sched_getaffinity()
    device = _resolve_device(torch, args.device)
    adata = _load_adata(args.adata_path, layer=args.layer, gene_col=args.gene_col)

    result_adata = scg.tasks.embed_data(
        adata,
        str(args.model_dir),
        gene_col=args.gene_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        use_fast_transformer=args.use_fast_transformer,
        return_new_adata=True,
    )
    embeddings, embedding_source = _extract_scgpt_embeddings(result_adata)

    metadata = {
        "model_name": "scGPT",
        "model_dir": str(args.model_dir),
        "adata_path": str(args.adata_path),
        "expression_layer": args.layer,
        "gene_col": args.gene_col,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": device,
        "use_fast_transformer": args.use_fast_transformer,
        "embedding_source": embedding_source,
        "scgpt_version": getattr(scg, "__version__", "unknown"),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "added_sched_getaffinity_fallback": added_sched_getaffinity_fallback,
        "reference_workflow": "scgpt.tasks.embed_data",
        "reference_url": (
            "https://github.com/bowang-lab/scGPT/blob/main/tutorials/zero-shot/"
            "Tutorial_ZeroShot_Integration.ipynb"
        ),
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
        output_path, metadata_path = generate_scgpt_embeddings(args)
    except (FileNotFoundError, ImportError, KeyError, ValueError) as error:
        print(error)
        raise SystemExit(1) from error
    print(f"Saved scGPT embeddings to {output_path}")
    print(f"Saved scGPT metadata to {metadata_path}")


if __name__ == "__main__":
    main()
