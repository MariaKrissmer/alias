from __future__ import annotations

"""Generate a draft synonym map from ontology sources.

The curated manuscript input, including manual Treg synonyms, is versioned under
``reproducibility/inputs/hiai_tcells/synonym_mapping.csv``.
"""

import json
import os
from pathlib import Path
import sys

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "alias").exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alias.util.load_hf_model import load_hf_dataset  # noqa: E402


DATASET_ID = "S2_heldout_donor_semantic_200k"
ANNOTATION_COLUMN = "AIFI_L2"
REVISION_DATA_ROOT = Path(
    os.environ.get("ALIAS_REVISION_DATA_ROOT", PROJECT_ROOT / "out" / "data" / "revision1_v1")
)
DATASET_DIR = REVISION_DATA_ROOT / "HIAI_Tcells" / DATASET_ID
SCRNA_TEST_PATH = DATASET_DIR / "datasets" / "scrna_test"
SCRNA_HF_DATASET = f"mariakrissmer/scrna_HIAI_Tcells_{DATASET_ID}"
OUTPUT_DIR = Path(
    os.environ.get("HIAI_TCELLS_SYNONYM_OUTPUT_DIR", DATASET_DIR / "celltype_annotation_synonym")
)
SYNONYM_MAPPING_PATH = OUTPUT_DIR / "synonym_mapping.csv"
CL_DESCRIPTION_PATH = (
    REVISION_DATA_ROOT
    / "HIAI"
    / "celltype_descriptions"
    / "hiai_aifi_l2_celltype_descriptions.csv"
)
CL_RAW_PATH = (
    REVISION_DATA_ROOT
    / "HIAI_Tcells"
    / "CL_raw_collection"
    / "hiai_tcells_cl_terms.csv"
)
FORCE_OVERWRITE = os.environ.get(
    "HIAI_TCELLS_SYNONYM_MAPPING_FORCE", "0"
).strip().lower() in {"1", "true", "yes", "on"}


def _load_scrna_test():
    if SCRNA_TEST_PATH.exists():
        try:
            print(f"Loading local scRNA test dataset from {SCRNA_TEST_PATH}")
            return load_from_disk(str(SCRNA_TEST_PATH))
        except Exception as error:
            print(
                f"load_from_disk failed for local scrna_test at {SCRNA_TEST_PATH}: "
                f"{type(error).__name__}: {error}"
            )
            arrow_files = sorted(SCRNA_TEST_PATH.glob("*.arrow"))
            if arrow_files:
                try:
                    import pyarrow as pa
                    import pyarrow.ipc as ipc

                    print(f"Loading local scRNA test dataset from Arrow files in {SCRNA_TEST_PATH}")
                    datasets = []
                    for path in arrow_files:
                        with pa.memory_map(str(path), "r") as source:
                            table = ipc.open_stream(source).read_all()
                        datasets.append(Dataset.from_pandas(table.to_pandas(), preserve_index=False))
                    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
                except Exception as arrow_error:
                    raise RuntimeError(
                        f"Local scrna_test exists but could not be loaded from {SCRNA_TEST_PATH}. "
                        f"Current working directory: {Path.cwd()}. "
                        f"load_from_disk error: {type(error).__name__}: {error}. "
                        f"Arrow fallback error: {type(arrow_error).__name__}: {arrow_error}"
                    ) from arrow_error
            raise RuntimeError(
                f"Local scrna_test exists but could not be loaded from {SCRNA_TEST_PATH}. "
                f"Current working directory: {Path.cwd()}. "
                f"No Arrow files were found. "
                f"Original error: {type(error).__name__}: {error}"
            ) from error
    print(f"Local scrna_test not found at {SCRNA_TEST_PATH}")
    print(f"Falling back to Hugging Face dataset: {SCRNA_HF_DATASET}")

    dataset = load_hf_dataset(SCRNA_HF_DATASET)
    if isinstance(dataset, dict):
        for split_name in ("test", "scrna_test"):
            if split_name in dataset:
                print(f"Using Hugging Face split {split_name!r} from {SCRNA_HF_DATASET}")
                return dataset[split_name]
        raise KeyError(
            f"Could not find a test split in {SCRNA_HF_DATASET}. "
            f"Available splits: {sorted(dataset.keys())}"
        )
    return dataset


def _clean_synonym_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).replace("_", " ").split()).strip()


def _split_synonym_field(value: object) -> list[str]:
    text = _clean_synonym_text(value)
    if not text:
        return []
    return [synonym.strip() for synonym in text.split(";") if synonym.strip()]


def _is_case_only_variant(canonical_label: str, synonym_label: str) -> bool:
    return (
        canonical_label != synonym_label
        and canonical_label.casefold() == synonym_label.casefold()
    )


def _add_synonym_record(
    records: list[dict[str, object]],
    seen: set[tuple[str, str]],
    *,
    canonical_label: str,
    synonym_label: str,
    source: str,
) -> None:
    synonym_label = _clean_synonym_text(synonym_label)
    if not synonym_label:
        return
    if _is_case_only_variant(canonical_label, synonym_label):
        return

    key = (canonical_label, synonym_label.casefold())
    if key in seen:
        return
    seen.add(key)
    records.append(
        {
            "canonical_label": canonical_label,
            "synonym_label": synonym_label,
            "is_canonical_label": synonym_label == canonical_label,
            "synonym_source": source,
        }
    )


def _cl_synonyms_by_label() -> dict[str, list[tuple[str, str]]]:
    synonyms_by_label: dict[str, list[tuple[str, str]]] = {}

    for path in [CL_DESCRIPTION_PATH, CL_RAW_PATH]:
        if not path.exists():
            print(f"Skipping CL synonym source because it is missing: {path}")
            continue

        df = pd.read_csv(path)
        label_column = (
            "canonical_label"
            if "canonical_label" in df.columns
            else "label"
            if "label" in df.columns
            else None
        )
        if label_column is None:
            print(f"Skipping CL synonym source without label column: {path}")
            continue

        for _, row in df.iterrows():
            canonical_label = _clean_synonym_text(row.get(label_column))
            if not canonical_label:
                continue

            values: list[tuple[str, str]] = []
            values.extend(
                ("cell_ontology_label", synonym)
                for synonym in _split_synonym_field(row.get("cell_ontology_label"))
            )
            values.extend(
                ("cell_ontology_synonym", synonym)
                for synonym in _split_synonym_field(row.get("cell_ontology_synonyms"))
            )
            values.extend(
                ("cell_ontology_exact_synonym", synonym)
                for synonym in _split_synonym_field(row.get("cell_ontology_exact_synonyms"))
            )
            synonyms_by_label.setdefault(canonical_label, []).extend(values)

    return synonyms_by_label


def _synonym_records(canonical_labels: set[str]) -> pd.DataFrame:
    records = []
    seen: set[tuple[str, str]] = set()
    cl_synonyms = _cl_synonyms_by_label()

    for canonical_label in sorted(canonical_labels):
        _add_synonym_record(
            records,
            seen,
            canonical_label=canonical_label,
            synonym_label=canonical_label,
            source="canonical_label",
        )
        for source, synonym in cl_synonyms.get(canonical_label, []):
            _add_synonym_record(
                records,
                seen,
                canonical_label=canonical_label,
                synonym_label=synonym,
                source=f"CL:{source}",
            )

    return pd.DataFrame(records)


def _write_summary(synonym_map: pd.DataFrame, observed_labels: set[str]) -> None:
    source_counts = (
        synonym_map["synonym_source"]
        .value_counts()
        .rename_axis("synonym_source")
        .reset_index(name="n_rows")
    )
    source_counts.to_csv(OUTPUT_DIR / "synonym_mapping_source_counts.csv", index=False)

    noncanonical = synonym_map[~synonym_map["is_canonical_label"]]
    labels_without_synonyms = sorted(observed_labels - set(noncanonical["canonical_label"]))
    summary = {
        "dataset_id": DATASET_ID,
        "annotation_column": ANNOTATION_COLUMN,
        "synonym_mapping_path": str(SYNONYM_MAPPING_PATH),
        "n_rows": int(len(synonym_map)),
        "n_canonical_labels": int(synonym_map["canonical_label"].nunique()),
        "n_noncanonical_synonyms": int(len(noncanonical)),
        "labels_without_noncanonical_synonyms": labels_without_synonyms,
        "source_counts": source_counts.to_dict(orient="records"),
    }
    with (OUTPUT_DIR / "synonym_mapping_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SYNONYM_MAPPING_PATH.exists() and not FORCE_OVERWRITE:
        print(
            "Synonym mapping already exists and was left untouched:\n"
            f"{SYNONYM_MAPPING_PATH}\n"
            "Set HIAI_TCELLS_SYNONYM_MAPPING_FORCE=1 to regenerate it."
        )
        return

    scrna_test = _load_scrna_test()
    observed_labels = set(map(str, scrna_test[ANNOTATION_COLUMN]))
    synonym_map = _synonym_records(observed_labels)
    synonym_map.to_csv(SYNONYM_MAPPING_PATH, index=False)
    _write_summary(synonym_map, observed_labels)
    print(f"Wrote synonym mapping to {SYNONYM_MAPPING_PATH}")


if __name__ == "__main__":
    main()
