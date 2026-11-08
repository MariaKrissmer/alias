"""Build portable query, PMID, and split manifests from local artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_literature_manifests(source_root: Path, destination_root: Path) -> None:
    sources = {
        "N1_ncbi_literature": {
            "path": source_root
            / "out/data/revision1_v1/HIAI_Tcells/NCBI_raw_collection/"
            / "hiai_tcells_ncbi_articles_exact_title_abstract_blood_cells.csv",
            "cell_type_column": "AIFI_L2",
        },
        "N3_ncbi_literature_shuffled_labels": {
            "path": source_root
            / "out/data/revision1_v1/HIAI_Tcells/NCBI_raw_collection/"
            / "hiai_tcells_ncbi_articles_exact_title_abstract_blood_cells.csv",
            "cell_type_column": "AIFI_L2",
        },
        "N4_ncbi_literature_cmv": {
            "path": source_root
            / "out/data/revision1_v1/HIAI_Tcells/NCBI_raw_collection/"
            / "hiai_tcells_ncbi_cmv_articles_exact_title_abstract_blood_cells.csv",
            "cell_type_column": "AIFI_L2",
        },
        "N5_ncbi_literature": {
            "path": source_root
            / "out/data/revision1_v1/LaManno/NCBI_raw_collection/"
            / "lamanno_ncbi_articles_exact_title_abstract_mouse_embryo.csv",
            "cell_type_column": "celltype",
        },
    }
    query_rows: dict[tuple[str, str, str, str], dict[str, str]] = {}
    pmid_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    raw_pmid_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for dataset_id, source_config in sources.items():
        source_path = source_config["path"]
        cell_type_column = source_config["cell_type_column"]
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                cell_type = row[cell_type_column]
                query_key = (dataset_id, cell_type, row["label"], row["Query"])
                query_rows[query_key] = {
                    "dataset_id": dataset_id,
                    "cell_type": cell_type,
                    "label": row["label"],
                    "query": row["Query"],
                }
                pmid_key = (dataset_id, row["PMID"], cell_type)
                pmid_rows[pmid_key] = {
                    "dataset_id": dataset_id,
                    "pmid": row["PMID"],
                    "cell_type": cell_type,
                }
                raw_key = (source_path.name, row["PMID"], cell_type, row["Query"])
                raw_pmid_rows[raw_key] = {
                    "dataset_id": "NCBI_raw_collection",
                    "source_record": source_path.name,
                    "pmid": row["PMID"],
                    "cell_type": cell_type,
                    "label": row["label"],
                    "query": row["Query"],
                    "query_index": row.get("query_index", ""),
                    "n_pmids_found": row.get("n_pmids_found", ""),
                    "n_pmids_returned": row.get("n_pmids_returned", ""),
                }
    _write_csv(
        destination_root / "query_manifest.csv",
        ["dataset_id", "cell_type", "label", "query"],
        sorted(query_rows.values(), key=lambda row: (row["dataset_id"], row["cell_type"])),
    )
    _write_csv(
        destination_root / "pmid_manifest.csv",
        ["dataset_id", "pmid", "cell_type"],
        sorted(pmid_rows.values(), key=lambda row: (row["dataset_id"], row["pmid"])),
    )
    _write_csv(
        destination_root / "ncbi_raw_pmid_manifest.csv",
        [
            "dataset_id",
            "source_record",
            "pmid",
            "cell_type",
            "label",
            "query",
            "query_index",
            "n_pmids_found",
            "n_pmids_returned",
        ],
        sorted(raw_pmid_rows.values(), key=lambda row: (row["cell_type"], row["pmid"])),
    )


def build_split_manifest(metadata_root: Path, destination: Path) -> None:
    split_manifest = {}
    for split_path in sorted(metadata_root.rglob("split_indices.json")):
        with split_path.open("r", encoding="utf-8") as handle:
            split_manifest[split_path.parent.name] = json.load(handle)
    destination.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("metadata_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    args = parser.parse_args()
    args.destination_root.mkdir(parents=True, exist_ok=True)
    build_literature_manifests(args.source_root, args.destination_root)
    build_split_manifest(args.metadata_root, args.destination_root / "split_manifest.json")


if __name__ == "__main__":
    main()
