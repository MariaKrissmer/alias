from pathlib import Path
import runpy
import unittest

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "revision1_v1"
    / "HIAI_Tcells"
    / "celltype_annotation"
    / "generate_synonym_mapping.py"
)
CURATED_PATH = (
    Path(__file__).resolve().parents[1]
    / "reproducibility"
    / "inputs"
    / "hiai_tcells"
    / "synonym_mapping.csv"
)


class SynonymMappingTests(unittest.TestCase):
    def test_curated_manuscript_mapping_preserves_manual_treg_synonyms(self):
        mapping = pd.read_csv(CURATED_PATH)

        self.assertEqual(len(mapping), 42)
        manual = mapping.loc[mapping["synonym_source"] == "custom_synonym"]
        self.assertEqual(
            set(manual["synonym_label"]),
            {"FOXP3+ T lymphocyte", "FOXP3+ T cell"},
        )
        self.assertEqual(set(manual["canonical_label"]), {"Treg"})

    def test_mapping_includes_canonical_and_deduplicated_ontology_synonyms(self):
        namespace = runpy.run_path(str(SCRIPT_PATH))
        build_records = namespace["_synonym_records"]
        build_records.__globals__["_cl_synonyms_by_label"] = lambda: {
            "Treg": [
                ("cell_ontology_synonym", "FOXP3+ T lymphocyte"),
                ("cell_ontology_exact_synonym", "foxp3+ t lymphocyte"),
                ("cell_ontology_synonym", "treg"),
            ]
        }

        mapping = build_records({"Treg", "MAIT"})

        self.assertEqual(
            list(mapping.columns),
            ["canonical_label", "synonym_label", "is_canonical_label", "synonym_source"],
        )
        self.assertEqual(
            mapping.loc[mapping["canonical_label"] == "Treg", "synonym_label"].tolist(),
            ["Treg", "FOXP3+ T lymphocyte"],
        )
        self.assertEqual(
            mapping.loc[mapping["canonical_label"] == "MAIT", "synonym_label"].tolist(),
            ["MAIT"],
        )


if __name__ == "__main__":
    unittest.main()
