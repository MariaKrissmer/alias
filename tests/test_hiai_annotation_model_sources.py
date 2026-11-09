import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "revision1_v1"
    / "HIAI_Tcells"
    / "celltype_annotation"
)

EXPECTED_MODEL_IDS = {
    "MB": "mariakrissmer/ALIAS-MA",
    "MJ": "mariakrissmer/ALIAS-MA-star",
    "MG": "mariakrissmer/ALIAS-MB",
    "MF": "mariakrissmer/ALIAS-MC",
    "MH": "mariakrissmer/ALIAS-MC-star",
    "MI": "mariakrissmer/ALIAS-MD",
    "Base": "neuml/pubmedbert-base-embeddings",
}


class AnnotationModelSourceTests(unittest.TestCase):
    def test_annotation_paths_can_use_existing_data_and_a_new_result_directory(self):
        revision_root = Path("/private/tmp/alias-revision-inputs")
        annotation_output = Path("/private/tmp/alias-annotation-run/annotation")
        synonym_output = Path("/private/tmp/alias-annotation-run/synonyms")
        with patch.dict(
            os.environ,
            {
                "ALIAS_REVISION_DATA_ROOT": str(revision_root),
                "HIAI_TCELLS_ANNOTATION_OUTPUT_DIR": str(annotation_output),
                "HIAI_TCELLS_SYNONYM_OUTPUT_DIR": str(synonym_output),
            },
        ):
            normal = runpy.run_path(str(SCRIPT_DIR / "celltype_annotation.py"))
            synonym = runpy.run_path(str(SCRIPT_DIR / "celltype_annotation_synonym.py"))
            plotting = runpy.run_path(str(SCRIPT_DIR / "_plotting_common.py"))
            generator = runpy.run_path(str(SCRIPT_DIR / "generate_synonym_mapping.py"))

        expected_data = revision_root / "HIAI_Tcells" / "S2_heldout_donor_semantic_200k"
        self.assertEqual(normal["DATASET_DIR"], expected_data)
        self.assertEqual(synonym["DATASET_DIR"], expected_data)
        self.assertEqual(generator["DATASET_DIR"], expected_data)
        self.assertEqual(normal["OUTPUT_DIR"], annotation_output)
        self.assertEqual(synonym["OUTPUT_DIR"], synonym_output)
        self.assertEqual(generator["OUTPUT_DIR"], synonym_output)
        self.assertEqual(plotting["PLOTS_DIR"], annotation_output / "plots")

    def test_synonym_annotation_defaults_to_versioned_manuscript_mapping(self):
        namespace = runpy.run_path(str(SCRIPT_DIR / "celltype_annotation_synonym.py"))
        self.assertEqual(
            namespace["SYNONYM_MAPPING_PATH"],
            Path(__file__).resolve().parents[1]
            / "reproducibility"
            / "inputs"
            / "hiai_tcells"
            / "synonym_mapping.csv",
        )

    def test_non_cmv_annotation_workflows_use_release_model_ids(self):
        for script_name in ("celltype_annotation.py", "celltype_annotation_synonym.py"):
            with self.subTest(script=script_name):
                namespace = runpy.run_path(str(SCRIPT_DIR / script_name))
                models = {
                    model["name"]: str(model["model"])
                    for model in namespace["COMPARISON_MODELS"]
                }
                for canonical_id, hub_id in EXPECTED_MODEL_IDS.items():
                    self.assertEqual(models[canonical_id], hub_id)


if __name__ == "__main__":
    unittest.main()
