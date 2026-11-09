import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from alias.evaluation.celltype_annotation.benchmark import (
    REQUIRED_EVALUATION_FILES,
    REQUIRED_PREDICTION_FILE,
    get_annotation_cache_status,
)


class AnnotationCacheSourceTests(unittest.TestCase):
    def test_reuses_predictions_only_for_the_configured_model_source(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / REQUIRED_PREDICTION_FILE).touch()
            for name in REQUIRED_EVALUATION_FILES:
                (model_dir / name).touch()
            (model_dir / "metadata.json").write_text(
                json.dumps({"model": "mariakrissmer/old-model"}),
                encoding="utf-8",
            )

            self.assertEqual(
                get_annotation_cache_status(
                    model_dir,
                    expected_model_source="mariakrissmer/ALIAS-MA",
                ),
                "different_model_source",
            )
            self.assertEqual(
                get_annotation_cache_status(
                    model_dir,
                    expected_model_source="mariakrissmer/old-model",
                ),
                "complete",
            )
            self.assertEqual(get_annotation_cache_status(model_dir), "complete")


if __name__ == "__main__":
    unittest.main()
