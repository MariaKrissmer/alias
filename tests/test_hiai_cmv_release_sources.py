from pathlib import Path
import os
import runpy
import unittest
from unittest.mock import patch


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts/revision1_v1/HIAI_Tcells/functionality_assessment/run_functionality_downstream_disease_comparison.py"
)


class CMVReleaseSourceTests(unittest.TestCase):
    def test_embedding_generation_stays_inside_the_current_run(self) -> None:
        embedding_root = Path("/private/tmp/alias-cmv-current-run/embeddings")
        with patch.dict(os.environ, {"HIAI_TCELLS_CMV_EMBEDDING_ROOT": str(embedding_root)}):
            namespace = runpy.run_path(str(RUNNER))
        generate = namespace["_generate_matching_disease_embeddings"]
        with patch.dict(generate.__globals__, {"_load_scrna_test": lambda: object()}):
            with patch("alias.evaluation.generate_embeddings") as generate_embeddings:
                generate_embeddings.return_value = {"ALIASMA": {"scrna": {}}}
                generate("MB", ["CMV description"])

        config = generate_embeddings.call_args.kwargs["embedding_config"]
        self.assertEqual(config.embedding_models, ["mariakrissmer/ALIAS-MA"])
        self.assertEqual(Path(config.output_dir), embedding_root / "MB")


if __name__ == "__main__":
    unittest.main()
