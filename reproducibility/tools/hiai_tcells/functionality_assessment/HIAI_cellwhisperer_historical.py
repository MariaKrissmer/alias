from pathlib import Path
import sys

# Get the absolute path to the project root (parent of 'scripts/')
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Construct the path to 'src/'
SRC_DIR = ROOT_DIR / "src"

# Add it to sys.path if it's not already there
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import load_from_disk

out_path = Path("/Users/mariakrissmer/Desktop/Repositories/alias/out/data/HIAI/2025-10-30_HIAI_200k_final")
eval_path = out_path / "evaluation_cellwhisperer"
eval_path.mkdir(parents=True, exist_ok=True)

import pandas as pd

out_path = Path("/Users/mariakrissmer/Desktop/Repositories/alias/out/data/HIAI/2025-10-30_HIAI_200k_final")
scrna_test_path = out_path / "datasets" / "scrna_test"

scrna_test = load_from_disk(str(scrna_test_path))
scrna_test_df = scrna_test.to_pandas()

cell_annotations = scrna_test_df['AIFI_L2']

from evaluation.functionality_cell_similarity_cellwhisperer import functionality_similarity_cellwhisperer, FunctionalitySimilarityConfig


df_additional = pd.read_csv("/Users/mariakrissmer/Desktop/Repositories/alias/out/data/HIAI/2025-10-30_HIAI_200k_final/cellwhisperer/text_scores.csv", index_col=0)

similarity_df = df_additional.T
similarity_df = similarity_df.iloc[:, :-3]

functionality_names = similarity_df.columns.to_list()

config = FunctionalitySimilarityConfig(output_dir=eval_path)

results = functionality_similarity_cellwhisperer(
    similarity_df=similarity_df,
    cell_annotations=cell_annotations,
    functionality_names=list(similarity_df.columns),
    config=config
)

print(results.head())

results.to_csv("/Users/mariakrissmer/Desktop/Repositories/alias/out/data/HIAI/2025-10-30_HIAI_200k_final/evaluation_cellwhisperer/results_df.csv")