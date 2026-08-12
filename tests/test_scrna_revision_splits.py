import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from datasets import Dataset

from alias.data import DatascRNAConfig, TripletGenerationConfig, build_datasets, build_triplets
from alias.data.scrna_splits import (
    SplitIndices,
    load_split_indices,
    make_heldout_donor_and_value_split,
    make_heldout_donor_split,
    validate_no_donor_leakage,
    write_generation_metadata,
    write_split_indices,
    write_split_report,
)


def _donor_adata(n_donors=10, cells_per_donor=4):
    n_obs = n_donors * cells_per_donor
    n_vars = 12
    obs = pd.DataFrame(
        {
            "celltype": ["T_cell", "T_cell", "B_cell", "Monocyte"] * n_donors,
            "donor": [
                f"donor_{donor_idx}"
                for donor_idx in range(n_donors)
                for _ in range(cells_per_donor)
            ],
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {
            "highly_variable": [True] * n_vars,
        },
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    X = sparse.csr_matrix(np.random.default_rng(42).poisson(5, (n_obs, n_vars)))
    return ad.AnnData(X=X, obs=obs, var=var)


def _donor_cmv_adata(n_donors=10, cells_per_donor=4):
    adata = _donor_adata(n_donors=n_donors, cells_per_donor=cells_per_donor)
    donor_status = {
        f"donor_{donor_idx}": "Positive" if donor_idx % 2 else "Negative"
        for donor_idx in range(n_donors)
    }
    adata.obs["subject.cmv"] = adata.obs["donor"].map(donor_status)
    return adata


def _joined_gene_sentence(gene_list):
    return " ".join(list(gene_list))


def test_split_indices_json_roundtrip(tmp_path):
    split = SplitIndices(
        dataset_id="S2",
        strategy="heldout_donor",
        random_state=42,
        train_indices=["cell_0", "cell_1"],
        test_indices=["cell_2"],
        columns={"annotation_column": "celltype", "donor_column": "donor"},
        heldout_values={"donors": ["donor_b"]},
        metadata={"source": "unit-test"},
    )

    path = write_split_indices(split, tmp_path / "split_indices.json")

    assert load_split_indices(path) == split


def test_heldout_donor_split_has_no_donor_leakage(tiny_adata):
    adata = tiny_adata.copy()
    adata.obs["donor"] = [
        "donor_a",
        "donor_a",
        "donor_b",
        "donor_b",
        "donor_c",
        "donor_c",
        "donor_d",
        "donor_d",
        "donor_e",
        "donor_e",
    ]

    split = make_heldout_donor_split(
        adata.obs,
        dataset_id="S2",
        donor_column="donor",
        annotation_column="celltype",
        heldout_donors=["donor_b", "donor_d"],
        random_state=42,
    )

    validate_no_donor_leakage(split, adata.obs, donor_column="donor")
    test_donors = set(adata.obs.loc[split.test_indices, "donor"])
    assert test_donors == {"donor_b", "donor_d"}


def test_validate_no_donor_leakage_raises_for_overlap(tiny_adata):
    adata = tiny_adata.copy()
    adata.obs["donor"] = ["donor_a"] * 5 + ["donor_b"] * 5
    split = SplitIndices(
        dataset_id="bad",
        strategy="heldout_donor",
        random_state=42,
        train_indices=["cell_0", "cell_5"],
        test_indices=["cell_1", "cell_6"],
    )

    with pytest.raises(ValueError, match="Donor leakage"):
        validate_no_donor_leakage(split, adata.obs, donor_column="donor")


def test_write_split_report_creates_summary_and_plots(tmp_path, tiny_adata):
    adata = tiny_adata.copy()
    adata.obs["donor"] = ["donor_a"] * 5 + ["donor_b"] * 5
    split = SplitIndices(
        dataset_id="S2",
        strategy="heldout_donor",
        random_state=42,
        train_indices=[f"cell_{i}" for i in range(5)],
        test_indices=[f"cell_{i}" for i in range(5, 10)],
    )

    artifacts = write_split_report(
        adata.obs,
        split,
        output_dir=tmp_path,
        annotation_column="celltype",
        donor_column="donor",
    )

    assert (tmp_path / "split_composition_summary.csv").exists()
    assert (tmp_path / "celltype_proportions.pdf").exists()
    assert (tmp_path / "donor_proportions.pdf").exists()
    assert (tmp_path / "donor_leakage_report.csv").exists()
    assert artifacts["summary"].endswith("split_composition_summary.csv")


def test_build_datasets_from_split_indices_uses_exact_cell_ids(tiny_adata):
    split = SplitIndices(
        dataset_id="S1",
        strategy="manual",
        random_state=42,
        train_indices=["cell_0", "cell_2", "cell_4"],
        test_indices=["cell_1", "cell_3"],
    )
    config = DatascRNAConfig(
        annotation_column="celltype",
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=False,
    )

    dataset_dict, adata_test = build_datasets(
        adata=tiny_adata,
        datasets=["scrna"],
        scrna_config=config,
        scrna_split_indices=split,
    )

    assert dataset_dict["scrna"]["data"]["index"] == split.train_indices
    assert dataset_dict["scrna"]["test"]["index"] == split.test_indices
    assert adata_test.obs_names.tolist() == split.test_indices


def test_heldout_donor_and_celltype_split_excludes_celltype_from_train(tiny_adata):
    adata = tiny_adata.copy()
    adata.obs["donor"] = [
        "donor_a",
        "donor_a",
        "donor_b",
        "donor_b",
        "donor_c",
        "donor_c",
        "donor_d",
        "donor_d",
        "donor_e",
        "donor_e",
    ]

    split = make_heldout_donor_and_value_split(
        adata.obs,
        dataset_id="S4",
        donor_column="donor",
        value_column="celltype",
        annotation_column="celltype",
        heldout_donors=["donor_b", "donor_d"],
        heldout_values=["Monocyte"],
        heldout_key="cell_types",
        random_state=42,
    )

    validate_no_donor_leakage(split, adata.obs, donor_column="donor")
    train_celltypes = set(adata.obs.loc[split.train_indices, "celltype"])
    assert "Monocyte" not in train_celltypes


def test_write_generation_metadata_collects_dataset_details(tmp_path):
    split = SplitIndices(
        dataset_id="S2",
        strategy="heldout_donor",
        random_state=42,
        train_indices=["cell_0"],
        test_indices=["cell_1"],
        columns={"annotation_column": "celltype", "donor_column": "donor"},
        heldout_values={"donors": ["donor_b"]},
    )

    path = write_generation_metadata(
        tmp_path / "generation_metadata.json",
        split=split,
        source="HIAI",
        scrna_config={"annotation_column": "celltype", "cs_length": [50]},
        dataset_artifacts={"scrna_data": "datasets/scrna_data"},
        report_artifacts={"summary": "reports/split_composition_summary.csv"},
    )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["dataset_id"] == "S2"
    assert payload["source"] == "HIAI"
    assert payload["split"]["strategy"] == "heldout_donor"
    assert payload["scrna_config"]["cs_length"] == [50]
    assert payload["dataset_artifacts"]["scrna_data"] == "datasets/scrna_data"
    assert payload["report_artifacts"]["summary"].endswith("split_composition_summary.csv")


def test_build_datasets_can_generate_and_save_s1_artifacts(tmp_path, tiny_adata):
    config = DatascRNAConfig(
        dataset_id="S1_random_stratified_semantic",
        annotation_column="celltype",
        split_strategy="random_stratified",
        output_dir=tmp_path / "S1_random_stratified_semantic",
        source="tiny_adata",
        save_artifacts=True,
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=True,
        test_size=0.3,
        random_state=42,
    )

    dataset_dict, adata_test = build_datasets(
        adata=tiny_adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    output_dir = tmp_path / "S1_random_stratified_semantic"
    assert (output_dir / "datasets" / "scrna_data").exists()
    assert (output_dir / "datasets" / "scrna_test").exists()
    assert (output_dir / "adata_test.h5ad").exists()
    assert (output_dir / "metadata" / "split_indices.json").exists()
    assert (output_dir / "metadata" / "generation_metadata.json").exists()
    assert (output_dir / "reports" / "split_composition_summary.csv").exists()
    assert (output_dir / "reports" / "celltype_proportions.pdf").exists()

    split = load_split_indices(output_dir / "metadata" / "split_indices.json")
    assert split.dataset_id == "S1_random_stratified_semantic"
    assert split.strategy == "random_stratified"
    assert dataset_dict["scrna"]["test"]["index"] == split.test_indices
    assert adata_test.obs_names.tolist() == split.test_indices


def test_build_datasets_splits_after_preprocessing_filters_cells(tmp_path, tiny_adata):
    adata = tiny_adata.copy()
    adata.X[0, :] = 0

    config = DatascRNAConfig(
        dataset_id="S1_preprocessed",
        annotation_column="celltype",
        split_strategy="random_stratified",
        output_dir=tmp_path / "S1_preprocessed",
        source="tiny_adata",
        save_artifacts=True,
        preprocessing=True,
        highly_variable_genes=False,
        housekeeping_genes=True,
        cs_length=(5,),
        semantic=False,
        test_size=0.3,
        random_state=42,
        min_genes=1,
        min_cells=1,
        min_batch_cells=1,
        hvg_number=10,
        mt_threshold=100,
        verbose=False,
    )

    dataset_dict, adata_test = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    all_split_indices = (
        list(dataset_dict["scrna"]["data"]["index"])
        + list(dataset_dict["scrna"]["test"]["index"])
    )
    assert "cell_0" not in all_split_indices
    assert "cell_0" not in adata_test.obs_names
    assert (tmp_path / "S1_preprocessed" / "reports" / "scrna_train_head.csv").exists()


def test_heldout_donor_strategy_defaults_donor_test_size_to_test_size():
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S2_default_donor_fraction",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=False,
        test_size=0.4,
        random_state=42,
    )

    dataset_dict, adata_test = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    train_indices = list(dataset_dict["scrna"]["data"]["index"])
    test_indices = list(dataset_dict["scrna"]["test"]["index"])
    train_donors = set(adata.obs.loc[train_indices, "donor"])
    test_donors = set(adata.obs.loc[test_indices, "donor"])

    assert train_donors.isdisjoint(test_donors)
    assert len(test_donors) == 4
    assert len(train_indices) == round((10 - 4) * 4 * 0.4)
    assert len(test_indices) == round(4 * 4 * 0.4)
    assert adata_test.obs_names.tolist() == test_indices


def test_heldout_donor_strategy_uses_explicit_donor_test_size():
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S2_explicit_donor_fraction",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=False,
        donor_test_size=0.3,
        test_size=0.5,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    train_indices = list(dataset_dict["scrna"]["data"]["index"])
    test_indices = list(dataset_dict["scrna"]["test"]["index"])
    train_donors = set(adata.obs.loc[train_indices, "donor"])
    test_donors = set(adata.obs.loc[test_indices, "donor"])

    assert train_donors.isdisjoint(test_donors)
    assert len(test_donors) == 3
    assert len(train_indices) == round((10 - 3) * 4 * 0.5)
    assert len(test_indices) == round(3 * 4 * 0.5)


def test_heldout_donor_strategy_uses_total_cells_with_test_fraction():
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S2_total_cells",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=False,
        donor_test_size=0.3,
        test_size=0.3,
        total_cells=20,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    train_indices = list(dataset_dict["scrna"]["data"]["index"])
    test_indices = list(dataset_dict["scrna"]["test"]["index"])
    train_donors = set(adata.obs.loc[train_indices, "donor"])
    test_donors = set(adata.obs.loc[test_indices, "donor"])

    assert train_donors.isdisjoint(test_donors)
    assert len(test_donors) == 3
    assert len(train_indices) == 14
    assert len(test_indices) == 6
    assert len(train_indices) + len(test_indices) == 20


def test_heldout_donor_total_cells_metadata_is_saved(tmp_path):
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    output_dir = tmp_path / "S2_total_cells"
    config = DatascRNAConfig(
        dataset_id="S2_total_cells",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        output_dir=output_dir,
        source="donor_adata",
        save_artifacts=True,
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=False,
        donor_test_size=0.3,
        test_size=0.3,
        total_cells=20,
        random_state=42,
    )

    build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    with (output_dir / "metadata" / "generation_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    split_metadata = metadata["split"]["metadata"]

    assert split_metadata["total_cells"] == 20
    assert split_metadata["n_target_train_cells"] == 14
    assert split_metadata["n_target_test_cells"] == 6
    assert split_metadata["n_selected_total_cells"] == 20
    assert metadata["scrna_config"]["total_cells"] == 20


def test_heldout_donor_and_value_strategy_total_cells_counts_final_split():
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S4_total_cells",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor_and_value",
        heldout_column="celltype",
        heldout_values=["B_cell"],
        heldout_key="cell_types",
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=False,
        donor_test_size=0.3,
        test_size=0.3,
        total_cells=16,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    train_indices = list(dataset_dict["scrna"]["data"]["index"])
    test_indices = list(dataset_dict["scrna"]["test"]["index"])

    assert len(train_indices) == 11
    assert len(test_indices) == 5
    assert len(train_indices) + len(test_indices) == 16
    assert "B_cell" not in set(adata.obs.loc[train_indices, "celltype"])


def test_heldout_donor_strategy_subsamples_toward_full_celltype_proportions():
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S2_stratified",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=False,
        donor_test_size=0.3,
        test_size=0.5,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    full_props = adata.obs["celltype"].value_counts(normalize=True).sort_index()
    for split_name in ["data", "test"]:
        indices = list(dataset_dict["scrna"][split_name]["index"])
        split_props = adata.obs.loc[indices, "celltype"].value_counts(normalize=True).sort_index()
        split_props = split_props.reindex(full_props.index, fill_value=0)
        assert (split_props - full_props).abs().max() <= 0.1


def test_build_datasets_can_generate_and_save_s2_heldout_donor_artifacts(tmp_path):
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S2_heldout_donor_semantic",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        output_dir=tmp_path / "S2_heldout_donor_semantic",
        source="donor_adata",
        save_artifacts=True,
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=True,
        donor_test_size=0.3,
        test_size=0.5,
        random_state=42,
    )

    dataset_dict, adata_test = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    output_dir = tmp_path / "S2_heldout_donor_semantic"
    split = load_split_indices(output_dir / "metadata" / "split_indices.json")
    with (output_dir / "metadata" / "generation_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert split.strategy == "heldout_donor"
    assert split.metadata["donor_test_size"] == 0.3
    assert split.metadata["test_size"] == 0.5
    assert split.metadata["stratified_subsample"] is True
    assert len(split.heldout_values["donors"]) == 3
    assert dataset_dict["scrna"]["test"]["index"] == split.test_indices
    assert adata_test.obs_names.tolist() == split.test_indices
    assert (output_dir / "reports" / "split_composition_summary.csv").exists()
    assert (output_dir / "reports" / "donor_proportions.pdf").exists()
    assert (output_dir / "reports" / "donor_leakage_report.csv").exists()
    assert (output_dir / "reports" / "scrna_train_head.csv").exists()
    assert metadata["report_artifacts"]["donor_proportions"].endswith("donor_proportions.pdf")


def test_heldout_donor_strategy_splits_after_preprocessing_filters_cells(tmp_path, tiny_adata):
    adata = tiny_adata.copy()
    adata.X[0, :] = 0
    adata.obs["donor"] = [
        "donor_a",
        "donor_a",
        "donor_b",
        "donor_b",
        "donor_c",
        "donor_c",
        "donor_d",
        "donor_d",
        "donor_e",
        "donor_e",
    ]
    config = DatascRNAConfig(
        dataset_id="S2_preprocessed",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        output_dir=tmp_path / "S2_preprocessed",
        source="tiny_adata",
        save_artifacts=True,
        preprocessing=True,
        highly_variable_genes=False,
        housekeeping_genes=True,
        cs_length=(5,),
        semantic=False,
        donor_test_size=0.4,
        test_size=0.5,
        random_state=42,
        min_genes=1,
        min_cells=1,
        min_batch_cells=1,
        hvg_number=10,
        mt_threshold=100,
        verbose=False,
    )

    dataset_dict, adata_test = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    all_split_indices = (
        list(dataset_dict["scrna"]["data"]["index"])
        + list(dataset_dict["scrna"]["test"]["index"])
    )
    assert "cell_0" not in all_split_indices
    assert "cell_0" not in adata_test.obs_names


def test_heldout_donor_and_value_strategy_removes_multiple_values_from_train_only():
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S4_heldout_celltypes",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor_and_value",
        heldout_column="celltype",
        heldout_values=["Monocyte", "B_cell"],
        heldout_key="cell_types",
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=True,
        donor_test_size=0.3,
        test_size=0.5,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    train_indices = list(dataset_dict["scrna"]["data"]["index"])
    test_indices = list(dataset_dict["scrna"]["test"]["index"])
    train_celltypes = set(adata.obs.loc[train_indices, "celltype"])
    test_celltypes = set(adata.obs.loc[test_indices, "celltype"])
    train_donors = set(adata.obs.loc[train_indices, "donor"])
    test_donors = set(adata.obs.loc[test_indices, "donor"])

    assert train_donors.isdisjoint(test_donors)
    assert "B_cell" not in train_celltypes
    assert "Monocyte" not in train_celltypes
    assert {"B_cell", "Monocyte"}.issubset(test_celltypes)


def test_heldout_donor_and_value_strategy_saves_metadata_and_artifacts(tmp_path):
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S4_heldout_donor_heldout_celltype_semantic",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor_and_value",
        heldout_column="celltype",
        heldout_values=["Monocyte", "B_cell"],
        heldout_key="cell_types",
        output_dir=tmp_path / "S4_heldout_donor_heldout_celltype_semantic",
        source="donor_adata",
        save_artifacts=True,
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=True,
        donor_test_size=0.3,
        test_size=0.5,
        random_state=42,
    )

    dataset_dict, adata_test = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
    )

    output_dir = tmp_path / "S4_heldout_donor_heldout_celltype_semantic"
    split = load_split_indices(output_dir / "metadata" / "split_indices.json")
    with (output_dir / "metadata" / "generation_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    train_indices = list(dataset_dict["scrna"]["data"]["index"])
    assert split.strategy == "heldout_donor_and_celltype"
    assert split.columns["heldout_column"] == "celltype"
    assert split.heldout_values["cell_types"] == ["B_cell", "Monocyte"]
    assert split.metadata["n_train_removed_heldout_cells"] > 0
    assert split.test_indices == dataset_dict["scrna"]["test"]["index"]
    assert adata_test.obs_names.tolist() == split.test_indices
    assert set(adata.obs.loc[train_indices, "celltype"]).isdisjoint({"B_cell", "Monocyte"})
    assert metadata["split"]["heldout_values"]["cell_types"] == ["B_cell", "Monocyte"]
    assert (output_dir / "reports" / "split_composition_summary.csv").exists()
    assert (output_dir / "reports" / "donor_proportions.pdf").exists()
    assert (output_dir / "reports" / "donor_leakage_report.csv").exists()
    assert (output_dir / "reports" / "scrna_train_head.csv").exists()


def test_nonsemantic_train_values_keep_cells_and_mix_sentence_modes():
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S6_heldout_donor_celltype_nonsemantic",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        nonsemantic_train_column="celltype",
        nonsemantic_train_values=["B_cell", "Monocyte"],
        nonsemantic_train_key="cell_types",
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=True,
        donor_test_size=0.3,
        test_size=0.8,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
        train_semantic=True,
        test_semantic=False,
    )

    train_df = dataset_dict["scrna"]["data"].to_pandas()
    test_df = dataset_dict["scrna"]["test"].to_pandas()
    train_donors = set(adata.obs.loc[train_df["index"], "donor"])
    test_donors = set(adata.obs.loc[test_df["index"], "donor"])
    nonsemantic_train = train_df[train_df["celltype"].isin(["B_cell", "Monocyte"])]
    semantic_train = train_df[~train_df["celltype"].isin(["B_cell", "Monocyte"])]

    assert train_donors.isdisjoint(test_donors)
    assert {"B_cell", "Monocyte"}.issubset(set(train_df["celltype"]))
    assert set(nonsemantic_train["sentence_mode"]) == {"nonsemantic"}
    assert set(semantic_train["sentence_mode"]) == {"semantic"}
    assert all(
        row.sentence1 == _joined_gene_sentence(row.gene_list)
        for row in nonsemantic_train.itertuples()
    )
    assert all(
        row.sentence1 != _joined_gene_sentence(row.gene_list)
        for row in semantic_train.itertuples()
    )
    assert set(test_df["sentence_mode"]) == {"nonsemantic"}
    assert all(row.sentence1 == _joined_gene_sentence(row.gene_list) for row in test_df.itertuples())


def test_nonsemantic_train_values_save_metadata_and_sentence_mode_report(tmp_path):
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    output_dir = tmp_path / "S6_heldout_donor_celltype_nonsemantic"
    config = DatascRNAConfig(
        dataset_id="S6_heldout_donor_celltype_nonsemantic",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        nonsemantic_train_column="celltype",
        nonsemantic_train_values=["B_cell", "Monocyte"],
        nonsemantic_train_key="cell_types",
        output_dir=output_dir,
        source="donor_adata",
        save_artifacts=True,
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=True,
        donor_test_size=0.3,
        test_size=0.8,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
        train_semantic=True,
        test_semantic=False,
    )

    with (output_dir / "metadata" / "generation_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    sentence_report = pd.read_csv(output_dir / "reports" / "sentence_mode_report.csv")
    train_head = pd.read_csv(output_dir / "reports" / "scrna_train_head.csv")

    assert metadata["sentence_mode"]["nonsemantic_train_column"] == "celltype"
    assert metadata["sentence_mode"]["cell_types"] == ["B_cell", "Monocyte"]
    assert metadata["sentence_mode"]["n_train_nonsemantic_rows"] > 0
    assert metadata["sentence_mode"]["n_train_rows"] == len(dataset_dict["scrna"]["data"])
    assert metadata["report_artifacts"]["sentence_mode_report"].endswith("sentence_mode_report.csv")
    assert "sentence_mode" in train_head.columns
    assert {"train", "test"}.issubset(set(sentence_report["split"]))
    assert {"semantic", "nonsemantic"}.issubset(set(sentence_report["sentence_mode"]))
    assert (output_dir / "reports" / "split_composition_summary.csv").exists()
    assert (output_dir / "reports" / "donor_proportions.pdf").exists()
    assert (output_dir / "reports" / "donor_leakage_report.csv").exists()
    assert (output_dir / "reports" / "sentence_mode_report.csv").exists()


def test_shuffle_train_labels_happens_after_semantic_sentence_generation():
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S5_heldout_donor_semantic_shuffled_labels",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=True,
        donor_test_size=0.3,
        test_size=0.8,
        shuffle_train_labels=True,
        label_shuffle_seed=7,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
        train_semantic=True,
        test_semantic=False,
    )

    train_ds = dataset_dict["scrna"]["data"]
    test_ds = dataset_dict["scrna"]["test"]
    train_df = train_ds.to_pandas()
    test_df = test_ds.to_pandas()

    assert "sentence1" in train_ds.features
    assert "original_label" in train_ds.features
    assert "original_label" not in test_ds.features
    assert not train_df["sentence1"].str.startswith("gene_").all()
    assert train_df["celltype"].equals(train_df["original_label"])
    assert train_df["label"].value_counts().sort_index().equals(
        train_df["original_label"].value_counts().sort_index()
    )
    assert (train_df["label"] != train_df["original_label"]).any()
    assert test_df["label"].equals(test_df["celltype"])


def test_shuffle_train_labels_saves_metadata_and_reports(tmp_path):
    adata = _donor_adata(n_donors=10, cells_per_donor=4)
    output_dir = tmp_path / "S5_heldout_donor_semantic_shuffled_labels"
    config = DatascRNAConfig(
        dataset_id="S5_heldout_donor_semantic_shuffled_labels",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        output_dir=output_dir,
        source="donor_adata",
        save_artifacts=True,
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=True,
        donor_test_size=0.3,
        test_size=0.8,
        shuffle_train_labels=True,
        label_shuffle_seed=7,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
        train_semantic=True,
        test_semantic=False,
    )

    with (output_dir / "metadata" / "generation_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    train_head = pd.read_csv(output_dir / "reports" / "scrna_train_head.csv")

    assert "original_label" in train_head.columns
    assert "label_shuffle" in metadata
    assert metadata["label_shuffle"]["shuffle_train_labels"] is True
    assert metadata["label_shuffle"]["label_shuffle_seed"] == 7
    assert metadata["label_shuffle"]["label_shuffle_original_column"] == "original_label"
    assert metadata["label_shuffle"]["n_train_rows"] == len(dataset_dict["scrna"]["data"])
    assert metadata["label_shuffle"]["n_changed_labels"] > 0
    assert metadata["report_artifacts"]["label_shuffle_report"].endswith("label_shuffle_report.csv")
    assert metadata["report_artifacts"]["label_shuffle_confusion"].endswith("label_shuffle_confusion.pdf")
    assert metadata["report_artifacts"]["label_shuffle_correct_label_proportions"].endswith(
        "label_shuffle_correct_label_proportions.pdf"
    )
    shuffle_report = pd.read_csv(output_dir / "reports" / "label_shuffle_report.csv")
    assert {"proportion_within_original_label", "correct_label"}.issubset(shuffle_report.columns)
    assert shuffle_report["proportion_within_original_label"].between(0, 1).all()
    assert shuffle_report["correct_label"].any()
    assert (output_dir / "reports" / "label_shuffle_report.csv").exists()
    assert (output_dir / "reports" / "label_shuffle_confusion.pdf").exists()
    assert (output_dir / "reports" / "label_shuffle_correct_label_proportions.pdf").exists()
    assert (output_dir / "reports" / "donor_leakage_report.csv").exists()


def test_triplet_generation_uses_shuffled_label_column_not_annotation_column():
    dataset_dict = {
        "scrna": {
            "data": Dataset.from_dict(
                {
                    "sentence1": ["t_cell_a", "b_cell_a", "t_cell_b", "b_cell_b"],
                    "celltype": ["T_cell", "B_cell", "T_cell", "B_cell"],
                    "original_label": ["T_cell", "B_cell", "T_cell", "B_cell"],
                    "label": ["group_1", "group_1", "group_2", "group_2"],
                }
            )
        }
    }
    config = TripletGenerationConfig(
        annotation_column="celltype",
        loss="MNR",
        eval_split=0.5,
        seed=42,
        random_negative_mining=True,
        hard_negative_mining=False,
    )

    triplet_dict = build_triplets(dataset_dict, triplet_config=config)
    paired_rows = (
        triplet_dict["scrna"]["train_MNR_rnm"].to_pandas().loc[:, ["sentence1", "sentence2"]]
    )
    paired_rows = pd.concat(
        [
            paired_rows,
            triplet_dict["scrna"]["eval_MNR_rnm"].to_pandas().loc[:, ["sentence1", "sentence2"]],
        ],
        ignore_index=True,
    )
    observed_pairs = {tuple(sorted(row)) for row in paired_rows.to_numpy().tolist()}

    assert ("b_cell_a", "t_cell_a") in observed_pairs
    assert ("b_cell_b", "t_cell_b") in observed_pairs
    assert ("t_cell_a", "t_cell_b") not in observed_pairs
    assert ("b_cell_a", "b_cell_b") not in observed_pairs


def test_cmv_metadata_is_mapped_into_dataset_columns_labels_and_semantic_sentences():
    adata = _donor_cmv_adata(n_donors=10, cells_per_donor=4)
    config = DatascRNAConfig(
        dataset_id="S7_heldout_donor_semantic_cmv",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        disease_column="subject.cmv",
        disease_value_map={"Positive": "CMV", "Negative": "healthy"},
        disease_output_column="cmv_status",
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=True,
        template_weights_disease={"genes_celltype_disease": 1.0},
        donor_test_size=0.3,
        test_size=0.8,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
        train_semantic=True,
        test_semantic=False,
    )

    train_df = dataset_dict["scrna"]["data"].to_pandas()
    test_df = dataset_dict["scrna"]["test"].to_pandas()

    assert set(train_df["cmv_status"]).issubset({"CMV", "healthy"})
    assert set(test_df["cmv_status"]).issubset({"CMV", "healthy"})
    assert "disease_status" not in train_df.columns
    assert train_df["label"].equals(train_df["celltype"] + "_" + train_df["cmv_status"])
    assert test_df["label"].equals(test_df["celltype"] + "_" + test_df["cmv_status"])
    assert train_df["sentence1"].str.contains("CMV|healthy", regex=True).any()


def test_cmv_metadata_reports_are_saved_for_s7(tmp_path):
    adata = _donor_cmv_adata(n_donors=10, cells_per_donor=4)
    output_dir = tmp_path / "S7_heldout_donor_semantic_cmv"
    config = DatascRNAConfig(
        dataset_id="S7_heldout_donor_semantic_cmv",
        annotation_column="celltype",
        donor_column="donor",
        split_strategy="heldout_donor",
        disease_column="subject.cmv",
        disease_value_map={"Positive": "CMV", "Negative": "healthy"},
        disease_output_column="cmv_status",
        output_dir=output_dir,
        source="donor_cmv_adata",
        save_artifacts=True,
        preprocessing=False,
        highly_variable_genes=False,
        cs_length=(5,),
        semantic=True,
        donor_test_size=0.3,
        test_size=0.8,
        random_state=42,
    )

    dataset_dict, adata_test = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=config,
        train_semantic=True,
        test_semantic=False,
    )

    with (output_dir / "metadata" / "generation_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    split = load_split_indices(output_dir / "metadata" / "split_indices.json")
    summary = pd.read_csv(output_dir / "reports" / "split_composition_summary.csv")
    train_head = pd.read_csv(output_dir / "reports" / "scrna_train_head.csv")

    train_indices = list(dataset_dict["scrna"]["data"]["index"])
    test_indices = list(dataset_dict["scrna"]["test"]["index"])
    train_donors = set(adata.obs.loc[train_indices, "donor"])
    test_donors = set(adata.obs.loc[test_indices, "donor"])

    assert train_donors.isdisjoint(test_donors)
    assert adata_test.obs_names.tolist() == split.test_indices
    assert "cmv_status" in train_head.columns
    assert "cmv_status" in set(summary["column"])
    assert metadata["scrna_config"]["disease_value_map"] == {"Negative": "healthy", "Positive": "CMV"}
    assert metadata["report_artifacts"]["cmv_status_proportions"].endswith("cmv_status_proportions.pdf")
    assert (output_dir / "reports" / "split_composition_summary.csv").exists()
    assert (output_dir / "reports" / "donor_proportions.pdf").exists()
    assert (output_dir / "reports" / "donor_leakage_report.csv").exists()
    assert (output_dir / "reports" / "cmv_status_proportions.pdf").exists()
