from __future__ import annotations

import json

import pandas as pd

from alias.data import DataNCBIConfig, build_datasets
from alias.data.cl import DataCLConfig, collect_cl_terms
from alias.data.ncbi import (
    apply_ncbi_multilabel_pmid_filter,
    build_ncbi_raw_articles,
    build_pubmed_queries,
    shuffle_ncbi_labels,
)


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return str(text).split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def _raw_articles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PMID": "1",
                "Title": "Treg title",
                "Abstract": "Treg abstract",
                "Query": "(homo sapiens[Mesh]) AND Treg",
                "label": "Treg",
                "disease": "",
                "AIFI_L2": "Treg",
                "n_pmids_found": 10,
                "n_pmids_returned": 1,
                "query_index": 0,
            },
            {
                "PMID": "2",
                "Title": "B cell title",
                "Abstract": "B cell abstract",
                "Query": "(homo sapiens[Mesh]) AND B cell",
                "label": "B cell",
                "disease": "",
                "AIFI_L2": "B cell",
                "n_pmids_found": 20,
                "n_pmids_returned": 1,
                "query_index": 0,
            },
            {
                "PMID": "3",
                "Title": "Monocyte title",
                "Abstract": "Monocyte abstract",
                "Query": "(homo sapiens[Mesh]) AND Intermediate monocyte",
                "label": "Intermediate monocyte",
                "disease": "",
                "AIFI_L2": "Intermediate monocyte",
                "n_pmids_found": 30,
                "n_pmids_returned": 1,
                "query_index": 0,
            },
        ]
    )


def test_build_ncbi_raw_articles_reuses_existing_csv(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw_articles.csv"
    expected = _raw_articles()
    expected.to_csv(raw_path, index=False)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("fetch_articles should not be called when raw CSV exists")

    monkeypatch.setattr("alias.data.ncbi.fetch_articles", fail_fetch)
    cfg = DataNCBIConfig(
        email="test@example.com",
        annotation_column="AIFI_L2",
        raw_articles_path=raw_path,
        fetch_if_missing=False,
    )

    observed = build_ncbi_raw_articles(None, cfg)

    assert len(observed) == len(expected)
    assert set(observed["label"]) == set(expected["label"])


def test_pubmed_query_default_uses_existing_broad_mesh_query():
    cfg = DataNCBIConfig(
        email="test@example.com",
        organism="homo sapiens",
    )

    queries, query_to_label, query_to_disease = build_pubmed_queries("Memory CD4 T cell", cfg)

    assert queries == ["(homo sapiens[Mesh]) AND Memory CD4 T cell"]
    assert query_to_label[queries[0]] == "Memory CD4 T cell"
    assert query_to_disease[queries[0]] == ""


def test_pubmed_query_exact_title_abstract_uses_singular_and_plural_phrases():
    cfg = DataNCBIConfig(
        email="test@example.com",
        organism="homo sapiens",
        query_mode="exact_title_abstract",
    )

    queries, query_to_label, query_to_disease = build_pubmed_queries("Memory CD4 T cell", cfg)

    assert queries == [
        '(homo sapiens[Mesh]) AND ("Memory CD4 T cell"[Title/Abstract] OR '
        '"Memory CD4 T cells"[Title/Abstract])'
    ]
    assert query_to_label[queries[0]] == "Memory CD4 T cell"
    assert query_to_disease[queries[0]] == ""


def test_pubmed_query_exact_title_abstract_supports_tissue_mesh_constraint():
    cfg = DataNCBIConfig(
        email="test@example.com",
        organism="homo sapiens",
        query_mode="exact_title_abstract",
        tissue="Blood Cells[Mesh]",
    )

    queries, query_to_label, query_to_disease = build_pubmed_queries("Memory CD4 T cell", cfg)

    assert queries == [
        '(homo sapiens[Mesh]) AND (Blood Cells[Mesh]) AND '
        '("Memory CD4 T cell"[Title/Abstract] OR "Memory CD4 T cells"[Title/Abstract])'
    ]
    assert query_to_label[queries[0]] == "Memory CD4 T cell"
    assert query_to_disease[queries[0]] == ""


def test_pubmed_query_exact_title_abstract_keeps_existing_plural_phrase():
    cfg = DataNCBIConfig(
        email="test@example.com",
        organism="homo sapiens",
        query_mode="exact_title_abstract",
    )

    queries, _, _ = build_pubmed_queries("B cells", cfg)

    assert queries == ['(homo sapiens[Mesh]) AND ("B cells"[Title/Abstract])']


def test_pubmed_query_exact_title_abstract_supports_disease_split():
    cfg = DataNCBIConfig(
        email="test@example.com",
        organism="homo sapiens",
        query_mode="exact_title_abstract",
        diseases=["CMV"],
    )

    queries, query_to_label, query_to_disease = build_pubmed_queries("Treg", cfg)

    assert queries == [
        '((homo sapiens[Mesh]) AND ("Treg"[Title/Abstract] OR "Tregs"[Title/Abstract]) AND CMV)',
        '((homo sapiens[Mesh]) AND ("Treg"[Title/Abstract] OR "Tregs"[Title/Abstract]) NOT CMV)',
    ]
    assert query_to_label[queries[0]] == "Treg_CMV"
    assert query_to_label[queries[1]] == "Treg"
    assert query_to_disease[queries[0]] == "CMV"
    assert query_to_disease[queries[1]] == ""


def test_ncbi_heldout_values_are_removed_before_processing(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw_articles.csv"
    _raw_articles().to_csv(raw_path, index=False)
    monkeypatch.setattr(
        "alias.data.ncbi.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    cfg = DataNCBIConfig(
        email="test@example.com",
        dataset_id="N2_ncbi_literature_heldout_celltype",
        annotation_column="AIFI_L2",
        raw_articles_path=raw_path,
        fetch_if_missing=False,
        heldout_values=["Treg", "Intermediate monocyte"],
    )

    dataset_dict, _ = build_datasets(None, datasets=["ncbi"], ncbi_config=cfg)
    train_df = dataset_dict["ncbi"]["data"].to_pandas()

    assert set(train_df["label"]) == {"B cell"}
    assert "Treg" not in set(train_df["AIFI_L2"])
    assert "Intermediate monocyte" not in set(train_df["AIFI_L2"])


def test_ncbi_multilabel_pmids_are_removed_before_processing(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw_articles.csv"
    pd.DataFrame(
        [
            {
                "PMID": "1",
                "Title": "shared title A",
                "Abstract": "shared abstract A",
                "Query": "q1",
                "label": "Treg",
                "disease": "",
                "AIFI_L2": "Treg",
                "n_pmids_found": 2,
                "n_pmids_returned": 1,
                "query_index": 0,
            },
            {
                "PMID": "1",
                "Title": "shared title B",
                "Abstract": "shared abstract B",
                "Query": "q2",
                "label": "MAIT",
                "disease": "",
                "AIFI_L2": "MAIT",
                "n_pmids_found": 2,
                "n_pmids_returned": 1,
                "query_index": 0,
            },
            {
                "PMID": "2",
                "Title": "single title",
                "Abstract": "single abstract",
                "Query": "q3",
                "label": "DN T cell",
                "disease": "",
                "AIFI_L2": "DN T cell",
                "n_pmids_found": 1,
                "n_pmids_returned": 1,
                "query_index": 0,
            },
        ]
    ).to_csv(raw_path, index=False)
    monkeypatch.setattr(
        "alias.data.ncbi.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    cfg = DataNCBIConfig(
        email="test@example.com",
        annotation_column="AIFI_L2",
        raw_articles_path=raw_path,
        fetch_if_missing=False,
        remove_multilabel_pmids=True,
    )

    dataset_dict, _ = build_datasets(None, datasets=["ncbi"], ncbi_config=cfg)
    train_df = dataset_dict["ncbi"]["data"].to_pandas()

    assert set(train_df["PMID"]) == {"2"}
    assert set(train_df["label"]) == {"DN T cell"}


def test_apply_ncbi_multilabel_pmid_filter_reports_removed_counts():
    raw_df = pd.DataFrame(
        {
            "PMID": ["1", "1", "2"],
            "label": ["A", "B", "C"],
        }
    )

    filtered, metadata = apply_ncbi_multilabel_pmid_filter(raw_df)

    assert filtered["PMID"].tolist() == ["2"]
    assert metadata == {
        "n_removed_by_multilabel_filter": 2,
        "n_removed_multilabel_pmids": 1,
    }


def test_ncbi_label_shuffle_preserves_content_and_distribution():
    ds = pd.DataFrame(
        {
            "sentence1": [f"text {i}" for i in range(8)],
            "label": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "AIFI_L2": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "type": ["title"] * 8,
        }
    )

    shuffled, metadata = shuffle_ncbi_labels(
        ds,
        seed=7,
        original_column="original_label",
    )

    assert "original_label" in shuffled.columns
    assert shuffled["sentence1"].tolist() == ds["sentence1"].tolist()
    assert shuffled["AIFI_L2"].tolist() == ds["AIFI_L2"].tolist()
    assert shuffled["label"].value_counts().to_dict() == ds["label"].value_counts().to_dict()
    assert (shuffled["label"] != shuffled["original_label"]).any()
    assert metadata["n_rows"] == len(ds)
    assert metadata["n_changed_labels"] > 0


def test_ncbi_artifacts_include_counts_and_shuffle_reports(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw_articles.csv"
    _raw_articles().to_csv(raw_path, index=False)
    monkeypatch.setattr(
        "alias.data.ncbi.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    cfg = DataNCBIConfig(
        email="test@example.com",
        dataset_id="N3_ncbi_literature_shuffled_labels",
        output_dir=tmp_path / "N3_ncbi_literature_shuffled_labels",
        save_artifacts=True,
        annotation_column="AIFI_L2",
        raw_articles_path=raw_path,
        fetch_if_missing=False,
        shuffle_labels=True,
        label_shuffle_seed=11,
    )

    dataset_dict, _ = build_datasets(None, datasets=["ncbi"], ncbi_config=cfg)
    train_df = dataset_dict["ncbi"]["data"].to_pandas()

    output_dir = tmp_path / "N3_ncbi_literature_shuffled_labels"
    assert (output_dir / "datasets" / "ncbi_data").exists()
    assert (output_dir / "reports" / "ncbi_train_head.csv").exists()
    assert (output_dir / "reports" / "article_counts_by_celltype.csv").exists()
    assert (output_dir / "reports" / "article_counts_by_celltype.pdf").exists()
    assert (output_dir / "reports" / "article_collection_summary.csv").exists()
    assert (output_dir / "reports" / "label_shuffle_report.csv").exists()
    assert (output_dir / "reports" / "label_shuffle_confusion.pdf").exists()
    assert (output_dir / "metadata" / "generation_metadata.json").exists()
    assert "original_label" in train_df.columns

    with open(output_dir / "metadata" / "generation_metadata.json") as handle:
        metadata = json.load(handle)

    assert metadata["dataset_id"] == "N3_ncbi_literature_shuffled_labels"
    assert metadata["shuffle_labels"] is True
    assert metadata["label_shuffle_seed"] == 11


def test_collect_cl_terms_writes_definition_and_description_sentence_rows(tmp_path):
    description_path = tmp_path / "descriptions.csv"
    raw_cl_path = tmp_path / "raw_cl.csv"
    pd.DataFrame(
        [
            {
                "canonical_label": "Treg",
                "hiai_label": "Treg",
                "cell_ontology_id": "CL:0000815",
                "cell_ontology_label": "regulatory T cell",
                "definition": "A regulatory T cell definition.",
                "description": (
                    "Treg cells suppress immune responses. "
                    "They can express FOXP3 and IL2RA."
                ),
            }
        ]
    ).to_csv(description_path, index=False)

    df = collect_cl_terms(
        DataCLConfig(
            description_path=description_path,
            raw_cl_path=raw_cl_path,
            labels=["Treg"],
            annotation_column="AIFI_L2",
            split_descriptions=True,
            min_sentence_words=3,
            marker_map={"Treg": {"positive": ["FOXP3", "IL2RA"]}},
        )
    )

    assert raw_cl_path.exists()
    assert df["label"].tolist() == ["Treg", "Treg", "Treg", "Treg"]
    assert df["type"].tolist() == [
        "cl_definition",
        "cl_description_sentence",
        "cl_description_sentence",
        "sctype_positive_marker_genes",
    ]
    assert set(df["source"]) == {"CL"}
    assert set(df["AIFI_L2"]) == {"Treg"}
    marker_row = df[df["type"] == "sctype_positive_marker_genes"].iloc[0]
    assert marker_row["positive_markers"] == "FOXP3;IL2RA"
    assert marker_row["sentence1"] == "Treg positive marker genes include FOXP3 and IL2RA."


def test_ncbi_generation_can_append_cl_rows(tmp_path, monkeypatch):
    raw_articles_path = tmp_path / "raw_articles.csv"
    raw_cl_path = tmp_path / "raw_cl.csv"
    description_path = tmp_path / "descriptions.csv"
    _raw_articles().iloc[:1].to_csv(raw_articles_path, index=False)
    pd.DataFrame(
        [
            {
                "canonical_label": "Treg",
                "hiai_label": "Treg",
                "cell_ontology_id": "CL:0000815",
                "cell_ontology_label": "regulatory T cell",
                "definition": "A regulatory T cell definition.",
                "description": "Treg cells suppress immune responses.",
            }
        ]
    ).to_csv(description_path, index=False)
    monkeypatch.setattr(
        "alias.data.ncbi.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    cfg = DataNCBIConfig(
        email="test@example.com",
        annotation_column="AIFI_L2",
        raw_articles_path=raw_articles_path,
        fetch_if_missing=False,
        collect_cl_terms=True,
        cl_description_path=description_path,
        cl_raw_path=raw_cl_path,
        cl_marker_map={"Treg": {"positive": ["FOXP3", "IL2RA"]}},
    )

    dataset_dict, _ = build_datasets(None, datasets=["ncbi"], ncbi_config=cfg)
    train_df = dataset_dict["ncbi"]["data"].to_pandas()

    assert raw_cl_path.exists()
    assert "CL" in set(train_df["source"])
    assert "NCBI" in set(train_df["source"])
    assert "A regulatory T cell definition." in set(train_df["sentence1"])
    assert "sctype_positive_marker_genes" in set(train_df["type"])


def test_ncbi_generation_can_infuse_cl_rows_as_title_and_abstract(tmp_path, monkeypatch):
    raw_articles_path = tmp_path / "raw_articles.csv"
    raw_cl_path = tmp_path / "raw_cl.csv"
    description_path = tmp_path / "descriptions.csv"
    _raw_articles().iloc[:1].to_csv(raw_articles_path, index=False)
    pd.DataFrame(
        [
            {
                "canonical_label": "Treg",
                "hiai_label": "Treg",
                "cell_ontology_id": "CL:0000815",
                "cell_ontology_label": "regulatory T cell",
                "definition": "A regulatory T cell definition.",
                "description": "",
            }
        ]
    ).to_csv(description_path, index=False)
    monkeypatch.setattr(
        "alias.data.ncbi.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    cfg = DataNCBIConfig(
        email="test@example.com",
        annotation_column="AIFI_L2",
        raw_articles_path=raw_articles_path,
        fetch_if_missing=False,
        collect_cl_terms=True,
        cl_description_path=description_path,
        cl_raw_path=raw_cl_path,
        cl_marker_map={},
        cl_infusion_mode="title_abstract",
    )

    dataset_dict, _ = build_datasets(None, datasets=["ncbi"], ncbi_config=cfg)
    train_df = dataset_dict["ncbi"]["data"].to_pandas()
    raw_cl_df = pd.read_csv(raw_cl_path)
    infused = train_df[
        (train_df["source"] == "CL")
        & (train_df["sentence1"] == "A regulatory T cell definition.")
    ]

    assert raw_cl_df["type"].tolist() == ["cl_definition"]
    assert sorted(infused["type"].tolist()) == ["abstract", "title"]
    assert set(infused["cl_original_type"]) == {"cl_definition"}
