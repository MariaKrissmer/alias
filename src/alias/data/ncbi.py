from dataclasses import dataclass, asdict
from datasets import Dataset
import time, random
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from Bio import Entrez
from transformers import AutoTokenizer

@dataclass
class DataNCBIConfig:
    email: str
    organism: str = "homo sapiens"
    max_articles: int = 100
    batch_size: int = 250
    max_retries: int = 3
    model: str = "neuml/pubmedbert-base-embeddings"
    max_tokens: int = 512
    overlap: int = 20
    diseases: Optional[List[str]] = None
    semantic: bool = False
    celltypes_from_adata: bool = True
    celltypes_list: Optional[List[str]] = None
    annotation_column: str = "celltype"
    test_split: float = None

def fetch_articles(celltype: str, cfg: Dict) -> pd.DataFrame:
    """Fetch PubMed titles and abstracts for a given cell type."""
    Entrez.email = cfg["email"]
    base_query = f"({cfg['organism']}[Mesh]) AND {celltype}"

    queries, query_to_label, query_to_disease = [], {}, {}

    if cfg.get("diseases"):
        for disease in cfg["diseases"]:
            q_and = f"({base_query} AND {disease})"
            q_not = f"({base_query} NOT {disease})"
            queries += [q_and, q_not]
            query_to_label[q_and], query_to_label[q_not] = f"{celltype}_{disease}", celltype
            query_to_disease[q_and], query_to_disease[q_not] = disease, ""
    else:
        queries = [base_query]
        query_to_label[base_query] = celltype
        query_to_disease[base_query] = ""

    all_abstracts = {}

    for query in queries:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=cfg["max_articles"])
        record = Entrez.read(handle)
        handle.close()
        pmids = record.get("IdList", [])
        print(f"{len(pmids)} PMIDs for query: {query}")

        for i in tqdm(range(0, len(pmids), cfg["batch_size"])):
            batch = pmids[i : i + cfg["batch_size"]]
            for attempt in range(cfg["max_retries"]):
                try:
                    handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml", retmode="text")
                    records = Entrez.read(handle)
                    handle.close()

                    for article in records.get("PubmedArticle", []):
                        pmid = article["MedlineCitation"]["PMID"]
                        if pmid in all_abstracts:
                            continue
                        art = article["MedlineCitation"]["Article"]
                        title = art.get("ArticleTitle", "No Title Available")
                        abstract = art.get("Abstract", {}).get("AbstractText", ["No Abstract"])[0]
                        all_abstracts[pmid] = {
                            "PMID": pmid,
                            "Title": title,
                            "Abstract": abstract,
                            "Query": query,
                            "label": query_to_label[query],
                            "disease": query_to_disease[query],
                        }

                    time.sleep(random.uniform(1, 3))
                    break
                except Exception as e:
                    print(f"Retry {attempt+1}/{cfg['max_retries']} for batch {i}: {e}")
                    time.sleep(2 ** attempt)

    return pd.DataFrame(list(all_abstracts.values()))

def split_text_by_tokens(tokenizer, text: str, cfg: Dict) -> Tuple[List[str], List[int]]:
    """Split text into overlapping chunks based on max_tokens and overlap."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    step = cfg["max_tokens"] - cfg["overlap"]
    chunks = [tokens[i : i + cfg["max_tokens"]] for i in range(0, len(tokens), step)]
    decoded = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
    return decoded, [len(c) for c in chunks]


def process_ncbi_df(df: pd.DataFrame, annotation_column: str, cfg: Dict) -> pd.DataFrame:
    """Convert abstracts and titles into a training DataFrame."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    records = []

    for _, row in df.iterrows():
        for field in ["Abstract", "Title"]:
            text = row.get(field)
            if not text:
                continue
            substrings, lengths = split_text_by_tokens(tokenizer, text, cfg)
            for s, l in zip(substrings, lengths):
                rec = {
                    "sentence1": s,
                    "token_length": l,
                    annotation_column: row.get(annotation_column, None),
                    "Query": row.get("Query"),
                    "type": field.lower(),
                    "label": row.get("label"),
                    "disease": row.get("disease"),
                }
                records.append(rec)

    return pd.DataFrame(records)

def gen_ncbi_dataset(
    adata=None, ncbi_config: DataNCBIConfig = None, **kwargs
) -> Tuple[Dataset, Dataset]:
    """Generate train/test HuggingFace Datasets from AnnData or config, overridable with kwargs."""
    cfg = asdict(ncbi_config)
    cfg.update(kwargs)  
    all_dfs = []
    annotation_column = cfg["annotation_column"]

    if cfg["celltypes_from_adata"] and adata is not None:
        celltypes = list(adata.obs[annotation_column].astype(str).unique())
    else:
        celltypes = cfg.get("celltypes_list") or []

    for ct in celltypes:
        df = fetch_articles(ct, cfg)
        if not df.empty:
            df[annotation_column] = ct
            all_dfs.append(df)
        else:
            print(f"No NCBI articles found for cell type '{ct}'")

    if not all_dfs:
        print("No data collected. Returning empty Datasets.")
        return Dataset.from_dict({}), Dataset.from_dict({})

    df_full = pd.concat(all_dfs, ignore_index=True)
    df_processed = process_ncbi_df(df_full, annotation_column, cfg)

    ds = Dataset.from_pandas(df_processed.reset_index())

    # Split into train/test
    if cfg['test_split'] is not None:
        ds_split = ds.train_test_split(test_size=cfg['test_split'], seed=cfg['random_state'])
        train_ds = ds_split["train"]
        test_ds = ds_split["test"]
    else:
        train_ds = ds
        test_ds = {}
    return train_ds, test_ds