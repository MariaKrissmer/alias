from dataclasses import dataclass, asdict
from typing import Dict, Optional
from datasets import Dataset
import scanpy as sc

from data.scrna import DatascRNAConfig, gen_scrna_dataset
from data.ncbi import DataNCBIConfig, gen_ncbi_dataset
from data.triplet_generation import TripletGenerationConfig, generate_triplets


def build_datasets(
    adata,
    datasets, 
    scrna_config: Optional[DatascRNAConfig] = None,
    ncbi_config: Optional[DataNCBIConfig] = None, 
    **kwargs
) -> Dict[str, Dict[str, Dataset]]:
    """
    Build structured datasets (train/test) for all supported modalities (e.g. scRNA, NCBI).

    Args:
        adata: AnnData object loaded by the user.
        scrna_config: configuration for scRNA dataset creation.
        ncbi_config: configuration for NCBI dataset creation (optional, future).

    Returns:
        dict: structured dictionary of datasets like:
            {
                "scrna": {"train": ..., "test": ...},
                "ncbi": {"train": ..., "test": ...}
            }
    """
    dataset_dict: Dict[str, Dict[str, Dataset]] = {}
    adata_test = None

    # scRNA dataset
    if 'scrna' in datasets:
        scrna_train, scrna_test, adata_test = gen_scrna_dataset(adata, scrna_config, **kwargs)
        dataset_dict["scrna"] = {
            "data": scrna_train,
            "test": scrna_test
        }


    # NCBI dataset 
    if 'ncbi' in datasets:
        ncbi_train, ncbi_test = gen_ncbi_dataset(adata, ncbi_config)
        dataset_dict["ncbi"] = {
            "data": ncbi_train,
            "test": ncbi_test
        }

    return dataset_dict, adata_test

def build_triplets(
    dataset_dict,
    triplet_config: Optional[TripletGenerationConfig] = None, 
    **kwargs
) -> Dict[str, Dict[str, Dataset]]:
    
    train_data_dict = {}

    for dataset_name, splits in dataset_dict.items():
        if "data" not in splits:
            continue

        print(f"Building triplets for {dataset_name}...")
        if dataset_name not in train_data_dict:
            train_data_dict[dataset_name] = {}

        train_data = splits["data"]

        # Determine naming suffix based on config
        suffix_parts = [triplet_config.loss]
        if triplet_config.hard_negative_mining:
            suffix_parts.append("hnm")
        elif triplet_config.random_negative_mining:
            suffix_parts.append("rnm")
        suffix = "_".join(suffix_parts)

        # Call your triplet generator
        train_ds, eval_ds = generate_triplets(
            train_data=train_data, 
            type=dataset_name,
            triplet_config=triplet_config, 
            **kwargs
        )

        train_data_dict[dataset_name].update({
            f"train_{suffix}": train_ds,
            f"eval_{suffix}": eval_ds
        })

    return train_data_dict
