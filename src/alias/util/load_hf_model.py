from huggingface_hub import HfApi, ModelInfo, DatasetInfo
from huggingface_hub.utils import HfHubHTTPError
from sentence_transformers import SentenceTransformer
from datetime import datetime
from datasets import DatasetDict, load_dataset
import os


def is_model_private(model_id: str, token: str = None) -> bool:
    """Check if a model is private. Returns False if token is not provided."""
    if not token:
        return False
    api = HfApi()
    try:
        model_info: ModelInfo = api.model_info(model_id, token=token)
        return model_info.private
    except HfHubHTTPError as e:
        print(f"Error accessing model '{model_id}': {e}")
        return False  # Default to public if it fails (or handle differently)

def load_model(model_id: str):
    """Load a model from HuggingFace, using token only if available and needed."""
    # Only import and use token if available
    token = os.getenv("HF_TOKEN_DOWNLOAD")
    
    if token and is_model_private(model_id, token):
        print(f"Model '{model_id}' is private. Using token.")
        model = SentenceTransformer(model_id, token=token)
    else:
        print(f"Model '{model_id}' is public. No token needed.")
        model = SentenceTransformer(model_id)
    
    return model

def upload_dataset_to_hf(
    dataset_dict,
    dataset_name, 
    name: str = None,
    private: bool = True,
    token: str = None,
    org: str = None,
    **kwargs
):
    """
    Uploads a Hugging Face `Dataset` or `DatasetDict` to the Hub.
    
    Args:
        dataset_dict: Either a `Dataset`, `DatasetDict`, or dict of Datasets.
        name: Optional repo name. Defaults to current date (YYYYMMDD_HHMM).
        private: Whether to make the repo private (default: True).
        token: HF access token (required for upload).
        org: Optional organization name to upload under.
    """
    api = HfApi()
    if not token:
        token = os.getenv("HF_TOKEN_UPLOAD")
    if not token:
        raise ValueError(
            "No Hugging Face token provided. "
            "Set HF_TOKEN_UPLOAD in your .env file or pass token parameter."
        )

    if name is None:
        repo_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    else:
        repo_name = f"{dataset_name}_{name}"

    user = api.whoami(token=token)["name"]
    repo_id = f"{org or user}/{repo_name}"

    print(f"Creating repository '{repo_id}' (private={private})...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

    if isinstance(dataset_dict, dict) and not isinstance(dataset_dict, DatasetDict):
        dataset_dict = DatasetDict(dataset_dict)

    print(f"Uploading dataset to Hugging Face Hub → {repo_id}")
    dataset_dict.push_to_hub(repo_id, token=token, **kwargs)

    print(f"Successfully uploaded: https://huggingface.co/datasets/{repo_id}")
    return repo_id

def load_hf_dataset(dataset_name: str, hf_token: str = None, **kwargs):
    """
    Loads a dataset from Hugging Face, printing its privacy status and using token if required.

    Args:
        dataset_name (str): e.g., 'username/dataset_name'
        hf_token (str): Hugging Face token for private datasets (optional).
        **kwargs: Any additional arguments to pass to load_dataset.

    Returns:
        A Dataset or DatasetDict object.
    """
    api = HfApi()
    
    # Get token from env if not provided
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN_DOWNLOAD")
    
    if not hf_token:
        raise ValueError(
            "No Hugging Face token available for loading datasets. "
            "Set HF_TOKEN_DOWNLOAD in your .env file or pass hf_token parameter."
        )

    # Step 1: Check privacy status
    try:
        dataset_info: DatasetInfo = api.dataset_info(dataset_name, token=hf_token)
        is_private = dataset_info.private
        print(f"Dataset '{dataset_name}' is {'PRIVATE' if is_private else 'PUBLIC'}")
    except HfHubHTTPError as e:
        print(f"Could not retrieve dataset info for '{dataset_name}': {e}")
        is_private = False  # Default to public or handle this differently

    # Step 2: Load dataset
    try:
        dataset = load_dataset(
            dataset_name,
            token=hf_token if is_private else False,
            **kwargs
        )
        print(f"Successfully loaded the dataset: {dataset_name}")
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")