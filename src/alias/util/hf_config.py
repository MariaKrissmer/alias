"""
Hugging Face configuration loader.

This module loads HF tokens from a .env file in the project root.
Copy .env.example to .env and add your actual tokens.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")
    print(f"Please copy .env.example to .env and add your Hugging Face tokens.")


class HFConfig:
    """Configuration class for Hugging Face tokens."""
    
    @property
    def HF_TOKEN_DOWNLOAD(self) -> str:
        """Token for downloading private models/datasets."""
        token = os.getenv("HF_TOKEN_DOWNLOAD")
        if not token:
            raise ValueError(
                "HF_TOKEN_DOWNLOAD not found in environment. "
                "Please set it in your .env file (see .env.example)"
            )
        return token
    
    @property
    def HF_TOKEN_UPLOAD(self) -> str:
        """Token for uploading models/datasets."""
        token = os.getenv("HF_TOKEN_UPLOAD")
        if not token:
            raise ValueError(
                "HF_TOKEN_UPLOAD not found in environment. "
                "Please set it in your .env file (see .env.example)"
            )
        return token


# Create a singleton instance
hf_config = HFConfig()

