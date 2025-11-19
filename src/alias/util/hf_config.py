"""
Hugging Face configuration loader.

This module loads HF tokens from a .env file in the project root.
Copy .env.example to .env and add your actual tokens.
"""

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Try to find .env file in the following order:
# 1. Current working directory (where user runs their script)
# 2. Search up directory tree from cwd (for nested project structures)
# 3. User home directory (~/.env as fallback)

env_path = find_dotenv(usecwd=True)  # Searches from cwd upwards

if env_path:
    load_dotenv(env_path)
else:
    # Try home directory as fallback
    home_env = Path.home() / ".env"
    if home_env.exists():
        load_dotenv(home_env)
    else:
        print("Warning: .env file not found.")
        print("Searched in: current directory (and parents), and ~/.env")
        print("Please create a .env file with your Hugging Face tokens (see .env.example)")


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

