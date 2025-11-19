# Configuration Guide

## Hugging Face Token Setup

This project uses environment variables for managing API keys and tokens, following industry best practices.

### Quick Start

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your tokens to `.env`:**
   ```bash
   HF_TOKEN_DOWNLOAD=hf_xxxxxxxxxxxxx
   HF_TOKEN_UPLOAD=hf_xxxxxxxxxxxxx
   ```

3. **Get tokens from Hugging Face:**
   - Go to https://huggingface.co/settings/tokens
   - Create a token with read/write permissions
   - You can use the same token for both DOWNLOAD and UPLOAD

### Usage in Code

Import and use the configuration in your code:

```python
from alias.util.hf_config import hf_config

# Access tokens
token = hf_config.HF_TOKEN_DOWNLOAD
upload_token = hf_config.HF_TOKEN_UPLOAD
```

### Troubleshooting

**Error: "HF_TOKEN_DOWNLOAD not found in environment"**
- Make sure you created `.env` from `.env.example`
- Make sure you added actual tokens (not the placeholder text)
- Make sure `.env` is in the project root (same directory as `pyproject.toml`)

**Error: ".env file not found"**
- Run `cp .env.example .env` from the project root
- Or manually create a `.env` file with the required variables

