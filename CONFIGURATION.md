# Configuration Guide

## Hugging Face Token Setup

This project uses environment variables for managing API keys and tokens, following industry best practices.

### Get Your Token

1. Go to https://huggingface.co/settings/tokens
2. Create a token with read/write permissions
3. You can use the same token for both DOWNLOAD and UPLOAD

### Setup Methods

Choose the method that works best for your use case:

#### Method 1: `.env` File (Recommended for Development)

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your tokens to `.env`:**
   ```bash
   HF_TOKEN_DOWNLOAD=hf_xxxxxxxxxxxxx
   HF_TOKEN_UPLOAD=hf_xxxxxxxxxxxxx
   ```

#### Method 2: Environment Variables (Recommended for Production)

**Shell export (current session):**
```bash
export HF_TOKEN_DOWNLOAD="hf_xxxxxxxxxxxxx"
export HF_TOKEN_UPLOAD="hf_xxxxxxxxxxxxx"
python my_script.py
```

**Persistent (add to `~/.bashrc` or `~/.zshrc`):**
```bash
echo 'export HF_TOKEN_DOWNLOAD="hf_xxxxxxxxxxxxx"' >> ~/.zshrc
echo 'export HF_TOKEN_UPLOAD="hf_xxxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

**Docker/Container:**
```dockerfile
ENV HF_TOKEN_DOWNLOAD=hf_xxxxxxxxxxxxx
ENV HF_TOKEN_UPLOAD=hf_xxxxxxxxxxxxx
```

**Note:** Environment variables take precedence over `.env` files.

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
- Make sure `.env` is in your current working directory or a parent directory

**Error: ".env file not found"**
- Create `.env` in your project directory (where you run your scripts)
- Or create `~/.env` in your home directory for system-wide config
- The package searches: current directory → parent dirs → ~/.env

**For pip installed package:**
- Put `.env` in the directory where you run your Python scripts
- Not in the installed package location!
- Example: If you run `python my_script.py` from `/my/project`, put `.env` in `/my/project/`

