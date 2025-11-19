# GitHub Actions Workflows

## test.yml

This workflow runs the test suite across multiple Python versions to ensure compatibility.

### What it does:

1. **Triggers on:**
   - Push to `main`, `master`, or `develop` branches
   - Pull requests to those branches

2. **Tests Python versions:**
   - Python 3.11
   - Python 3.12
   - Python 3.13

3. **Steps:**
   - Checks out the code
   - Sets up the specified Python version
   - Installs `uv` (fast Python package manager)
   - Installs dependencies with `uv sync --extra test`
   - Runs import tests with pytest
   - Verifies package can be imported successfully

### Viewing results:

Go to the "Actions" tab in your GitHub repository to see test results.

### Adding the badge to README:

Update the badge URL in README.md:
```markdown
[![Tests](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/test.yml/badge.svg)](https://github.com/MariaKrissmer/alias/actions/workflows/test.yml)

