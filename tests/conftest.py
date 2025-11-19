"""
Pytest configuration file.

This file is automatically loaded by pytest and can be used to:
- Define fixtures available to all tests
- Configure test environment
- Add custom pytest hooks
"""

import sys
from pathlib import Path

# Add src directory to Python path for all tests
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

