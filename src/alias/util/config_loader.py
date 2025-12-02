"""
Configuration loader utility for converting YAML configs to dataclass objects.
"""

from pathlib import Path
from typing import Any, Type, TypeVar
from dataclasses import fields, is_dataclass
import os
import re
import yaml


T = TypeVar('T')


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR_NAME} or ${VAR_NAME:-default} in config values."""
    if isinstance(value, str):
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
        def replacer(match):
            var_name, default = match.group(1), match.group(2)
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                raise ValueError(f"Environment variable '{var_name}' is not set and no default provided.")
        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file with environment variable substitution."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return _substitute_env_vars(config)


def dataclass_from_dict(dataclass_type: Type[T], config_dict: dict[str, Any]) -> T:
    """Create a dataclass instance from a dict, using only fields defined in the dataclass."""
    if not is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type} is not a dataclass")
    field_names = {field.name for field in fields(dataclass_type)}
    field_values = {k: v for k, v in config_dict.items() if k in field_names}
    return dataclass_type(**field_values)

