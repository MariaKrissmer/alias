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


def substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in configuration values.
    
    Supports syntax: ${VAR_NAME} or ${VAR_NAME:-default_value}
    
    Parameters
    ----------
    value : Any
        Configuration value (can be string, dict, list, etc.)
        
    Returns
    -------
    Any
        Value with environment variables substituted.
        
    Examples
    --------
    >>> os.environ['TEST_VAR'] = 'hello'
    >>> substitute_env_vars('${TEST_VAR}')
    'hello'
    >>> substitute_env_vars('${MISSING_VAR:-default}')
    'default'
    >>> substitute_env_vars({'key': '${TEST_VAR}'})
    {'key': 'hello'}
    """
    if isinstance(value, str):
        # Pattern matches ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)
            
            env_value = os.getenv(var_name)
            
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ValueError(
                    f"Environment variable '{var_name}' is not set and no default provided. "
                    f"Please set it in your .env file or environment."
                )
        
        return re.sub(pattern, replacer, value)
    
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    
    else:
        return value


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.
        
    Returns
    -------
    dict[str, Any]
        dictionary containing the configuration.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    config = substitute_env_vars(config)
    
    return config


def dataclass_from_dict(dataclass_type: Type[T], config_dict: dict[str, Any]) -> T:
    """
    Create a dataclass instance from a dictionary, handling optional fields.
    
    Only includes fields that exist in the dataclass definition. Fields with
    defaults will use their defaults if not present in the config_dict.
    
    Parameters
    ----------
    dataclass_type : Type[T]
        The dataclass type to instantiate.
    config_dict : dict[str, Any]
        dictionary containing configuration values.
        
    Returns
    -------
    T
        Instance of the dataclass.
    """
    if not is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type} is not a dataclass")
    
    # Only include fields that exist in the dataclass definition
    # This filters out extra keys and lets the dataclass handle defaults
    field_names = {field.name for field in fields(dataclass_type)}
    field_values = {k: v for k, v in config_dict.items() if k in field_names}
    
    return dataclass_type(**field_values)


def validate_required_fields(config: dict[str, Any], section: str, required_fields: list) -> None:
    """
    Validate that required fields are present in the config.
    
    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    section : str
        Section name for error messages.
    required_fields : list
        List of required field names.
        
    Raises
    ------
    ValueError
        If any required field is missing or None.
    """
    if section not in config:
        raise ValueError(f"Missing required config section: {section}")
    
    for field in required_fields:
        if field not in config[section] or config[section][field] is None:
            raise ValueError(f"Missing required field in {section}: {field}")


def load_dataset_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load dataset generation configuration.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the user's configuration file.
        
    Returns
    -------
    dict[str, Any]
        Configuration dictionary.
    """
    config = load_yaml_config(config_path)
    
    # Validate required fields
    validate_required_fields(config, "general", ["data_path", "input_adata_path"])
    
    # Validate NCBI email if NCBI is enabled
    if config.get("datasets", {}).get("generate_ncbi", False):
        validate_required_fields(config, "ncbi_config", ["email"])
    
    return config

