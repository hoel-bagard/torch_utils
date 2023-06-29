from dataclasses import asdict, fields
from typing import Any


def get_config_as_dict(config: object) -> dict[str, object]:
    """Take a config object (class with class attributes) and return it as dictionnary.

    Args:
        config: A config is just a python class with some class variables.

    Returns:
        config_dict: A dictionnary mapping a (lowercase) class variable name to its value.
    """
    config_attribute_dict = vars(config)

    config_dict: dict[str, object] = {}
    for key, value in config_attribute_dict.items():
        if not key.startswith("__") and key[0].isupper():
            config_dict[key.lower()] = value

    return config_dict


def get_dataclass_as_dict(config: object, *, lower_case: bool = True) -> dict[str, Any]:
    """Take a dataclass instance and returns it as a dictionnary.

    Args:
        config: The dataclass instance.
        lower_case: If true then the field names will all be lowercase.

    Returns:
        A dictionnary where the keys are the field names and the values are the config values.
    """
    if lower_case:
        return {
            field.name.lower(): getattr(config, field.name)
            for field in fields(config)}   # type: ignore[reportGeneralTypeIssues]

    return asdict(config)  # type: ignore[reportGeneralTypeIssues]
