"""Model endpoint registry and configuration contracts."""

from cage.models.endpoint import ModelConfig
from cage.models.registry import load_models

__all__ = ["ModelConfig", "load_models"]
