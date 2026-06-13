"""Configuration input parsing: YAML / project files → the experiment model.

This package owns the word "config". It presents the repo-local configuration
API as a stable facade (``from cage.config import find_repo_root`` etc.) so
benchmarks and the web/CLI never depend on the internal module split:

  repo.py        repo-local settings (``./config/cage.yml``, model registry
                 location, web-inspector auth). Re-exported here — the public,
                 cross-layer surface.
  experiment.py  the legacy per-run experiment configuration loader. Internal
                 framework plumbing (slated for removal in Phase B); imported
                 explicitly as ``cage.config.experiment``.
"""

from cage.config.repo import (
    CageConfig,
    WebInspectorAuthConfig,
    WebInspectorBoardConfig,
    WebInspectorConfig,
    WebInspectorUIConfig,
    expand_env_refs,
    find_repo_root,
    load_repo_config,
    resolve_models_file,
)

__all__ = [
    "CageConfig",
    "WebInspectorAuthConfig",
    "WebInspectorBoardConfig",
    "WebInspectorConfig",
    "WebInspectorUIConfig",
    "expand_env_refs",
    "find_repo_root",
    "load_repo_config",
    "resolve_models_file",
]
