"""
Simple Blacklist Configuration for MCP Biomni Server
Loads blacklist from YAML file and provides filtering functions.
"""

from pathlib import Path
from typing import Any

import yaml


class BlacklistConfig:
    """Simple blacklist configuration manager."""

    def __init__(self, blacklist_data: dict[str, Any] | None = None):
        self.data = blacklist_data or {}

    def is_tool_blocked(self, tool_name: str) -> bool:
        """Check if a tool is blacklisted."""
        return tool_name in self.data.get("tools", [])

    def is_file_blocked(self, file_path: str) -> bool:
        """Check if a file is blacklisted."""
        return file_path in self.data.get("files", [])

    def is_capability_blocked(self, category: str, name: str) -> bool:
        """Check if a capability (package/tool) is blacklisted."""
        capabilities = self.data.get("capabilities", {})
        return name in capabilities.get(category, [])

    def is_module_blocked(self, module_name: str) -> bool:
        """Check if a module is blacklisted."""
        return module_name in self.data.get("modules", [])


def load_blacklist_config(project_root: str | Path | None = None) -> BlacklistConfig:
    """Load blacklist configuration from project's blacklist.yaml file."""
    if project_root is None:
        # Try to find project root from current file location
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # Go up to project root

    project_root = Path(project_root)
    blacklist_file = project_root / "blacklist.yaml"

    if not blacklist_file.exists():
        print(f"No blacklist.yaml found at {blacklist_file}, using empty blacklist")
        return BlacklistConfig({})

    try:
        with open(blacklist_file) as f:
            blacklist_data = yaml.safe_load(f) or {}
        print(f"Loaded blacklist config from: {blacklist_file}")
        return BlacklistConfig(blacklist_data)
    except Exception as e:
        print(f"Warning: Failed to load blacklist config from {blacklist_file}: {e}")
        return BlacklistConfig({})
