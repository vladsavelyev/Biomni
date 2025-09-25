"""
Tests for blacklist configuration module.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from mcp_biomni.server.blacklist_config import BlacklistConfig, load_blacklist_config


class TestBlacklistConfig:
    """Test the BlacklistConfig class."""

    def test_init_empty(self):
        """Test initialization with empty data."""
        config = BlacklistConfig()
        assert config.data == {}

    def test_init_with_data(self):
        """Test initialization with provided data."""
        data = {
            "tools": ["tool1", "tool2"],
            "files": ["file1.csv"],
            "capabilities": {"python_packages": ["pkg1"], "cli_tools": ["cli1"]},
            "modules": ["mod1"],
        }
        config = BlacklistConfig(data)
        assert config.data == data

    def test_is_tool_blocked(self):
        """Test tool blocking detection."""
        data = {"tools": ["blocked_tool", "another_blocked"]}
        config = BlacklistConfig(data)

        assert config.is_tool_blocked("blocked_tool") is True
        assert config.is_tool_blocked("another_blocked") is True
        assert config.is_tool_blocked("allowed_tool") is False
        assert config.is_tool_blocked("") is False

    def test_is_file_blocked(self):
        """Test file blocking detection."""
        data = {"files": ["sensitive.csv", "private.parquet"]}
        config = BlacklistConfig(data)

        assert config.is_file_blocked("sensitive.csv") is True
        assert config.is_file_blocked("private.parquet") is True
        assert config.is_file_blocked("public.csv") is False
        assert config.is_file_blocked("") is False

    def test_is_capability_blocked(self):
        """Test capability blocking detection."""
        data = {
            "capabilities": {
                "python_packages": ["dangerous_pkg", "deprecated_lib"],
                "r_packages": ["blocked_r"],
                "cli_tools": ["restricted_cli"],
            }
        }
        config = BlacklistConfig(data)

        # Test python packages
        assert config.is_capability_blocked("python_packages", "dangerous_pkg") is True
        assert config.is_capability_blocked("python_packages", "safe_pkg") is False

        # Test r packages
        assert config.is_capability_blocked("r_packages", "blocked_r") is True
        assert config.is_capability_blocked("r_packages", "safe_r") is False

        # Test cli tools
        assert config.is_capability_blocked("cli_tools", "restricted_cli") is True
        assert config.is_capability_blocked("cli_tools", "safe_cli") is False

        # Test non-existent category
        assert config.is_capability_blocked("nonexistent", "anything") is False

    def test_is_module_blocked(self):
        """Test module blocking detection."""
        data = {"modules": ["experimental", "deprecated"]}
        config = BlacklistConfig(data)

        assert config.is_module_blocked("experimental") is True
        assert config.is_module_blocked("deprecated") is True
        assert config.is_module_blocked("stable") is False
        assert config.is_module_blocked("") is False

    def test_empty_config_allows_everything(self):
        """Test that empty config allows everything."""
        config = BlacklistConfig({})

        assert config.is_tool_blocked("any_tool") is False
        assert config.is_file_blocked("any_file.csv") is False
        assert config.is_capability_blocked("python_packages", "any_pkg") is False
        assert config.is_module_blocked("any_module") is False


class TestLoadBlacklistConfig:
    """Test the load_blacklist_config function."""

    def test_load_from_valid_yaml_file(self):
        """Test loading from a valid YAML file."""
        yaml_content = {
            "tools": ["tool1", "tool2"],
            "files": ["file1.csv"],
            "capabilities": {"python_packages": ["pkg1"], "cli_tools": ["cli1"]},
            "modules": ["mod1"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            blacklist_file = project_root / "blacklist.yaml"

            with open(blacklist_file, "w") as f:
                yaml.dump(yaml_content, f)

            config = load_blacklist_config(project_root)

            assert config.is_tool_blocked("tool1") is True
            assert config.is_file_blocked("file1.csv") is True
            assert config.is_capability_blocked("python_packages", "pkg1") is True
            assert config.is_module_blocked("mod1") is True

    def test_load_nonexistent_file(self):
        """Test loading when blacklist.yaml doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_blacklist_config(temp_dir)

            # Should return empty config that allows everything
            assert config.is_tool_blocked("any_tool") is False
            assert config.is_file_blocked("any_file") is False

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            blacklist_file = project_root / "blacklist.yaml"

            with open(blacklist_file, "w") as f:
                f.write("invalid: yaml: content: [\n")

            config = load_blacklist_config(project_root)

            # Should return empty config on error
            assert config.is_tool_blocked("any_tool") is False

    def test_load_empty_yaml(self):
        """Test loading empty YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            blacklist_file = project_root / "blacklist.yaml"

            with open(blacklist_file, "w") as f:
                f.write("")

            config = load_blacklist_config(project_root)

            # Should work with empty content
            assert config.is_tool_blocked("any_tool") is False


if __name__ == "__main__":
    pytest.main([__file__])
