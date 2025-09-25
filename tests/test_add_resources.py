"""Tests for add_resources.py server logic functions."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mcp_biomni.server.add_resources import (
    FileMeta,
    build_file_index,
    get_available_resources,
    guess_mime,
    looks_textual,
    parse_library_content,
    read_bytes,
    sanitize_tool_name,
)


class TestSanitizeToolName:
    """Tests for sanitize_tool_name function."""

    def test_simple_name(self):
        """Test sanitizing a simple filename."""
        assert sanitize_tool_name("gene_info") == "file_gene_info"

    def test_name_with_extension(self):
        """Test sanitizing filename with extension."""
        assert sanitize_tool_name("gene_info.parquet") == "file_gene_info"

    def test_name_with_special_chars(self):
        """Test sanitizing filename with special characters."""
        assert (
            sanitize_tool_name("broad-repurposing_hub.csv")
            == "file_broad_repurposing_hub"
        )

    def test_name_starting_with_digit(self):
        """Test sanitizing filename starting with digit."""
        assert sanitize_tool_name("123_data.txt") == "file_f_123_data"

    def test_name_with_only_special_chars(self):
        """Test sanitizing filename with only special characters."""
        assert sanitize_tool_name("---...___") == "file______"


class TestGuessMime:
    """Tests for guess_mime function."""

    def test_parquet_file(self):
        """Test MIME type detection for Parquet files."""
        path = Path("test.parquet")
        assert guess_mime(path) == "application/vnd.apache.parquet"

    def test_pickle_file(self):
        """Test MIME type detection for pickle files."""
        path = Path("test.pkl")
        assert guess_mime(path) == "application/octet-stream"

    def test_tsv_file(self):
        """Test MIME type detection for TSV files."""
        path = Path("test.tsv")
        assert guess_mime(path) == "text/tab-separated-values"

    def test_obo_file(self):
        """Test MIME type detection for OBO files."""
        path = Path("test.obo")
        assert guess_mime(path) == "text/plain"

    def test_csv_file(self):
        """Test MIME type detection for CSV files (standard detection)."""
        path = Path("test.csv")
        assert guess_mime(path) == "text/csv"

    def test_unknown_file(self):
        """Test MIME type detection for unknown extensions."""
        path = Path("test.unknown")
        assert guess_mime(path) == "application/octet-stream"


class TestLooksTextual:
    """Tests for looks_textual function."""

    def test_text_mime_types(self):
        """Test that text/* MIME types are recognized as textual."""
        assert looks_textual("text/plain") is True
        assert looks_textual("text/csv") is True
        assert looks_textual("text/html") is True

    def test_special_textual_types(self):
        """Test that special textual types are recognized."""
        assert looks_textual("application/json") is True
        assert looks_textual("application/xml") is True
        assert looks_textual("text/tab-separated-values") is True

    def test_binary_mime_types(self):
        """Test that binary MIME types are not recognized as textual."""
        assert looks_textual("application/vnd.apache.parquet") is False
        assert looks_textual("application/octet-stream") is False
        assert looks_textual("image/png") is False


class TestReadBytes:
    """Tests for read_bytes async function."""

    async def test_read_full_file(self):
        """Test reading a complete file."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
            test_content = b"Hello, World!"
            tmp.write(test_content)
            tmp.flush()

            try:
                result = await read_bytes(Path(tmp.name))
                assert result == test_content
            finally:
                os.unlink(tmp.name)

    async def test_read_file_with_limit(self):
        """Test reading file with byte limit."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
            test_content = b"Hello, World! This is a longer message."
            tmp.write(test_content)
            tmp.flush()

            try:
                result = await read_bytes(Path(tmp.name), max_bytes=5)
                assert result == b"Hello"
            finally:
                os.unlink(tmp.name)


class TestFileMeta:
    """Tests for FileMeta dataclass."""

    def test_file_meta_creation(self):
        """Test creating a FileMeta instance."""
        path = Path("/test/path.txt")
        desc = "Test description"
        mime = "text/plain"

        meta = FileMeta(path=path, desc=desc, mime=mime)

        assert meta.path == path
        assert meta.desc == desc
        assert meta.mime == mime


class TestBuildFileIndex:
    """Tests for build_file_index function."""

    @patch(
        "mcp_biomni.server.add_resources.data_lake_dict",
        {
            "test_file.parquet": "Test file description",
            "missing_file.csv": "Missing file description",
        },
    )
    def test_build_file_index_with_existing_files(self):
        """Test building file index with existing files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            data_lake_path = base_path / "biomni_data" / "data_lake"
            data_lake_path.mkdir(parents=True)

            # Create a test file
            test_file = data_lake_path / "test_file.parquet"
            test_file.write_bytes(b"test data")

            index = build_file_index(base_path)

            assert len(index) == 1
            assert "test_file.parquet" in index
            assert index["test_file.parquet"].desc == "Test file description"
            assert index["test_file.parquet"].mime == "application/vnd.apache.parquet"

    def test_build_file_index_nonexistent_base_path(self):
        """Test building file index with nonexistent base path."""
        index = build_file_index(Path("/nonexistent/path"))
        assert index == {}

    def test_build_file_index_none_base_path(self):
        """Test building file index with None base path."""
        index = build_file_index(None)
        assert index == {}


class TestParseLibraryContent:
    """Tests for parse_library_content function."""

    @patch(
        "mcp_biomni.server.add_resources.library_content_dict",
        {
            "biopython": "[Python Package] A set of tools for biological computation.",
            "ggplot2": "[R Package] A system for declaratively creating graphics. Use with subprocess.",
            "samtools": "[CLI Tool] A suite of programs for interacting with sequencing data.",
            "mystery_tool": "A mysterious tool without explicit type markers.",
        },
    )
    def test_parse_library_content_all_types(self):
        """Test parsing library content with all package types."""
        result = parse_library_content()

        # Check structure
        assert "python_packages" in result
        assert "r_packages" in result
        assert "cli_tools" in result

        # Check Python package
        assert "biopython" in result["python_packages"]
        bio_pkg = result["python_packages"]["biopython"]
        assert bio_pkg["import_name"] == "Bio"  # Special mapping
        assert bio_pkg["package_name"] == "biopython"
        assert bio_pkg["import_example"] == "import Bio"
        assert bio_pkg["type"] == "python_package"
        assert "biological computation" in bio_pkg["description"]

        # Check R package
        assert "ggplot2" in result["r_packages"]
        r_pkg = result["r_packages"]["ggplot2"]
        assert r_pkg["package_name"] == "ggplot2"
        assert (
            "subprocess.run(['Rscript', '-e', 'library(ggplot2)"
            in r_pkg["subprocess_example"]
        )
        assert r_pkg["type"] == "r_package"

        # Check CLI tool
        assert "samtools" in result["cli_tools"]
        cli_tool = result["cli_tools"]["samtools"]
        assert cli_tool["executable"] == "samtools"
        assert (
            "subprocess.run(['samtools', 'arg1', 'arg2'])"
            in cli_tool["subprocess_example"]
        )
        assert cli_tool["type"] == "cli_tool"

        # Check mystery tool (should default to CLI)
        assert "mystery_tool" in result["cli_tools"]
        mystery = result["cli_tools"]["mystery_tool"]
        assert mystery["type"] == "cli_tool"

    @patch(
        "mcp_biomni.server.add_resources.library_content_dict",
        {
            "scikit-learn": "[Python Package] Machine learning library.",
            "opencv-python": "[Python Package] Computer vision library.",
        },
    )
    def test_parse_library_content_import_name_mapping(self):
        """Test that special import name mappings work correctly."""
        result = parse_library_content()

        # Test scikit-learn -> sklearn mapping
        sklearn_pkg = result["python_packages"]["scikit-learn"]
        assert sklearn_pkg["import_name"] == "sklearn"
        assert sklearn_pkg["import_example"] == "import sklearn"

        # Test opencv-python -> cv2 mapping
        cv_pkg = result["python_packages"]["opencv-python"]
        assert cv_pkg["import_name"] == "cv2"
        assert cv_pkg["import_example"] == "import cv2"


class TestGetAvailableResources:
    """Tests for get_available_resources function."""

    @patch("mcp_biomni.server.add_resources.build_file_index")
    @patch.dict(os.environ, {"BIOMNI_DATA_PATH": "/test/path"})
    def test_get_available_resources_with_env_var(self, mock_build_index):
        """Test getting available resources with environment variable set."""
        mock_build_index.return_value = {"file1.txt": Mock(), "file2.csv": Mock()}

        result = get_available_resources()

        assert result == ["file1.txt", "file2.csv"]
        mock_build_index.assert_called_once()
        # Check that Path was called with the environment variable
        called_path = mock_build_index.call_args[0][0]
        assert str(called_path) == "/test/path"

    @patch("mcp_biomni.server.add_resources.build_file_index")
    def test_get_available_resources_no_env_var(self, mock_build_index):
        """Test getting available resources without environment variable."""
        # Ensure BIOMNI_DATA_PATH is not set
        with patch.dict(os.environ, {}, clear=True):
            mock_build_index.return_value = {}

            result = get_available_resources()

            assert result == []
            mock_build_index.assert_called_once_with(None)


if __name__ == "__main__":
    pytest.main([__file__])
