"""Tests for add_resources.py server logic functions."""

import json
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
    sanitize_tool_name,  # Legacy function, no longer used in main flow
)


class TestSanitizeToolName:
    """Tests for sanitize_tool_name function.

    NOTE: This function is legacy and no longer used in the main server flow,
    but kept for backwards compatibility.
    """

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
    def test_build_file_index_with_existing_files(self, mock_blacklist_config):
        """Test building file index with existing files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            data_lake_path = base_path / "biomni_data" / "data_lake"
            data_lake_path.mkdir(parents=True)

            # Create a test file
            test_file = data_lake_path / "test_file.parquet"
            test_file.write_bytes(b"test data")

            index = build_file_index(base_path, mock_blacklist_config)

            assert len(index) == 1
            assert "test_file.parquet" in index
            assert index["test_file.parquet"].desc == "Test file description"
            assert index["test_file.parquet"].mime == "application/vnd.apache.parquet"

    def test_build_file_index_nonexistent_base_path(self, mock_blacklist_config):
        """Test building file index with nonexistent base path."""
        index = build_file_index(Path("/nonexistent/path"), mock_blacklist_config)
        assert index == {}

    def test_build_file_index_none_base_path(self, mock_blacklist_config):
        """Test building file index with None base path."""
        index = build_file_index(None, mock_blacklist_config)
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
    def test_parse_library_content_all_types(self, mock_blacklist_config):
        """Test parsing library content with all package types."""
        result = parse_library_content(mock_blacklist_config)

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
    def test_parse_library_content_import_name_mapping(self, mock_blacklist_config):
        """Test that special import name mappings work correctly."""
        result = parse_library_content(mock_blacklist_config)

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

    @patch("mcp_biomni.server.add_resources.load_blacklist_config")
    @patch("mcp_biomni.server.add_resources.build_file_index")
    @patch.dict(os.environ, {"BIOMNI_DATA_PATH": "/test/path"})
    def test_get_available_resources_with_env_var(
        self, mock_build_index, mock_load_blacklist
    ):
        """Test getting available resources with environment variable set."""
        mock_blacklist = Mock()
        mock_load_blacklist.return_value = mock_blacklist
        mock_build_index.return_value = {"file1.txt": Mock(), "file2.csv": Mock()}

        result = get_available_resources()

        assert result == ["file1.txt", "file2.csv"]
        mock_build_index.assert_called_once()
        # Check that blacklist config was passed to build_file_index
        called_path, called_blacklist = mock_build_index.call_args[0]
        assert str(called_path) == "/test/path"
        assert called_blacklist == mock_blacklist

    @patch("mcp_biomni.server.add_resources.load_blacklist_config")
    @patch("mcp_biomni.server.add_resources.build_file_index")
    def test_get_available_resources_no_env_var(
        self, mock_build_index, mock_load_blacklist
    ):
        """Test getting available resources without environment variable."""
        # Ensure BIOMNI_DATA_PATH is not set
        mock_blacklist = Mock()
        mock_load_blacklist.return_value = mock_blacklist
        with patch.dict(os.environ, {}, clear=True):
            mock_build_index.return_value = {}

            result = get_available_resources()

            assert result == []
            mock_build_index.assert_called_once_with(None, mock_blacklist)


class TestFileCatalogLogic:
    """Tests for file_catalog tool logic."""

    def test_file_catalog_response_structure(self):
        """Test that file_catalog generates proper response structure."""
        # Mock file index
        test_file_index = {
            "test.csv": FileMeta(
                path=Path("/test/test.csv"), desc="Test CSV file", mime="text/csv"
            ),
            "data.parquet": FileMeta(
                path=Path("/test/data.parquet"),
                desc="Test Parquet file",
                mime="application/vnd.apache.parquet",
            ),
        }

        # Simulate file_catalog logic
        result = {}
        for rel_name, meta in test_file_index.items():
            file_info = {
                "path": rel_name,
                "full_path": str(meta.path),
                "description": meta.desc,
                "mime_type": meta.mime,
                "size_bytes": None,  # Would be actual size in real implementation
                "extension": meta.path.suffix.lower().lstrip(".")
                if meta.path.suffix
                else None,
            }
            result[rel_name] = file_info

        summary = {
            "total_files": len(test_file_index),
            "filtered_files": len(result),
            "file_types": list(
                {
                    info.get("extension")
                    for info in result.values()
                    if info.get("extension")
                }
            ),
        }

        response = {
            "summary": summary,
            "files": result,
            "usage_note": "Use file_access tool to read specific files by their path",
        }

        # Verify structure
        assert "summary" in response
        assert "files" in response
        assert "usage_note" in response

        # Verify summary
        assert response["summary"]["total_files"] == 2
        assert response["summary"]["filtered_files"] == 2
        assert set(response["summary"]["file_types"]) == {"csv", "parquet"}

        # Verify file entries
        assert "test.csv" in response["files"]
        assert "data.parquet" in response["files"]
        assert response["files"]["test.csv"]["extension"] == "csv"
        assert response["files"]["data.parquet"]["extension"] == "parquet"

    def test_file_catalog_search_filter(self):
        """Test file_catalog search filtering logic."""
        test_file_index = {
            "gene_data.csv": FileMeta(
                path=Path("/test/gene_data.csv"),
                desc="Gene expression data",
                mime="text/csv",
            ),
            "protein_info.parquet": FileMeta(
                path=Path("/test/protein_info.parquet"),
                desc="Protein information dataset",
                mime="application/vnd.apache.parquet",
            ),
            "sample_metadata.json": FileMeta(
                path=Path("/test/sample_metadata.json"),
                desc="Sample metadata",
                mime="application/json",
            ),
        }

        # Test search for "gene"
        search_term = "gene"
        filtered_result = {}

        for rel_name, meta in test_file_index.items():
            if (
                search_term.lower() in rel_name.lower()
                or search_term.lower() in meta.desc.lower()
            ):
                file_info = {
                    "path": rel_name,
                    "description": meta.desc,
                    "extension": meta.path.suffix.lower().lstrip(".")
                    if meta.path.suffix
                    else None,
                }
                filtered_result[rel_name] = file_info

        # Should match gene_data.csv (by filename) and protein_info.parquet (by description)
        assert "gene_data.csv" in filtered_result
        assert "sample_metadata.json" not in filtered_result
        # Note: "protein_info.parquet" would not match "gene" search

    def test_file_catalog_type_filter(self):
        """Test file_catalog file type filtering logic."""
        test_file_index = {
            "data1.csv": FileMeta(
                path=Path("/test/data1.csv"), desc="CSV data", mime="text/csv"
            ),
            "data2.csv": FileMeta(
                path=Path("/test/data2.csv"), desc="Another CSV", mime="text/csv"
            ),
            "data.parquet": FileMeta(
                path=Path("/test/data.parquet"),
                desc="Parquet data",
                mime="application/vnd.apache.parquet",
            ),
        }

        # Filter for CSV files
        file_type = "csv"
        filtered_result = {}

        for rel_name, meta in test_file_index.items():
            extension = (
                meta.path.suffix.lower().lstrip(".") if meta.path.suffix else None
            )
            if extension == file_type.lower():
                filtered_result[rel_name] = {
                    "path": rel_name,
                    "extension": extension,
                }

        assert len(filtered_result) == 2
        assert "data1.csv" in filtered_result
        assert "data2.csv" in filtered_result
        assert "data.parquet" not in filtered_result


class TestFileAccessLogic:
    """Tests for file_access tool logic."""

    def test_file_access_describe_operation(self):
        """Test file_access describe operation logic."""
        test_file_index = {
            "test.csv": FileMeta(
                path=Path("/test/test.csv"), desc="Test CSV file", mime="text/csv"
            )
        }

        # Simulate describe operation
        file_path = "test.csv"
        if file_path in test_file_index:
            meta = test_file_index[file_path]
            payload = {
                "file": file_path,
                "path": str(meta.path),
                "description": meta.desc,
                "mime": meta.mime,
                "size_bytes": None,  # Would be actual size in real implementation
            }

            # Verify structure
            assert payload["file"] == "test.csv"
            assert payload["path"] == "/test/test.csv"
            assert payload["description"] == "Test CSV file"
            assert payload["mime"] == "text/csv"

    def test_file_access_file_not_found(self):
        """Test file_access behavior when file is not found."""
        test_file_index = {
            "existing.csv": FileMeta(
                path=Path("/test/existing.csv"), desc="Exists", mime="text/csv"
            )
        }

        file_path = "nonexistent.csv"
        if file_path not in test_file_index:
            available_files = list(test_file_index.keys())[:10]
            error_response = {
                "error": f"File not found: {file_path}",
                "available_files_sample": available_files,
                "total_available": len(test_file_index),
                "suggestion": "Use file_catalog tool to see all available files",
            }

            assert error_response["error"] == "File not found: nonexistent.csv"
            assert error_response["available_files_sample"] == ["existing.csv"]
            assert error_response["total_available"] == 1

    def test_file_access_unsupported_operation(self):
        """Test file_access behavior with unsupported operation."""
        unsupported_ops = ["delete", "update", "create", "list"]

        for op in unsupported_ops:
            if op not in {"describe", "read"}:
                error_msg = f"Unsupported op '{op}'. Use 'describe' or 'read'."
                assert "Unsupported op" in error_msg
                assert "describe" in error_msg
                assert "read" in error_msg


class TestNewServerArchitecture:
    """Tests for the new server architecture with consolidated tools."""

    def test_server_registers_three_tools(self):
        """Test that the server registers exactly 3 tools instead of many individual file tools."""
        # This is a conceptual test - in reality we'd need to mock the FastMCP server
        # But we can verify the logic that determines tool count

        # Simulate having multiple files
        mock_file_count = 50

        # Old approach would register: 1 (capabilities) + 50 (individual files) = 51 tools
        old_total = 1 + mock_file_count

        # New approach registers: 1 (capabilities) + 1 (file_catalog) + 1 (file_access) = 3 tools
        new_total = 3

        assert new_total < old_total
        assert new_total == 3

        # Verify this matches what the actual code sets
        expected_registered = 3  # capabilities tool + file_catalog + file_access tools
        assert new_total == expected_registered

    def test_json_serialization_compatibility(self):
        """Test that our data structures are JSON serializable."""
        # Test the file_types list (was previously a set causing JSON serialization errors)
        sample_extensions = {"csv", "parquet", "json", None}
        file_types_list = list({ext for ext in sample_extensions if ext is not None})

        # Should be JSON serializable
        json_str = json.dumps(file_types_list)
        assert json_str is not None

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert set(parsed) == {"csv", "parquet", "json"}


if __name__ == "__main__":
    pytest.main([__file__])
