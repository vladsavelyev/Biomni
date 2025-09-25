import os
import tempfile
import unittest
from unittest.mock import patch

from mcp_biomni.scripts.check_and_download_files import (
    main,
    setup_biomni_data_environment,
)


class TestSetupBiomniDataEnvironment(unittest.TestCase):
    """Test cases for setup_biomni_data_environment function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self):
        """Clean up temporary directory after tests."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("mcp_biomni.scripts.check_and_download_files.check_and_download_s3_files")
    @patch("mcp_biomni.scripts.check_and_download_files.data_lake_dict")
    def test_setup_with_custom_path(self, mock_data_lake_dict, mock_download_s3):
        """Test setup with custom path parameter."""
        # Mock data_lake_dict
        mock_data_lake_dict.keys.return_value = ["file1.parquet", "file2.tsv"]
        mock_download_s3.return_value = {"file1.parquet": True, "file2.tsv": True}

        # Call function with custom path
        result_path = setup_biomni_data_environment(custom_path=self.temp_dir)

        # Assertions
        expected_path = os.path.join(self.temp_dir, "biomni_data")
        self.assertEqual(result_path, expected_path)

        # Check that directories were created
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "biomni_data", "benchmark"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "biomni_data", "data_lake"))
        )

        # Verify S3 download was called for data lake
        mock_download_s3.assert_any_call(
            s3_bucket_url="https://biomni-release.s3.amazonaws.com",
            local_data_lake_path=os.path.join(
                self.temp_dir, "biomni_data", "data_lake"
            ),
            expected_files=["file1.parquet", "file2.tsv"],
            folder="data_lake",
        )

    @patch.dict(os.environ, {"BIOMNI_DATA_PATH": "/custom/env/path"})
    @patch("mcp_biomni.scripts.check_and_download_files.check_and_download_s3_files")
    @patch("mcp_biomni.scripts.check_and_download_files.data_lake_dict")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_setup_with_env_variable(
        self, mock_makedirs, mock_exists, mock_data_lake_dict, mock_download_s3
    ):
        """Test setup using BIOMNI_DATA_PATH environment variable."""
        # Setup mocks
        mock_exists.return_value = False
        mock_data_lake_dict.keys.return_value = ["file1.parquet"]
        mock_download_s3.return_value = {"file1.parquet": True}

        with patch("builtins.print") as mock_print:
            result_path = setup_biomni_data_environment()

        # Verify environment variable was used
        expected_path = "/custom/env/path/biomni_data"
        self.assertEqual(result_path, expected_path)

        # Verify directory creation was called
        mock_makedirs.assert_any_call("/custom/env/path")
        mock_print.assert_any_call("Created directory: /custom/env/path")

    @patch("mcp_biomni.scripts.check_and_download_files.check_and_download_s3_files")
    @patch("mcp_biomni.scripts.check_and_download_files.data_lake_dict")
    @patch("os.path.expanduser")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_setup_with_default_path(
        self,
        mock_makedirs,
        mock_exists,
        mock_expanduser,
        mock_data_lake_dict,
        mock_download_s3,
    ):
        """Test setup using default path when no env variable is set."""
        # Setup mocks
        mock_expanduser.return_value = "/Users/test/biomni"
        mock_exists.return_value = True
        mock_data_lake_dict.keys.return_value = ["file1.parquet"]
        mock_download_s3.return_value = {"file1.parquet": True}

        with patch.dict(os.environ, {}, clear=True):
            result_path = setup_biomni_data_environment()

        # Verify default path was used
        expected_path = "/Users/test/biomni/biomni_data"
        self.assertEqual(result_path, expected_path)
        mock_expanduser.assert_called_with("~/biomni")

    @patch("mcp_biomni.scripts.check_and_download_files.check_and_download_s3_files")
    @patch("mcp_biomni.scripts.check_and_download_files.data_lake_dict")
    def test_setup_with_specific_data_lake_files(
        self, mock_data_lake_dict, mock_download_s3
    ):
        """Test setup with specific data lake files list."""
        mock_download_s3.return_value = {"specific_file.parquet": True}

        specific_files = ["specific_file.parquet"]
        _ = setup_biomni_data_environment(
            custom_path=self.temp_dir, expected_data_lake_files=specific_files
        )

        # Verify specific files were used instead of data_lake_dict.keys()
        mock_download_s3.assert_any_call(
            s3_bucket_url="https://biomni-release.s3.amazonaws.com",
            local_data_lake_path=os.path.join(
                self.temp_dir, "biomni_data", "data_lake"
            ),
            expected_files=specific_files,
            folder="data_lake",
        )

    @patch("mcp_biomni.scripts.check_and_download_files.check_and_download_s3_files")
    @patch("mcp_biomni.scripts.check_and_download_files.data_lake_dict")
    @patch("os.path.isdir")
    def test_benchmark_directory_exists_complete(
        self, mock_isdir, mock_data_lake_dict, mock_download_s3
    ):
        """Test when benchmark directory already exists and is complete."""
        # Setup mocks
        mock_data_lake_dict.keys.return_value = []
        mock_download_s3.return_value = {}

        def isdir_side_effect(path):
            if "benchmark" in path and "hle" not in path:
                return True
            elif "hle" in path:
                return True
            return False

        mock_isdir.side_effect = isdir_side_effect

        _ = setup_biomni_data_environment(custom_path=self.temp_dir)

        # Verify benchmark download was NOT called (since directory is complete)
        calls = mock_download_s3.call_args_list
        benchmark_calls = [
            call_item
            for call_item in calls
            if any("benchmark" in str(arg) for arg in call_item[1].values())
        ]
        self.assertEqual(len(benchmark_calls), 0)

    @patch("mcp_biomni.scripts.check_and_download_files.check_and_download_s3_files")
    @patch("mcp_biomni.scripts.check_and_download_files.data_lake_dict")
    @patch("os.path.isdir")
    def test_benchmark_directory_incomplete(
        self, mock_isdir, mock_data_lake_dict, mock_download_s3
    ):
        """Test when benchmark directory is incomplete or missing."""
        # Setup mocks
        mock_data_lake_dict.keys.return_value = []
        mock_download_s3.return_value = {}

        def isdir_side_effect(path):
            if "benchmark" in path and "hle" not in path:
                return True
            elif "hle" in path:
                return False  # hle directory is missing
            return False

        mock_isdir.side_effect = isdir_side_effect

        with patch("builtins.print") as mock_print:
            _ = setup_biomni_data_environment(custom_path=self.temp_dir)

        # Verify benchmark download WAS called
        mock_download_s3.assert_any_call(
            s3_bucket_url="https://biomni-release.s3.amazonaws.com",
            local_data_lake_path=os.path.join(
                self.temp_dir, "biomni_data", "benchmark"
            ),
            expected_files=[],
            folder="benchmark",
        )
        mock_print.assert_any_call("Checking and downloading benchmark files...")

    @patch("builtins.print")
    def test_directory_creation_output(self, mock_print):
        """Test that directory creation prints appropriate message."""
        with patch(
            "mcp_biomni.scripts.check_and_download_files.check_and_download_s3_files"
        ), patch(
            "mcp_biomni.scripts.check_and_download_files.data_lake_dict"
        ) as mock_data_lake_dict:
            mock_data_lake_dict.keys.return_value = []

            # Use a non-existent directory
            non_existent_path = os.path.join(self.temp_dir, "new_directory")
            result_path = setup_biomni_data_environment(custom_path=non_existent_path)

            # Check completion message was printed
            mock_print.assert_any_call(
                f"BIOMNI data environment setup complete at: {result_path}"
            )


class TestMainFunction(unittest.TestCase):
    """Test cases for main function."""

    @patch("mcp_biomni.scripts.check_and_download_files.setup_biomni_data_environment")
    @patch("builtins.print")
    def test_main_success(self, mock_print, mock_setup):
        """Test main function with successful setup."""
        mock_setup.return_value = "/path/to/biomni_data"

        result = main()

        self.assertEqual(result, 0)
        mock_print.assert_any_call("Setting up BIOMNI data environment...")
        mock_print.assert_any_call("Biomni Data Setup successful: /path/to/biomni_data")

    @patch("mcp_biomni.scripts.check_and_download_files.setup_biomni_data_environment")
    @patch("builtins.print")
    def test_main_failure(self, mock_print, mock_setup):
        """Test main function with setup failure."""
        mock_setup.side_effect = Exception("Test error")

        result = main()

        self.assertEqual(result, 1)
        mock_print.assert_any_call("Setting up BIOMNI data environment...")
        mock_print.assert_any_call("Biomni Data Setup failed: Test error")


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self):
        """Clean up temporary directory after tests."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("mcp_biomni.scripts.check_and_download_files.check_and_download_s3_files")
    @patch("mcp_biomni.scripts.check_and_download_files.data_lake_dict")
    def test_full_integration_flow(self, mock_data_lake_dict, mock_download_s3):
        """Test complete integration flow."""
        # Setup mocks
        mock_data_lake_dict.keys.return_value = ["file1.parquet", "file2.tsv"]
        mock_download_s3.return_value = {"file1.parquet": True, "file2.tsv": True}

        # Run the full setup
        result_path = setup_biomni_data_environment(custom_path=self.temp_dir)

        # Verify the complete flow
        self.assertEqual(result_path, os.path.join(self.temp_dir, "biomni_data"))

        # Check directory structure was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "biomni_data")))
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "biomni_data", "benchmark"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "biomni_data", "data_lake"))
        )

        # Verify both S3 download calls were made
        self.assertEqual(
            mock_download_s3.call_count, 2
        )  # One for data_lake, one for benchmark


if __name__ == "__main__":
    unittest.main()
