import os

from biomni.env_desc import data_lake_dict
from biomni.utils import check_and_download_s3_files


def setup_biomni_data_environment(
    custom_path: str | None = None,
    expected_data_lake_files: list[str] | None = None,
) -> str:
    """
    Set up the BIOMNI data environment by checking and downloading necessary files.

    Args:
        custom_path: Custom path to use instead of BIOMNI_DATA_PATH env variable
        expected_data_lake_files: List of specific data lake files to check/download
        If None, uses all files from data_lake_dict

    Returns:
        Path to the biomni_data directory
    """
    # Get path from custom parameter, environment variable, or default
    if custom_path:
        path = custom_path
    else:
        path = os.environ.get("BIOMNI_DATA_PATH", os.path.expanduser("~/biomni"))

    # Create base directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

    # --- Begin custom folder/file checks ---
    benchmark_dir = os.path.join(path, "biomni_data", "benchmark")
    data_lake_dir = os.path.join(path, "biomni_data", "data_lake")

    # Create the biomni_data directory structure
    os.makedirs(benchmark_dir, exist_ok=True)
    os.makedirs(data_lake_dir, exist_ok=True)

    if expected_data_lake_files is None:
        expected_data_lake_files = list(data_lake_dict.keys())

    # Check and download missing data lake files
    print("Checking and downloading missing data lake files...")
    check_and_download_s3_files(
        s3_bucket_url="https://biomni-release.s3.amazonaws.com",
        local_data_lake_path=data_lake_dir,
        expected_files=expected_data_lake_files,
        folder="data_lake",
    )

    # Check if benchmark directory structure is complete
    benchmark_ok = False
    if os.path.isdir(benchmark_dir):
        patient_gene_detection_dir = os.path.join(benchmark_dir, "hle")
        if os.path.isdir(patient_gene_detection_dir):
            benchmark_ok = True

    if not benchmark_ok:
        print("Checking and downloading benchmark files...")
        check_and_download_s3_files(
            s3_bucket_url="https://biomni-release.s3.amazonaws.com",
            local_data_lake_path=benchmark_dir,
            expected_files=[],  # Empty list - will download entire folder
            folder="benchmark",
        )

    biomni_data_path = os.path.join(path, "biomni_data")
    print(f"BIOMNI data environment setup complete at: {biomni_data_path}")
    return biomni_data_path


def main():
    """
    Main function to run the setup when this script is executed directly.
    """
    print("Setting up BIOMNI data environment...")
    try:
        setup_path = setup_biomni_data_environment()
        print(f"Biomni Data Setup successful: {setup_path}")
    except Exception as e:
        print(f"Biomni Data Setup failed: {str(e)}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
