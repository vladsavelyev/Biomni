import asyncio
import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware

from biomni.env_desc import data_lake_dict, library_content_dict
from mcp_biomni.scripts.check_and_download_files import setup_biomni_data_environment

# --------------------
# Helpers
# --------------------


def sanitize_tool_name(name: str) -> str:
    # Create a tool-safe identifier: file_<stem> (alnum + underscore only)
    stem = Path(name).stem
    safe = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    # Avoid leading digits
    if re.match(r"^\d", safe):
        safe = f"f_{safe}"
    return f"file_{safe}"


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        # Some common fallbacks
        if path.suffix.lower() in {".parquet"}:
            return "application/vnd.apache.parquet"
        if path.suffix.lower() in {".pkl"}:
            return "application/octet-stream"
        if path.suffix.lower() in {".tsv"}:
            return "text/tab-separated-values"
        if path.suffix.lower() in {".obo"}:
            return "text/plain"
        return "application/octet-stream"
    return mime


def looks_textual(mime: str) -> bool:
    if mime.startswith("text/"):
        return True
    # Consider JSON, CSV/TSV as text-like
    if mime in {"application/json", "application/xml"}:
        return True
    if mime in {"text/csv", "text/tab-separated-values"}:
        return True
    # Parquet, pickle, etc. are binary
    return False


async def read_bytes(path: Path, max_bytes: int | None = None) -> bytes:
    loop = asyncio.get_running_loop()

    def _read():
        with open(path, "rb") as f:
            if max_bytes is not None and max_bytes >= 0:
                return f.read(max_bytes)
            return f.read()

    return await loop.run_in_executor(None, _read)


@dataclass
class FileMeta:
    path: Path
    desc: str
    mime: str


def build_file_index(base_path: Path | None) -> dict[str, FileMeta]:
    """Build index of available data files following Biomni's directory structure."""
    index: dict[str, FileMeta] = {}
    if not base_path or not base_path.exists() or not base_path.is_dir():
        return index

    # Follow Biomni's expected directory structure: base_path/biomni_data/data_lake/
    data_lake_path = base_path / "biomni_data" / "data_lake"
    if not data_lake_path.exists() or not data_lake_path.is_dir():
        return index

    for rel, desc in data_lake_dict.items():
        p = data_lake_path / rel
        if p.exists() and p.is_file():
            index[rel] = FileMeta(path=p, desc=desc, mime=guess_mime(p))
    return index


def parse_library_content() -> dict[str, dict]:
    """Parse library_content_dict into structured categories."""
    python_packages = {}
    r_packages = {}
    cli_tools = {}

    # Special cases for Python package import names
    import_name_mapping = {
        "biopython": "Bio",
        "scikit-learn": "sklearn",
        "scikit-bio": "skbio",
        "scikit-image": "skimage",
        "opencv-python": "cv2",
        "umap-learn": "umap",
        "faiss-cpu": "faiss",
        "harmony-pytorch": "harmony",
        "python-libsbml": "libsbml",
        "PyPDF2": "PyPDF2",
        "googlesearch-python": "googlesearch",
        "cryosparc-tools": "cryosparc_tools",
    }

    for name, description in library_content_dict.items():
        # Extract the package type from the description
        if "[Python Package]" in description:
            # Clean description and extract import info
            clean_desc = description.replace("[Python Package] ", "")
            import_name = import_name_mapping.get(name, name)
            python_packages[name] = {
                "import_name": import_name,
                "package_name": name,
                "description": clean_desc,
                "import_example": f"import {import_name}",
                "type": "python_package",
            }
        elif "[R Package]" in description:
            # Extract subprocess usage info
            clean_desc = description.replace("[R Package] ", "")
            r_packages[name] = {
                "package_name": name,
                "description": clean_desc,
                "subprocess_example": f"subprocess.run(['Rscript', '-e', 'library({name}); your_r_code_here'])",
                "rscript_example": f'Rscript -e "library({name}); your_r_code_here"',
                "type": "r_package",
            }
        elif "[CLI Tool]" in description:
            # Extract CLI usage info
            clean_desc = description.replace("[CLI Tool] ", "")
            cli_tools[name] = {
                "executable": name,
                "description": clean_desc,
                "subprocess_example": f"subprocess.run(['{name}', 'arg1', 'arg2'])",
                "command_example": f"{name} --help  # for usage information",
                "type": "cli_tool",
            }
        else:
            # Handle packages without explicit type markers (assume CLI tools)
            cli_tools[name] = {
                "executable": name,
                "description": description,
                "subprocess_example": f"subprocess.run(['{name}', 'arg1', 'arg2'])",
                "command_example": f"{name} --help  # for usage information",
                "type": "cli_tool",
            }

    return {
        "python_packages": python_packages,
        "r_packages": r_packages,
        "cli_tools": cli_tools,
    }


def serve_resources(port: int, host: str = "0.0.0.0") -> None:  # noqa: B104
    """Start one FastMCP server that exposes data lake files as tools."""
    try:
        # Setup and download missing data files first
        print("[resources] Setting up BIOMNI data environment...")
        setup_biomni_data_environment()

        base = os.environ.get("BIOMNI_DATA_PATH")
        base_path = Path(base).resolve() if base else None

        file_index = build_file_index(base_path)

        if not file_index:
            print(
                f"✗ [resources] No data files found at {base_path if base_path else 'UNSET'}"
            )
            return

        mcp = FastMCP(name="Biomni-DataLake")

        # Add middleware for logging, error handling, and timing
        mcp.add_middleware(LoggingMiddleware(include_payloads=True))
        mcp.add_middleware(ErrorHandlingMiddleware(include_traceback=True))
        mcp.add_middleware(TimingMiddleware())

        # Parse library content for capabilities tool
        capabilities = parse_library_content()

        # Register tool_capabilities tool
        @mcp.tool(
            description="Returns a machine-readable catalog of available Python packages, R packages, and CLI tools with usage information",
            tags={"biomni", "capabilities", "packages"},
        )
        async def tool_capabilities(
            category: str | None = None, search: str | None = None
        ) -> str:
            """
            Returns catalog of available tools and packages.

            Args:
                category: Filter by category ("python", "r", "cli", "all"). Default: "all"
                search: Search term to filter results by name or description
            """
            result = {}
            chosen_category = (category or "all").lower()

            if chosen_category in ("all", "python"):
                result["python_packages"] = capabilities["python_packages"]
            if chosen_category in ("all", "r"):
                result["r_packages"] = capabilities["r_packages"]
            if chosen_category in ("all", "cli"):
                result["cli_tools"] = capabilities["cli_tools"]

            # Apply search filter if provided
            if search:
                search_term = search.lower()
                filtered_result = {}

                for cat_name, packages in result.items():
                    filtered_packages = {}
                    for pkg_name, pkg_info in packages.items():
                        if (
                            search_term in pkg_name.lower()
                            or search_term in pkg_info.get("description", "").lower()
                        ):
                            filtered_packages[pkg_name] = pkg_info

                    if filtered_packages:
                        filtered_result[cat_name] = filtered_packages

                result = filtered_result

            # Add summary statistics
            summary = {
                "total_python_packages": len(capabilities["python_packages"]),
                "total_r_packages": len(capabilities["r_packages"]),
                "total_cli_tools": len(capabilities["cli_tools"]),
            }

            return json.dumps(
                {
                    "summary": summary,
                    "capabilities": result,
                    "usage_notes": {
                        "python_packages": "Import directly: import package_name",
                        "r_packages": "Use via subprocess: subprocess.run(['Rscript', '-e', 'library(package_name); ...'])",
                        "cli_tools": "Use via subprocess: subprocess.run(['tool_name', 'args'])",
                    },
                },
                indent=2,
            )

        print("✓ [resources] registered tool_capabilities")

        # Register file_catalog tool
        @mcp.tool(
            description="Returns a machine-readable catalog of all available data lake files with their metadata and descriptions",
            tags={"biomni", "data-lake", "files"},
        )
        async def file_catalog(
            search: str | None = None, file_type: str | None = None
        ) -> str:
            """
            Returns catalog of all available data lake files.

            Args:
                search: Search term to filter files by name or description
                file_type: Filter by file extension (e.g., 'csv', 'parquet', 'json')
            """
            result = {}

            for rel_name, meta in file_index.items():
                file_info = {
                    "path": rel_name,
                    "full_path": str(meta.path),
                    "description": meta.desc,
                    "mime_type": meta.mime,
                    "size_bytes": meta.path.stat().st_size
                    if meta.path.exists()
                    else None,
                    "extension": meta.path.suffix.lower().lstrip(".")
                    if meta.path.suffix
                    else None,
                }

                # Apply filters
                include_file = True

                if search:
                    search_term = search.lower()
                    if not (
                        search_term in rel_name.lower()
                        or search_term in meta.desc.lower()
                    ):
                        include_file = False

                if file_type and include_file:
                    if file_info["extension"] != file_type.lower():
                        include_file = False

                if include_file:
                    result[rel_name] = file_info

            summary = {
                "total_files": len(file_index),
                "filtered_files": len(result),
                "file_types": list(
                    {
                        info.get("extension")
                        for info in result.values()
                        if info.get("extension")
                    }
                ),
            }

            return json.dumps(
                {
                    "summary": summary,
                    "files": result,
                    "usage_note": "Use file_access tool to read specific files by their path",
                },
                indent=2,
            )

        print("✓ [resources] registered file_catalog")

        # Register file_access tool
        @mcp.tool(
            description="Access and read specific data lake files by their relative path",
            tags={"biomni", "data-lake", "file-access"},
        )
        async def file_access(
            file_path: str,
            op: str | None = None,
            max_bytes: int | None = None,
            as_text: bool | None = None,
        ) -> str:
            """
            Access a specific data lake file.

            Args:
                file_path: Relative path of the file in the data lake
                op: Operation to perform ('describe' or 'read'). Default: 'describe'
                max_bytes: Maximum bytes to read (for 'read' operation)
                as_text: Force text mode for binary files (for 'read' operation)
            """
            chosen_op = (op or "describe").lower()

            if chosen_op not in {"describe", "read"}:
                return f"Unsupported op '{op}'. Use 'describe' or 'read'."

            if file_path not in file_index:
                available_files = list(file_index.keys())[
                    :10
                ]  # Show first 10 as examples
                return json.dumps(
                    {
                        "error": f"File not found: {file_path}",
                        "available_files_sample": available_files,
                        "total_available": len(file_index),
                        "suggestion": "Use file_catalog tool to see all available files",
                    },
                    indent=2,
                )

            meta = file_index[file_path]

            if chosen_op == "describe":
                payload = {
                    "file": file_path,
                    "path": str(meta.path),
                    "description": meta.desc,
                    "mime": meta.mime,
                    "size_bytes": meta.path.stat().st_size
                    if meta.path.exists()
                    else None,
                }
                return json.dumps(payload, indent=2)

            # read operation
            if not meta.path.exists():
                return f"File not found: {file_path}"

            mime = meta.mime
            want_text = looks_textual(mime)
            if as_text is True:
                want_text = True
            elif as_text is False:
                want_text = False

            data = await read_bytes(meta.path, max_bytes=max_bytes)

            if want_text:
                text = data.decode("utf-8", errors="replace")
                # Return as text with header
                header = f"[file={file_path} mime={mime} bytes={len(data)}]"
                return header + "\n" + text
            else:
                # Return base64 encoded data with metadata
                b64 = base64.b64encode(data).decode("ascii")
                header = (
                    f"[file={file_path} mime={mime} bytes={len(data)} encoding=base64]"
                )
                return header + "\n" + b64

        print("✓ [resources] registered file_access")

        total_registered = 3  # capabilities tool + file_catalog + file_access tools
        print(f"[resources] total data tools registered: {total_registered}")
        print(f"[resources] BIOMNI_DATA_PATH={base_path if base_path else 'UNSET'}")
        mcp.run(transport="streamable-http", host=host, port=port)

    except Exception as exc:
        print(f"✗ Failed to start resource server: {exc}")


def get_available_resources() -> list[str]:
    """Get list of available data lake files."""
    base = os.environ.get("BIOMNI_DATA_PATH")
    base_path = Path(base).resolve() if base else None

    file_index = build_file_index(base_path)
    return list(file_index.keys())
