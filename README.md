# MCP Biomni

A Model Context Protocol (MCP) server that exposes biomedical research tools and data resources from the Biomni project for AI Assistants.

## Overview

MCP Biomni is a server that implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) to provide biomedical research capabilities from the [Biomni project](https://github.com/snap-stanford/Biomni). It exposes both computational tools and data resources, enabling AI assistants to perform sophisticated biomedical analyses.

### Features

- **Biomni Tools**: 18+ biomedical tool modules (biochemistry, genetics, genomics, etc.)
- **Data Resources**: 40+ curated biological datasets (protein interactions, gene expression, GWAS, etc.)
- **Unified Server**: Single server process serving both tools and data via MCP
- **Scalable**: Multi-process architecture for concurrent tool execution
- **Configurable**: Support for multiple LLM providers and commercial/academic modes
- **Auto-Setup**: Automatically checks and downloads missing data files on startup

## Architecture

The server provides two main types of resources:

### Tool Modules

- **biochemistry**: Protein structure, molecular interactions
- **genetics**: Gene analysis, variant interpretation
- **genomics**: Sequence analysis, genome-wide studies
- **cell_biology**: Cell type analysis, cellular processes
- **molecular_biology**: Molecular mechanisms, pathways
- **pharmacology**: Drug discovery, ADMET prediction
- **literature**: Scientific paper search and analysis
- **database**: Biomedical database queries
- **support_tools**: Utility functions and helpers
- And 9 more specialized modules...

### Data Resources Module

The files provided by Biomni are converted into tools to access the biological datasets directly:

- `file_gene_info`: Comprehensive gene information and annotations
- `file_go_plus`: Gene Ontology data for functional annotations
- `file_gtex_tissue_gene_tpm`: Gene expression across human tissues
- `file_depmap_gene_dependency`: Cancer cell line dependency data
- ...

These files will be downloaded automatically before setting up the MCP server if no present on `BIOMNI_DATA_PATH`.

#### Tool Capabilities Discovery

The data resources server provides a special `tool_capabilities` tool that returns a machine-readable catalog of all available packages and tools in the Biomni environment:

```json
{
  "summary": {
    "total_python_packages": 70,
    "total_r_packages": 12,
    "total_cli_tools": 29
  },
  "capabilities": {
    "python_packages": {
      "biopython": {
        "import_name": "Bio",
        "package_name": "biopython",
        "description": "Tools for biological computation including parsers...",
        "import_example": "import Bio",
        "type": "python_package"
      }
    },
    "r_packages": {
      "ggplot2": {
        "package_name": "ggplot2",
        "description": "A system for creating graphics...",
        "subprocess_example": "subprocess.run(['Rscript', '-e', 'library(ggplot2); ...'])",
        "type": "r_package"
      }
    },
    "cli_tools": {
      "samtools": {
        "executable": "samtools",
        "description": "Suite of programs for sequencing data...",
        "subprocess_example": "subprocess.run(['samtools', 'view', 'file.bam'])",
        "type": "cli_tool"
      }
    }
  }
}
```

## Installation

### Prerequisites

- Python 3.11.2+
- Docker (optional)
- Biomni environment setup (see [Biomni installation guide](https://github.com/snap-stanford/Biomni))

### Local Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/azu-oncology-rd/mcp-biomni.git
   cd mcp-biomni
   ```

2. Install dependencies using uv:

   ```bash
   uv sync --all-extras
   ```

3. Activate virtual environment:

   ```bash
   source .venv/bin/activate
   ```

### Docker Installation

1. Build and run using Docker Compose:

   ```bash
   docker compose up -d
   ```

## Usage

### Starting the Server

The MCP Biomni server can serve tools, data resources, or both:

#### Command Line Options

```bash
# List available tool modules
uv run python -m mcp_biomni.server.server --list-modules

# List available data resources
uv run python -m mcp_biomni.server.server --list-resources

# Serve all tool modules (default)
uv run python -m mcp_biomni.server.server

# Serve specific tool modules
uv run python -m mcp_biomni.server.server --modules biochemistry genetics genomics

# Serve data resources only
uv run python -m mcp_biomni.server.server --resources

# Serve both tools and data resources
uv run python -m mcp_biomni.server.server --modules biochemistry --resources

# Custom host and port
uv run python -m mcp_biomni.server.server --host localhost --port 8001
```

#### Server Architecture

The server uses a multi-process architecture where each service runs on its own port:

- **Port 8001**: First tool module (e.g., biochemistry)
- **Port 8002**: Second tool module (e.g., genetics)
- **Port 8003**: Data resources server
- **...**: Additional modules on subsequent ports

Example output:

```text
Biomni MCP cluster running:
http://0.0.0.0:8001  →  biomni.tool.biochemistry
http://0.0.0.0:8002  →  biomni.tool.genetics
http://0.0.0.0:8003  →  biomni.data-lake
```

### Connecting to the Server

The MCP server exposes its API via HTTP (streamable HTTP transport).
You can connect to it using any MCP client, like [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector).

Each service endpoint provides:

- **Tool Modules**: Computational functions for biomedical analysis
- **Data Resources**: File access tools with `describe` and `read` operations

## Configuration

### Environment Variables

#### Required Setup

1. **Set Biomni environment variables**: Create a `.env` file with the following variables following the [`.env.example file`](./.env.example).

2. **Automatic Data Setup**: If the data is not in the `DATA_DIR` path, the server automatically:
   - Creates the required directory structure (`biomni_data/data_lake/`, `biomni_data/benchmark/`)
   - Checks for missing data files
   - Downloads missing files from Biomni's S3 repository
   - Sets up 40+ biological datasets for immediate use

   No manual data setup required. The server handles everything on first startup.

3. **Expected data structure** (created automatically):

   ```text
   $BIOMNI_DATA_PATH/
   └── biomni_data/
       ├── data_lake/          # 40+ biological datasets (auto-downloaded)
       │   ├── gene_info.parquet
       │   ├── go-plus.json
       │   ├── gtex_tissue_gene_tpm.parquet
       │   └── ...
       └── benchmark/          # Evaluation datasets (auto-downloaded)
           ├── DbQA/
           ├── SeqQA/
           └── hle/
   ```

### Blacklisting Configuration

MCP Biomni supports a flexible blacklisting system to exclude specific tools, files, and capabilities from being exposed via the MCP server. This is useful for controlling access, reducing server load, or complying with usage policies.

#### Configuration File

Create a `blacklist.yaml` file in the project root to define items to exclude:

```yaml
# MCP Biomni Blacklist Configuration
# This file defines items to exclude from MCP server exposure

# Tools to blacklist (exact tool names)
tools:
  - query_kegg
  - query_iucn

# Files/resources to blacklist (relative paths in data lake)
files:
  - DepMap_CRISPRGeneDependency.csv
  - DepMap_CRISPRGeneEffect.csv
  - DepMap_Model.csv
  - DepMap_OmicsExpressionProteinCodingGenesTPMLogp1.csv

# Capabilities to blacklist by category
capabilities:
  python_packages:
    - cryosparc-tools
    - cellpose
  cli_tools:
    - Homer

# Modules to blacklist (entire tool modules)
modules:
  - experimental_module
```

#### Blacklist Categories

**Tools**: Individual function names from any Biomni module that should not be exposed as MCP tools.

**Files**: Data lake files (relative paths) that should not be accessible via the `file_access` tool.

**Capabilities**: Specific packages or tools categorized by type:

- `python_packages`: Python packages excluded from the `tool_capabilities` catalog
- `r_packages`: R packages excluded from the catalog
- `cli_tools`: Command-line tools excluded from the catalog

**Modules**: Entire Biomni modules to exclude from the server (e.g., `biochemistry`, `genetics`).

#### How It Works

1. **Startup**: The blacklist configuration is loaded when each server process starts
2. **Tool Registration**: Tools listed in the blacklist are skipped during MCP tool registration
3. **File Access**: Blacklisted files return access denied errors when requested
4. **Capabilities**: The `tool_capabilities` tool excludes blacklisted packages from its catalog
5. **Module Filtering**: Entire modules are excluded from the available modules list

**Server logs will show when items are blacklisted:**

```text
✗ [database] query_kegg: blacklisted
✓ [database] query_uniprot: registered
Loaded blacklist config from: /app/blacklist.yaml
```

### Multi-Process Architecture

The MCP-Biomni server uses a multi-process architecture:

- **Main Orchestrator** (`server.py`): Manages worker processes and provides unified interface
- **Tool Servers** (`add_tools.py`): Each Biomni module runs on separate ports (8000+)
- **Resource Servers** (`add_resources.py`): Data file access on ports (8100+)

#### Port Allocation

- Tool modules: 8000, 8001, 8002, ... (one per module)
- Resource servers: 8100, 8101, 8102, ... (one per resource group)
- Each service is independently accessible and scalable

#### Available Services

**Tool Modules (18):**
biochemistry, bioengineering, biophysics, cancer_biology, cell_biology, database, genetics, genomics, immunology, literature, microbiology, molecular_biology, neuroscience, protein_structure, proteomics, synthetic_biology, systems_biology, transcriptomics

**Data Resources (40+):**
Gene expression data, protein interactions, chemical compounds, genomic annotations, pathway information, and more biological datasets

Use `--list-modules` and `--list-resources` commands to see complete listings.

## Development

### Pre-commit Hooks

This project uses pre-commit hooks with ruff for code quality. After cloning:

```bash
uv sync --all-extras
uv run pre-commit install
```

Now pre-commit will automatically run on every commit to ensure code quality.

### Project Structure

```text
mcp_biomni/
├── server/
│   ├── server.py           # Main server orchestrator
│   ├── add_tools.py        # Biomni tool module serving
│   └── add_resources.py    # Data resource serving
├── scripts/                # Utility scripts
│   └── check_and_download_files.py
└── tests/                  # Test suite
```

### Running Tests

```bash
pytest
# or with coverage
pytest --cov=mcp_biomni
```

### Linting

```bash
# Check and fix code with ruff
uv run ruff check --fix mcp_biomni/
uv run ruff format mcp_biomni/

# Or use pre-commit to run all checks
uv run pre-commit run --all-files
```

### Server Development

The server consists of three main components:

1. **server.py**: Main entry point that orchestrates tool and resource servers
2. **add_tools.py**: Serves Biomni tool modules as MCP functions
3. **add_resources.py**: Serves data files as MCP tools with describe/read operations

## Docker Configuration

The Docker setup includes:

- Base image: Ubuntu 22.04 with Biomni dependencies
- Python package management with uv
- Volume mounting for data persistence
- Development mode with file watching for hot reloading
- Conda environment for Biomni

Note: The `DATA_DIR` environment variable should point to a persistent directory for storing Biomni data and results. For example:

```bash
# In your .env file
DATA_DIR=/path/to/your/biomni/data
```

## Security Considerations

⚠️ **Important Security Warning**: Biomni executes LLM-generated code with system privileges. Always run in isolated/sandboxed environments for production use. The agent can access files, network, and system commands.

## Related Projects

- [Biomni](https://github.com/snap-stanford/Biomni) - The core biomedical AI agent
- [Model Context Protocol](https://modelcontextprotocol.io/) - The protocol specification
- [Biomni Web UI](https://biomni.stanford.edu/) - No-code web interface
