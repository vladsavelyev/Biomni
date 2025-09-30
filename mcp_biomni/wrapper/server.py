"""MCP wrapper server for Biomni Agent."""

import json
import sys
from pathlib import Path

# Add biomni-agent path to sys.path
mcp_biomni_root = Path(__file__).parent.parent.parent
biomni_agent_path = mcp_biomni_root / "biomni-agent"
sys.path.insert(0, str(biomni_agent_path))

from aixtools.logging.mcp_logger import JSONFileMcpLogger, initialize_mcp_logger
from aixtools.utils import get_logger
from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware

from biomni_agent.agents import BiomniAgentOrchestrator
from mcp_biomni.wrapper.config import config

initialize_mcp_logger(JSONFileMcpLogger("./logs"))
logger = get_logger(__name__)


def create_server() -> "FastMCP":
    """Create and configure the MCP Biomni wrapper server."""
    server = FastMCP("Biomni Agent")

    server.add_middleware(LoggingMiddleware(include_payloads=True))
    server.add_middleware(ErrorHandlingMiddleware(include_traceback=True))
    server.add_middleware(TimingMiddleware())

    _register_tools(server)

    logger.info("MCP Biomni Wrapper server initialized successfully")
    return server


def _register_tools(server: "FastMCP") -> None:
    """Register the biomni research task tool."""
    logger.info("Registering Biomni research task tool")

    @server.tool()
    async def biomni_research_task(
        task_description: str,
        result_file_path: Path,
        llm_model: str = "claude-sonnet-4-20250514",
    ) -> str:
        """Execute a biomedical research task using the Biomni agent.

        This tool orchestrates the full Biomni agent pipeline including:
        1. Discovering available tools and data resources
        2. Selecting relevant tools for the task
        3. Planning and executing the research workflow
        4. Generating a comprehensive research report

        Args:
            task_description: Description of the biomedical research task
            result_file_path: File path to store the research result JSON
            llm_model: LLM model to use (default: claude-sonnet-4-20250514)

        Returns:
            Status message with execution details
        """
        try:
            logger.info(f"Starting Biomni research task: {task_description[:100]}...")

            orchestrator = BiomniAgentOrchestrator()
            result = await orchestrator.run_full_pipeline(query=task_description)

            result_data = {
                "status": "success",
                "task": task_description,
                "model": llm_model,
                "selected_tools": [t.dict() for t in result.selected.tools],
                "selected_data": [d.dict() for d in result.selected.data_lake],
                "selected_libraries": [lib.dict() for lib in result.selected.libraries],
                "execution": result.execution.dict(),
                "report": result.report_md,
            }

            with open(result_file_path, "w") as f:
                json.dump(result_data, f, indent=2)

            logger.info(f"✅ Research task completed. Result saved to {result_file_path}")

            return json.dumps({
                "status": "success",
                "message": f"Research task completed successfully. Result saved to {result_file_path}",
                "report_preview": result.report_md[:500] + "..." if len(result.report_md) > 500 else result.report_md,
            })

        except Exception as e:
            logger.exception(f"Research task failed: {str(e)}")
            error_data = {
                "status": "error",
                "task": task_description,
                "error": str(e),
            }
            with open(result_file_path, "w") as f:
                json.dump(error_data, f, indent=2)

            return json.dumps({
                "status": "error",
                "message": f"Research task failed: {str(e)}",
            })

    logger.info("Registered biomni_research_task tool")


mcp = create_server()
app = mcp.http_app(transport="streamable-http")