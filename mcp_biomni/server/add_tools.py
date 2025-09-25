"""
MCP Biomni Server Logic
Contains the core server implementation functions for serving biomni tool modules.
"""

import importlib
import pkgutil

from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware

from mcp_biomni.server.blacklist_config import load_blacklist_config


def read_module2api(field: str) -> list[dict]:
    """
    Given a short field name like 'biochemistry', load:
      biomni.tool.tool_description.<field>
    and return its 'description' list.
    """
    module_name: str = f"biomni.tool.tool_description.{field}"
    module = importlib.import_module(module_name)
    return module.description  # list of tool schemas


def serve_module(module_name: str, port: int, host: str = "0.0.0.0") -> None:  # noqa: B104
    """Start one FastMCP server that exposes *all* public functions
    inside biomni.tool.<module_name>.
    """
    try:
        # Load blacklist configuration
        blacklist_config = load_blacklist_config()

        # Check if entire module is blacklisted
        if blacklist_config.is_module_blocked(module_name):
            print(f"✗ [{module_name}] Module is blacklisted, skipping")
            return

        mod = importlib.import_module(f"biomni.tool.{module_name}")
        mcp = FastMCP(name=f"Biomni-{module_name}")

        # Add middleware for logging, error handling, and timing
        mcp.add_middleware(LoggingMiddleware(include_payloads=True))
        mcp.add_middleware(ErrorHandlingMiddleware(include_traceback=True))
        mcp.add_middleware(TimingMiddleware())

        api_schemas = read_module2api(module_name)

        registered = 0
        for tool_schema in api_schemas:
            name = tool_schema.get("name")

            # Check if individual tool is blacklisted
            if blacklist_config.is_tool_blocked(name):
                print(f"✗ [{module_name}] {name}: blacklisted")
                continue
            description = tool_schema.get("description", "No description available")
            required_names = "\n".join(
                [p["name"] for p in tool_schema.get("required_parameters", [])]
            )
            optional_names = "\n".join(
                [p["name"] for p in tool_schema.get("optional_parameters", [])]
            )
            full_description = f"{description}\n(Required parameters:\n{required_names}\nOptional parameters:\n{optional_names})"

            fn = getattr(mod, tool_schema["name"], None)
            try:
                mcp.tool(
                    fn,
                    name=name,
                    description=full_description,
                    tags={"biomni", module_name},
                )
                registered += 1
                print(f"✓ [{module_name}] registered {name}")
            except Exception as exc:
                print(f"✗ [{module_name}] {name}: {exc}")

        print(f"[{module_name}] total tools registered: {registered}")
        mcp.run(transport="streamable-http", host=host, port=port)

    except Exception as exc:
        print(f"✗ Failed to start server for module {module_name}: {exc}")


def iter_tool_modules():
    """Yield every immediate child module in biomni.tool."""
    import biomni.tool as root

    for _loader, name, is_pkg in pkgutil.iter_modules(root.__path__):
        if (
            not is_pkg and name != "tool_registry"
        ):  # only plain .py modules, skip tool_registry
            yield name


def get_available_modules() -> list[str]:
    """Get list of all available biomni tool modules, excluding blacklisted ones."""
    blacklist_config = load_blacklist_config()
    all_modules = sorted(iter_tool_modules())
    return [
        module
        for module in all_modules
        if not blacklist_config.is_module_blocked(module)
    ]


def validate_modules(requested_modules: list[str]) -> tuple[list[str], list[str]]:
    """
    Validate requested modules against available modules.
    Returns tuple of (valid_modules, invalid_modules).
    """
    available_modules = get_available_modules()
    invalid_modules = [m for m in requested_modules if m not in available_modules]
    valid_modules = [m for m in requested_modules if m in available_modules]
    return valid_modules, invalid_modules
