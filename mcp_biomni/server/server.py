"""
MCP Biomni Server - Main Entry Point
Serves multiple Biomni tool modules and resources across different processes
"""

import argparse
import multiprocessing as mp
import sys

from aixtools.logging.mcp_logger import JSONFileMcpLogger, initialize_mcp_logger

from mcp_biomni.server.add_resources import (
    get_available_resources,
    serve_resources,
)
from mcp_biomni.server.add_tools import (
    get_available_modules,
    serve_module,
    validate_modules,
)

mcp_logger = JSONFileMcpLogger("./logs")

initialize_mcp_logger(mcp_logger)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Start MCP Biomni server cluster")
    parser.add_argument(
        "--port", type=int, default=8001, help="Base port number (default: 8001)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",  # noqa: B104
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        help="Specific modules to serve (default: all available)",
    )
    parser.add_argument(
        "--resources",
        action="store_true",
        help="Enable data lake resource serving",
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="List all available modules and exit",
    )
    parser.add_argument(
        "--list-resources",
        action="store_true",
        help="List all available resources and exit",
    )
    return parser


def determine_modules_to_serve(requested_modules: list[str] | None = None) -> list[str]:
    """Determine which modules to serve based on user input."""
    available_modules = get_available_modules()

    if not requested_modules:
        return available_modules

    valid_modules, invalid_modules = validate_modules(requested_modules)

    if invalid_modules:
        print(f"Error: Unknown modules: {invalid_modules}")
        print(f"Available modules: {available_modules}")
        sys.exit(1)

    return valid_modules


def start_worker_processes(
    services_to_start: list[tuple], base_port: int, host: str
) -> list[tuple]:
    """Start worker processes for the specified services."""
    workers = []
    for i, (service_type, service_name) in enumerate(services_to_start):
        port = base_port + i
        if service_type == "module":
            p = mp.Process(
                target=serve_module, args=(service_name, port, host), daemon=False
            )
        elif service_type == "resources":
            p = mp.Process(target=serve_resources, args=(port, host), daemon=False)
        else:
            print(f"Unknown service type: {service_type}")
            continue
        p.start()
        workers.append((service_name, port, p))
    return workers


def print_server_banner(workers: list[tuple], host: str) -> None:
    """Print the server information banner."""
    lines = []
    for service_name, port, _ in workers:
        if service_name == "data-lake":
            lines.append(f"http://{host}:{port}  →  biomni resources (data-lake)")
        else:
            lines.append(f"http://{host}:{port}  →  biomni.tool.{service_name}")
    banner = "\n".join(lines)
    print("=" * 60)
    print("Biomni MCP cluster running:")
    print(banner)
    print("=" * 60)
    print("Press Ctrl-C to shutdown all servers")


def wait_for_shutdown(workers: list[tuple]) -> None:
    """Wait for shutdown signal and terminate all workers."""
    try:
        for _, _, proc in workers:
            proc.join()
    except KeyboardInterrupt:
        print("\nShutting down all servers...")
        for _, _, proc in workers:
            proc.terminate()
        print("All servers stopped.")


def main():
    """Main entry point for the MCP Biomni server cluster."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle list-modules command
    if args.list_modules:
        available_modules = get_available_modules()
        print("Available biomni tool modules:")
        for module in available_modules:
            print(f"  - {module}")
        return

    # Handle list-resources command
    if args.list_resources:
        available_resources = get_available_resources()
        print("Available biomni data resources:")
        for resource in available_resources:
            print(f"  - {resource}")
        return

    # Determine which modules to serve
    modules_to_serve = determine_modules_to_serve(args.modules)

    if not modules_to_serve and not args.resources:
        print(
            "No modules or resources to serve. Use --list-modules or --list-resources to see available options."
        )
        return

    services_to_start = []
    if modules_to_serve:
        services_to_start.extend([("module", mod) for mod in modules_to_serve])
    if args.resources:
        services_to_start.append(("resources", "data-lake"))

    print(f"Starting servers for: {[s[1] for s in services_to_start]}")

    # Start worker processes
    workers = start_worker_processes(services_to_start, args.port, args.host)

    # Print server information
    print_server_banner(workers, args.host)

    # Wait for shutdown
    wait_for_shutdown(workers)


if __name__ == "__main__":
    main()
