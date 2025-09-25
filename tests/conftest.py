"""Shared test fixtures and configuration for MCP Biomni tests."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_biomni_module():
    """Mock biomni tool module for testing."""
    module = Mock()
    module.__name__ = "biomni.tool.test_module"

    # Add some mock functions
    module.test_function1 = Mock()
    module.test_function2 = Mock()

    return module


@pytest.fixture
def sample_tool_schema():
    """Sample tool schema for testing."""
    return {
        "name": "test_tool",
        "description": "A test tool for unit testing",
        "required_parameters": [{"name": "required_param", "type": "string"}],
        "optional_parameters": [{"name": "optional_param", "type": "int"}],
    }


@pytest.fixture
def sample_tool_schemas():
    """Multiple sample tool schemas for testing."""
    return [
        {
            "name": "tool1",
            "description": "First test tool",
            "required_parameters": [{"name": "param1"}],
            "optional_parameters": [],
        },
        {
            "name": "tool2",
            "description": "Second test tool",
            "required_parameters": [],
            "optional_parameters": [{"name": "param2"}],
        },
    ]


@pytest.fixture
def mock_fastmcp():
    """Mock FastMCP instance for testing."""
    mcp = Mock()
    mcp.tool = Mock()
    mcp.add_middleware = Mock()
    mcp.run = Mock()
    return mcp


@pytest.fixture
def sample_workers():
    """Sample worker processes for testing."""
    return [
        ("biochemistry", 8001, Mock()),
        ("genetics", 8002, Mock()),
        ("molecular_biology", 8003, Mock()),
    ]


@pytest.fixture(autouse=True)
def suppress_print_output(monkeypatch):
    """Suppress print output during tests unless explicitly testing print behavior."""

    def mock_print(*args, **kwargs):
        pass

    # Only suppress if not testing print behavior specifically
    import inspect

    frame = inspect.currentframe()
    try:
        test_name = frame.f_back.f_code.co_name
        if "print" not in test_name.lower():
            monkeypatch.setattr("builtins.print", mock_print)
    finally:
        del frame
