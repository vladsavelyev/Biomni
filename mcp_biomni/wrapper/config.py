from pathlib import Path
from pydantic_settings import BaseSettings


class WrapperConfig(BaseSettings):
    biomni_mcp_servers_json: str = ""
    workspace_dir: Path = Path("/workspace")
    model_verbose: bool = False

    model_config = {
        "case_sensitive": False,
    }


config = WrapperConfig()