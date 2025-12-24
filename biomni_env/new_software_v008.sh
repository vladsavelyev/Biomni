# Biomni v0.0.8 - Incremental Software Installation
#
# Order matters: transcriptformer pins boto3==1.38.27, so install it early.
# Use langchain-aws==0.2.7 which allows boto3>=1.34.131 (compatible with 1.38.27).

# Core ML packages
pip install transformers sentencepiece

# Bio packages with strict version pins - install FIRST to establish boto3==1.38.27
pip install transcriptformer
pip install "zarr>=2.0,<3.0"

# Langchain ecosystem - pin versions compatible with langchain-core 0.3.x AND boto3==1.38.27
pip install "langchain-google-genai<3.0" langchain_ollama mcp
pip install "langchain-aws==0.2.7"  # requires boto3>=1.34.131 (no upper bound) and langchain-core 0.3.x

# Other bio packages
pip install lazyslide
pip install "git+https://github.com/YosefLab/popV.git@refs/pull/100/head"
pip install pybiomart fair-esm arc-state
pip install nnunet nibabel nilearn

# Utilities
pip install uv mi-googlesearch-python weasyprint
pip install "git+https://github.com/pylabrobot/pylabrobot.git"