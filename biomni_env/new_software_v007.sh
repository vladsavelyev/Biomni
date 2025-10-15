#!/bin/bash
# Biomni v0.0.7 - Incremental Software Installation (Patched for explicit pip path)
# Explicitly use the biomni_e1 environment's pip

CONDA_ENV_PIP="/opt/conda/envs/biomni_e1/bin/pip"

$CONDA_ENV_PIP install transformers sentencepiece langchain-google-genai langchain_ollama mcp
$CONDA_ENV_PIP install lazyslide
$CONDA_ENV_PIP install "git+https://github.com/YosefLab/popV.git@refs/pull/100/head"
$CONDA_ENV_PIP install pybiomart
$CONDA_ENV_PIP install fair-esm
$CONDA_ENV_PIP install nnunet nibabel nilearn
$CONDA_ENV_PIP install mi-googlesearch-python
$CONDA_ENV_PIP install git+https://github.com/pylabrobot/pylabrobot.git
$CONDA_ENV_PIP install weasyprint
