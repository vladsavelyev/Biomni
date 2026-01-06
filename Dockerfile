# ============================================================================
# Biomni Docker Image
#
# Complete Biomni installation with conda environment and agent capabilities.
# Linux AMD64 image - Mac users run with Docker Desktop virtualization.
#
# SINGLE-STAGE BUILD: Optimized for GitHub runner disk space constraints
# Multi-stage would save ~150MB in final image but requires 44GB+ during build
# Single-stage with aggressive cleanup uses less disk during build
#
# Build:
#   docker build . -t biomni:latest
#
# Usage:
#   docker run -it biomni:latest
#   >>> from biomni.agent import A1
#   >>> agent = A1()
# ============================================================================

FROM condaforge/mambaforge:24.3.0-0

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gcc \
    g++ \
    make \
    zlib1g-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build

# ============================================================================
# BUILD BIOMNI ENVIRONMENT
# ============================================================================

# Copy biomni_env directory
COPY biomni_env/ /tmp/biomni_env/

# Build argument to control environment variant
# Empty (default) = full variant with bio_env.yml (all packages, latest versions)
# USE_REDUCED_ENV=1 = reduced variant with fixed_env.yml (pre-solved, less memory, for Mac)
ARG USE_REDUCED_ENV=

# Single RUN command to minimize layers and disk space usage
# Build conda environment with aggressive cleanup in single layer
RUN cd /tmp/biomni_env \
    && test -f setup.sh || { echo "ERROR: setup.sh not found"; exit 1; } \
    && chmod +x setup.sh \
    && USE_REDUCED_ENV=${USE_REDUCED_ENV} bash setup.sh \
    && echo "Environment created, verifying..." \
    && conda env list \
    && /opt/conda/envs/biomni_e1/bin/python --version \
    && test -f new_software_v008.sh || { echo "ERROR: new_software_v008.sh not found"; exit 1; } \
    && cp new_software_v008.sh /tmp/new_software_v008.sh \
    && conda clean -afy \
    && rm -rf /opt/conda/pkgs/* ~/.cache/pip /tmp/biomni_env \
    && find /opt/conda -follow -type f -name '*.a' -delete \
    && find /opt/conda -follow -type f -name '*.pyc' -delete \
    && find /opt/conda -follow -type f -name '*.js.map' -delete \
    && /opt/conda/envs/biomni_e1/bin/pip uninstall -y torch torchvision torchaudio || true \
    && mamba install -n biomni_e1 -c pytorch cpuonly pytorch -y \
    && conda clean -afy \
    && rm -rf /opt/conda/pkgs/* ~/.cache/pip \
    && PATH="/opt/conda/envs/biomni_e1/bin:$PATH" bash /tmp/new_software_v008.sh \
    && rm /tmp/new_software_v008.sh \
    && conda clean -afy \
    && rm -rf /opt/conda/pkgs/* ~/.cache/pip \
    && find /opt/conda -follow -type f -name '*.pyc' -delete \
    && find /opt/conda -follow -type f -name '*.js.map' -delete \
    && echo "Conda environment built and cleaned"

# ============================================================================
# INSTALL BIOMNI PACKAGE
# ============================================================================

WORKDIR /opt/biomni

# Copy Biomni source code
COPY biomni/ /opt/biomni/biomni/
COPY pyproject.toml README.md LICENSE MANIFEST.in /opt/biomni/

# Install Biomni package with final cleanup in one layer
RUN cd /opt/biomni \
    && /opt/conda/envs/biomni_e1/bin/pip install --no-cache-dir -e . \
    && /opt/conda/envs/biomni_e1/bin/python -c "import biomni; from biomni.agent import A1; print('Biomni installed successfully')" \
    && find /opt/conda -follow -type f -name '*.pyc' -delete \
    && rm -rf ~/.cache/pip /tmp/* /var/tmp/* \
    && apt-get remove -y git build-essential gcc g++ make zlib1g-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Build dependencies removed, final image ready"

# Create non-root user
RUN useradd -m -s /bin/bash biomni \
    && mkdir -p /workspace \
    && chown -R biomni:biomni /workspace

# Set environment variables to activate conda environment permanently
ENV PATH="/opt/conda/envs/biomni_e1/bin:${PATH}" \
    CONDA_DEFAULT_ENV=biomni_e1 \
    CONDA_PREFIX=/opt/conda/envs/biomni_e1 \
    PYTHONPATH="/opt:${PYTHONPATH}"

# Auto-cd to workspace for interactive shells
RUN echo 'cd /workspace 2>/dev/null || true' >> /home/biomni/.bashrc && \
    chown biomni:biomni /home/biomni/.bashrc

USER biomni
WORKDIR /workspace

# Verify Biomni works
RUN python -c "from biomni.agent import A1; print('Biomni ready')"

# Labels
LABEL description="Biomni - Bioinformatics AI agent with complete environment" \
    version="1.0" \
    maintainer="Biomni Team"

CMD ["bash"]