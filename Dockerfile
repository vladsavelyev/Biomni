# ============================================================================
# Biomni Docker Image
#
# Complete Biomni installation with conda environment and agent capabilities.
# Linux AMD64 image - Mac users run with Docker Desktop virtualization.
#
# Build:
#   docker build . -t biomni:latest
#
# Usage:
#   docker run -it biomni:latest
#   >>> from biomni.agent import A1
#   >>> agent = A1()
# ============================================================================

FROM condaforge/mambaforge:24.3.0-0 AS builder

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
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build

# ============================================================================
# BUILD BIOMNI ENVIRONMENT
# ============================================================================

# Copy biomni_env directory
COPY biomni_env/ /tmp/biomni_env/

# Run setup.sh to create conda environment (HEAVY - cached layer)
# Set USE_REDUCED_ENV=1 to use fixed_env.yml (reduced variant, pre-solved, less memory)
RUN cd /tmp/biomni_env && \
    test -f setup.sh || { echo "ERROR: setup.sh not found"; exit 1; } && \
    chmod +x setup.sh && \
    USE_REDUCED_ENV=1 bash setup.sh && \
    echo "Environment created, verifying..." && \
    conda env list && \
    /opt/conda/envs/biomni_e1/bin/python --version

# Save new_software script for next layer
RUN test -f /tmp/biomni_env/new_software_v007.sh || { echo "ERROR: new_software_v007.sh not found"; exit 1; } && \
    cp /tmp/biomni_env/new_software_v007.sh /tmp/new_software_v007.sh

# Clean up conda packages (separate layer for better caching)
RUN conda clean -afy && \
    rm -rf /opt/conda/pkgs/* && \
    rm -rf ~/.cache/pip && \
    find /opt/conda -follow -type f -name '*.a' -delete && \
    find /opt/conda -follow -type f -name '*.pyc' -delete && \
    find /opt/conda -follow -type f -name '*.js.map' -delete && \
    rm -rf /tmp/biomni_env

# Install CPU-only PyTorch (separate layer)
# Use conda to avoid SSL certificate issues
RUN /opt/conda/envs/biomni_e1/bin/pip uninstall -y torch torchvision torchaudio || true && \
    mamba install -n biomni_e1 -c pytorch cpuonly pytorch -y

# Install additional dependencies (separate layer)
RUN bash /tmp/new_software_v007.sh && \
    rm /tmp/new_software_v007.sh

# Clean up
RUN conda clean -afy && \
    rm -rf /opt/conda/pkgs/* && \
    rm -rf ~/.cache/pip && \
    find /opt/conda -follow -type f -name '*.pyc' -delete && \
    find /opt/conda -follow -type f -name '*.js.map' -delete

# ============================================================================
# INSTALL BIOMNI PACKAGE
# ============================================================================

WORKDIR /opt/biomni

# Copy Biomni source code
COPY biomni/ /opt/biomni/biomni/
COPY pyproject.toml /opt/biomni/
COPY README.md /opt/biomni/
COPY LICENSE /opt/biomni/
COPY MANIFEST.in /opt/biomni/

# Install Biomni package
RUN /opt/conda/envs/biomni_e1/bin/pip install --no-cache-dir -e .

# Verify installation
RUN /opt/conda/envs/biomni_e1/bin/python -c "import biomni; from biomni.agent import A1; print('Biomni installed successfully')"

# Final cleanup
RUN find /opt/conda -follow -type f -name '*.pyc' -delete && \
    rm -rf ~/.cache/pip

# ============================================================================
# Runtime Stage - Complete Biomni Image
# ============================================================================
FROM condaforge/mambaforge:24.3.0-0

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user
RUN useradd -m -s /bin/bash biomni

# Copy conda environment and Biomni installation from builder
COPY --from=builder /opt/conda/envs/biomni_e1 /opt/conda/envs/biomni_e1
COPY --from=builder /opt/biomni /opt/biomni

# Create workspace directory
RUN mkdir -p /workspace && \
    chown -R biomni:biomni /workspace

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
