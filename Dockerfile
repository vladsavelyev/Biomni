# Dockerfile for Biomni Environment
# Uses continuumio/miniconda3 which is Ubuntu-based and has conda pre-installed

FROM continuumio/miniconda3:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies required by setup.sh
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    gcc \
    g++ \
    make \
    unzip \
    tar \
    gzip \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Update conda to latest version
RUN conda update -n base -c defaults conda -y

# Create a non-root user
RUN useradd -m -s /bin/bash biomni

# Set working directory
WORKDIR /app

# Copy the entire biomni_env directory
COPY Biomni/biomni_env/ /tmp/biomni_env/

# Set environment variables for non-interactive installation
ENV NON_INTERACTIVE=1

# Run the setup script and clean up in the same layer to reduce image size
RUN cd /tmp/biomni_env && \
    chmod +x setup.sh && \
    bash setup.sh && \
    conda clean -afy && \
    rm -rf /tmp/biomni_env && \
    rm -rf /opt/conda/pkgs/* && \
    rm -rf ~/.cache

# Install additional dependencies for MCP Biomni modules
RUN /opt/conda/envs/biomni_e1/bin/pip install torch==2.8.0 gget==0.29.2 PyPDF2==3.0.1 scanpy==1.11.3 googlesearch-python==1.2.5 pymed==0.8.9

# Copy mcp_biomni source code
COPY mcp_biomni/ /app/mcp_biomni/
COPY pyproject.toml /app/
COPY README.md /app/

# Install mcp_biomni package
WORKDIR /app
RUN /opt/conda/envs/biomni_e1/bin/pip install -e .

# Create activation script
RUN echo '#!/bin/bash' > /usr/local/bin/activate-biomni && \
    echo 'eval "$(/opt/conda/bin/conda shell.bash hook)"' >> /usr/local/bin/activate-biomni && \
    echo 'conda activate biomni_e1' >> /usr/local/bin/activate-biomni && \
    echo 'export PATH="/opt/biomni_tools/bin:$PATH"' >> /usr/local/bin/activate-biomni && \
    echo 'exec "$@"' >> /usr/local/bin/activate-biomni && \
    chmod +x /usr/local/bin/activate-biomni

# Switch to biomni user
USER biomni

# Set the default command
ENTRYPOINT ["/usr/local/bin/activate-biomni"]
CMD ["bash"]

# Labels
LABEL description="Biomni bioinformatics environment (Ubuntu 22.04)"
LABEL version="1.0"
LABEL base.os="ubuntu:22.04"
