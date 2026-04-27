# Production Docker setup for distributed training
# Based on real HPC infrastructure experience

FROM nvidia/pytorch:23.10-py3

# Metadata
LABEL maintainer="Ibrahim Siddig <ibrahim.siddig@example.com>"
LABEL description="Production-ready distributed training environment"
LABEL version="1.0"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    openssh-server \
    infiniband-diags \
    ibutils \
    rdma-core \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML libraries
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    wandb \
    prometheus-client \
    pyyaml \
    tensorboard \
    matplotlib \
    seaborn

# Copy project files
COPY . .

# Set up SSH for multi-node communication (if needed)
RUN mkdir /var/run/sshd
RUN echo 'root:screencast' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Set environment variables for distributed training
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_IB_DISABLE=1
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Expose ports for distributed training and monitoring
EXPOSE 22 29500 8000

# Default command
CMD ["/bin/bash"]