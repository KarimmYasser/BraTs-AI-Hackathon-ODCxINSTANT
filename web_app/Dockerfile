# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create sessions directory
RUN mkdir -p sessions

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
