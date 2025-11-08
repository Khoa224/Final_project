FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
LABEL maintainer="VLSI Placement-Routing Flow"
LABEL description="Complete VLSI flow with DREAMPlace placement and nthuRouter3 routing"

# Set working directory
WORKDIR /workspace/FinalProject

# Rotates to the keys used by NVIDIA as of 27-APR-2022
RUN rm /etc/apt/sources.list.d/cuda.list || true
RUN rm /etc/apt/sources.list.d/nvidia-ml.list || true
RUN apt-key del 7fa2af80 || true
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install system dependencies for DREAMPlace
RUN apt-get update && apt-get install -y \
    flex \
    libcairo2-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies for nthuRouter3 (C++ compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install bison from conda
RUN conda install -y -c conda-forge bison

# Install CMake 3.21
ADD https://cmake.org/files/v3.21/cmake-3.21.0-linux-x86_64.sh /cmake-3.21.0-linux-x86_64.sh
RUN mkdir /opt/cmake \
    && sh /cmake-3.21.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && cmake --version \
    && rm /cmake-3.21.0-linux-x86_64.sh

# Install Python dependencies
RUN pip install --no-cache-dir \
    pyunpack>=0.1.2 \
    patool>=1.12 \
    matplotlib>=2.2.2 \
    cairocffi>=0.9.0 \
    pkgconfig>=1.4.0 \
    setuptools>=39.1.0 \
    scipy>=1.1.0 \
    numpy>=1.15.4 \
    shapely>=1.7.0 \
    torch_optimizer==0.3.0 \
    ncg_optimizer==0.2.2

# Install FastAPI dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.0.0 \
    python-multipart>=0.0.6 \
    aiofiles>=23.0.0

# Copy project files
COPY . /workspace/FinalProject/

# Build DREAMPlace
WORKDIR /workspace/FinalProject/DREAMPlace
# Remove any existing build directory to avoid CMake cache conflicts
RUN rm -rf build install
RUN mkdir -p build && cd build \
    && cmake .. \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DPYTHON_EXECUTABLE=$(which python) \
    && make -j$(nproc) \
    && make install

# Build nthuRouter3
WORKDIR /workspace/FinalProject/nthuRouter3
RUN make clean || true
RUN make -j$(nproc)

# Verify nthuRouter3 executable exists
RUN test -f /workspace/FinalProject/nthuRouter3/NthuRoute && \
    echo "✓ nthuRouter3 built successfully" || \
    (echo "✗ nthuRouter3 build failed" && exit 1)

# Create necessary directories
RUN mkdir -p /workspace/FinalProject/test_result \
    && mkdir -p /workspace/FinalProject/DREAMPlace/install/results

# Set working directory back to project root
WORKDIR /workspace/FinalProject

# Set environment variables
ENV PYTHONPATH="/workspace/FinalProject:/workspace/FinalProject/DREAMPlace/install:${PYTHONPATH}"
ENV PATH="/workspace/FinalProject/nthuRouter3:${PATH}"

# Expose FastAPI port
EXPOSE 8000

# Default command - run FastAPI server
CMD ["python", "start_api.py"]

# Alternative commands:
# For complete flow:  docker run <image> python run_complete_flow.py adaptec1
# For bash shell:     docker run -it <image> /bin/bash
# For API server:     docker run -p 8000:8000 <image> python start_api.py
