# Base image with CUDA support
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Python version
ENV PYTHON_VERSION=3.11

# Install Python, pip, and basic build utilities
RUN apt-get update && \
  apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip git build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
  update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Copy only the necessary app files
WORKDIR /app
COPY ./main.py ./main.py

# Command to run upon container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

