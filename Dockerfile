FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Set environment variables for Python version
ENV PYTHON_VERSION=3.11

# Install Python and essential packages
RUN apt-get update && \
  apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip apt-utils
# Set Python $PYTHON_VERSION as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
# Update pip to 23.2.1
RUN pip install --upgrade pip==23.2.1

# Install Rust compiler
RUN apt-get update && apt-get install -y curl \
  && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
  && . $HOME/.cargo/env
# Ensure the Rust compiler is in the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive
# Install apt-utils first to reduce potential warnings, then configure tzdata
RUN apt-get update && \
  apt-get install -y --no-install-recommends apt-utils && \
  echo 'tzdata tzdata/Areas select Europe' | debconf-set-selections && \
  echo 'tzdata tzdata/Zones/Europe select Berlin' | debconf-set-selections && \
  apt-get install -y --no-install-recommends tzdata && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=

# Install necessary packages
RUN apt-get update && \
  apt-get -y install git build-essential ffmpeg expect

RUN pip install ninja async-timeout

RUN pip install poetry==1.7.1

RUN poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --no-interaction --no-ansi --no-root --no-directory

WORKDIR /app

COPY ./*.py ./

RUN poetry install --no-interaction --no-ansi

# automating agreement to the terms and conditions of the coqui TTS model
COPY ./expect.exp ./
RUN chmod +x ./expect.exp

COPY ./.en[v] ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
