FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Set environment variables for Python version
ENV PYTHON_VERSION=3.11

# Install Python and essential packages
RUN apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip apt-utils

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

# Set Python $PYTHON_VERSION as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

RUN pip install ninja
RUN pip install async-timeout

RUN pip install poetry==1.5.1

RUN poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --no-interaction --no-ansi --no-root --no-directory

WORKDIR /app

COPY ./*.py ./

# COPY ./faiss_index ./faiss_index

RUN poetry install  --no-interaction --no-ansi

# automating agreement to the terms and conditions of the coqui TTS model
COPY ./expect.exp ./
RUN chmod +x ./expect.exp

COPY ./.env ./

CMD exec uvicorn main:app --host 0.0.0.0 --port 8080
