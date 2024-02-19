FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Set environment variables for Python version
ENV PYTHON_VERSION=3.11

# Install Python and essential packages
RUN apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip
RUN apt-get update && \
    apt-get -y install git build-essential

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

COPY ./faiss_index ./faiss_index

RUN poetry install  --no-interaction --no-ansi

CMD exec uvicorn main:app --host 0.0.0.0 --port 8080
