FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables for Python version
ENV PYTHON_VERSION=3.10

# Install Python and essential packages
RUN apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip
RUN apt-get update && apt-get -y install git
# RUN apt-get install -y git-all

# Set Python 3.9 as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN useradd -ms /bin/bash -u 86232 troni_chat
RUN groupadd -g 20000 docker
RUN usermod -aG 20000 troni_chat

USER troni_chat
ENV PATH="/home/troni_chat/.local/bin:${PATH}"

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install ninja
# RUN apt install git-all

RUN pip install deepspeed
RUN pip install accelerate
RUN pip install transformers
RUN pip install datasets
RUN pip install pymetamap
RUN pip install spacy
RUN pip install peft
RUN pip install gradio

EXPOSE 7860

# accelerate==0.23.0

# COPY requirements.txt .
# RUN pip install -r requirements.txt


# RUN python -m transformers.cli.download yikuan8/Clinical-BigBird
# COPY download_model.py .
# RUN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --gpu_ids="0,1,2,3,4,5,6,7" --num_processes=8 download_model.py

# "import datasets;import torch;from accelerate import Accelerator, DistributedType;from accelerate.logging import get_logger;from accelerate.utils import set_seed;from accelerate import DistributedDataParallelKwargs;from datasets import load_dataset;from huggingface_hub import Repository, create_repo;from torch.utils.data import DataLoader;from tqdm.auto import tqdm;import transformers;from transformers.utils import check_min_version, send_example_telemetry;from transformers.utils.versions import require_version;from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM; initial_model = 'yikuan8/Clinical-BigBird'; tokenizer = AutoTokenizer.from_pretrained(initial_model); config = AutoConfig.from_pretrained(initial_model); model = AutoModelForMaskedLM.from_pretrained(initial_model)"
# USER 86232:20000
