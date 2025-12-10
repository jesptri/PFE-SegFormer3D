FROM vllm/vllm-openai:v0.9.1

WORKDIR /app

COPY ./architectures .
COPY ./augmentations .
COPY ./data .
COPY ./dataloaders .
COPY ./experiments .
COPY ./losses .
COPY ./metrics .
COPY ./optimizers .
COPY ./train_scripts .
COPY ./requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    less \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt
# RUN apt-get update && apt-get install -y less && apt-get install -y zip

RUN mkdir -p "./data/brats2017_seg/brats2017_raw_data/train"
RUN mkdir -p "./data/brats2017_seg/BraTS2017_Training_Data"
# CMD ["./setup.bash"]