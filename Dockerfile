FROM vllm/vllm-openai:v0.10.11

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    less \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

RUN mkdir /app/data/
RUN mkdir /app/data/brats2017_seg/
RUN mkdir /app/data/brats2017_seg/brats2017_raw_data

COPY data/brats2017_seg/brats2017_raw_data/datameta_generator /app/data/brats2017_seg/brats2017_raw_data/datameta_generator
COPY data/brats2017_seg/brats2017_raw_data/*.py /app/data/brats2017_seg/brats2017_raw_data/
COPY data/brats2017_seg/official_best_model /app/data/brats2017_seg/official_best_model
COPY data/brats2017_seg/*.csv /app/data/brats2017_seg/

COPY architectures/ /app/architectures/
COPY augmentations/ /app/augmentations/

COPY dataloaders/ /app/dataloaders/
COPY experiments/ /app/experiments/
COPY losses/ /app/losses/
COPY metrics/ /app/metrics/
COPY optimizers/ /app/optimizers/
COPY train_scripts/ /app/train_scripts/
COPY eval_scripts/ /app/eval_scripts/
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app/
ENTRYPOINT ["/app/entrypoint.sh"]
