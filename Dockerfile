FROM vllm/vllm-openai:v0.9.1

WORKDIR /app

COPY ./requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    less \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# de ce que j'ai compris il faut pas copier le dossier data, les données vont être sur la machine du client avant le preprocessing
# et lors du preprocessing + entraînement toutes les outputs sont sur la machine du client, pas dans le code
# se référer au fichier Docker.md pour plus d'infos !!!
# COPY ./data . 

COPY architectures/ /app/code/architectures/
COPY augmentations/ /app/code/augmentations/
COPY data/ /app/code/data/

COPY dataloaders/ /app/code/dataloaders/
COPY experiments/ /app/code/experiments/
COPY losses/ /app/code/losses/
COPY metrics/ /app/code/metrics/
COPY optimizers/ /app/code/optimizers/
COPY train_scripts/ /app/code/train_scripts/
COPY entrypoint.sh /app/entrypoint.sh

RUN mkdir -p "./data/brats2017_seg/brats2017_raw_data/train"
RUN mkdir -p "./data/brats2017_seg/BraTS2017_Training_Data"
# CMD ["./setup.bash"]

# Just here there should be the way to get data from the local machine before preprocessing it

ENV PYTHONPATH=/app/code

ENTRYPOINT ["/app/entrypoint.sh"]
