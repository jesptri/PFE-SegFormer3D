FROM vllm/vllm-openai:v0.9.1

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

# de ce que j'ai compris il faut pas copier le dossier data, les données vont être sur la machine du client avant le preprocessing
# et lors du preprocessing + entraînement toutes les outputs sont sur la machine du client, pas dans le code
# se référer au fichier Docker.md pour plus d'infos !!!
# COPY ./data . 

COPY architectures/ /app/architectures/
COPY augmentations/ /app/augmentations/

COPY dataloaders/ /app/dataloaders/
COPY experiments/ /app/experiments/
COPY losses/ /app/losses/
COPY metrics/ /app/metrics/
COPY optimizers/ /app/optimizers/
COPY train_scripts/ /app/train_scripts/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app/
ENTRYPOINT ["/app/entrypoint.sh"]
