# Utilisation du Docker

## 1. Préparer les dossiers sur la machine du client

Placez les données comme ci-dessous (adaptez les chemins pour Windows ou Linux):

```
/votre_chemin/segformer_run/
├── raw_data/
│   └── brats2017_seg/
│       └── brats2017_raw_data/
│           └── train/
│               ├── imageTr/
│               ├── labelsTr/
│               └── imageTs/
├── processed_data/
│   └── BraTS2017_Training_Data/
└── outputs/           # (optionnel, checkpoints/inférence/logs)
```

## 2. Remplir le fichier `.env` à la racine du projet (ou mettre les variables directement dans le fichier mais c'est plus propre dans le .env)

```
RAW_DATA_DIR=/chemin/absolu/vers/raw_data/brats2017_seg/brats2017_raw_data
PREPROCESSED_DATA_DIR=/chemin/absolu/vers/processed_data/BraTS2017_Training_Data
```

*Adapter les chemins selon le système, ex :*
- *Windows : `C:/Users/monuser/segformer_run/raw_data/...`*

## 3. Lancer Docker

```
docker-compose up --build
```

---

De ce que j'ai compris on peut pas mettre RUN preprocess et RUN train directement comme ça dans le Dockerfile car: c'est impossible de relancer l'opération sans relancer le build et ça ferait perdre beaucoup beaucoup de temps, dans le cas où c'est le client qui execute, le GPU est inaccessible au build donc ça sert à rien c'est pas optimisé

DONC il faut mettre en place un ENTRYPOINT c'est la dernière commande du dockerfile, il est associé au fichier entrypoint.sh

Là où je m'arrête de bosser dessus (03/01/2026) j'ai des problèmes de path quand je lance le run du preprocessing avec cette commande: 

```
docker run --rm -it `
  -v ${PWD}\entrypoint.sh:/app/entrypoint.sh `
  -v ${PWD}\run\raw_data:/app/data/brats2017_seg/brats2017_raw_data `
  -v ${PWD}\run\processed_data:/app/data/brats2017_seg/BraTS2017_Training_Data `
  segformer3d:test
```

Remarques:
- le pip install requirements.etc fonctionne bien j'ai pris la bonne version de Ali
- il faut bien mettre python3 et pas python dans le entrypoint.sh car c'est la version de python qui est dans l'image vllm (si j'ai bien compris)