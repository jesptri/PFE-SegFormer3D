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

## 2. Remplir le fichier `.env` à la racine du projet

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