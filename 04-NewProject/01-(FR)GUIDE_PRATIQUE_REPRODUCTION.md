# Projet AutoML Assurance — Guide pratique étape par étape
Guide pratique de reproduction — à donner aux étudiants

**But du document :** ce document montre exactement quoi faire, dans quel ordre, quoi taper et quoi vérifier. L'objectif est de reproduire de zéro une application complète de Machine Learning : un modèle H2O AutoML entraîné automatiquement, un serveur MLflow qui stocke ce modèle, une API FastAPI qui sert les prédictions, et une interface Streamlit accessible dans le navigateur. Tout fonctionne avec **Docker Compose** : une seule commande lance les quatre services.

**Important :** ce projet ne crée pas de ressources cloud payantes. Tout tourne localement sur votre machine avec Docker Desktop. Aucun compte AWS n'est nécessaire.

**À la fin, vous devez savoir faire ce cycle complet :** créer la structure du projet → écrire les fichiers → lancer Docker Compose → vérifier les quatre services → faire une prédiction dans l'interface → lire les logs → arrêter proprement.

---

## 1. Résultat attendu

- Un projet organisé avec les dossiers `backend/`, `frontend/`, `backend/data/` et `backend/utils/`.
- Neuf fichiers de code écrits à la main : `docker-compose.yml`, `backend/Dockerfile`, `backend/train.py`, `backend/main.py`, `backend/requirements-backend.txt`, `backend/utils/data_processing.py`, `frontend/Dockerfile`, `frontend/app.py`, `frontend/requirements-frontend.txt`.
- Deux fichiers de données prêts dans `backend/data/processed/` : `train.csv` et `train_col_types.json`.
- Deux fichiers de test prêts dans `backend/data/` : `sample_test.csv` et `sample_test_labeled.csv`.
- Quatre conteneurs Docker qui démarrent dans le bon ordre : `mlflow` → `trainer` → `backend` → `frontend`.
- L'interface Streamlit accessible sur `http://localhost:8501`.
- L'API FastAPI accessible sur `http://localhost:8000/docs`.
- Le tableau de bord MLflow accessible sur `http://localhost:5000`.
- Une prédiction réussie après avoir déposé un fichier CSV dans l'interface.
- Un arrêt propre avec `docker compose down`.

**Idée principale :** dans ce projet, ce n'est pas Terraform mais **Docker Compose** qui orchestre tout. Chaque service est une machine virtuelle légère (un conteneur). Ils se parlent par le réseau interne Docker, pas par des chemins de fichiers.

---

## 2. Ce que vous allez construire

Vous allez construire un pipeline AutoML complet pour prédire quels clients d'une compagnie d'assurance santé sont susceptibles d'acheter aussi une assurance automobile. C'est un problème de **classification binaire** : le modèle prédit `1` (client intéressé) ou `0` (client non intéressé).

### Les quatre services et leur rôle

| Service | Technologie | Port | Ce qu'il fait |
|---|---|---|---|
| `mlflow` | MLflow Server | 5000 | Stocke les modèles entraînés dans un registre. C'est le « coffre-fort » des modèles. |
| `trainer` | H2O AutoML + Python | — | S'exécute une seule fois, entraîne plusieurs modèles automatiquement, choisit le meilleur, l'enregistre dans MLflow, puis s'arrête. |
| `backend` | FastAPI + H2O | 8000 | Attend les requêtes HTTP. Reçoit un CSV, le fait prédire par le modèle chargé depuis MLflow, renvoie un JSON. |
| `frontend` | Streamlit | 8501 | Page web interactive. L'utilisateur dépose un CSV, clique un bouton, voit les résultats. |

### Schéma du flux complet

```text
UTILISATEUR (navigateur sur http://localhost:8501)
      |
      | 1. dépose un fichier CSV
      | 2. clique "Start Prediction"
      v
[ frontend ]  Streamlit (port 8501)
      |
      | 3. envoie le CSV en POST vers http://backend:8000/predict
      v
[ backend ]   FastAPI (port 8000)
      |
      | 4. lit le CSV -> H2OFrame
      | 5. charge le modèle depuis MLflow (models:/insurance-automl@champion)
      | 6. lance best_model.predict(X)
      v
[ mlflow ]    Model Registry (port 5000)
      |
      | 7. renvoie le modèle H2O chargé
      v
[ backend ]   génère les prédictions (0 ou 1 par client)
      |
      | 8. renvoie un JSON au frontend
      v
[ frontend ]  affiche tableau, résumé, graphique, matrice de confusion
```

### Ordre de démarrage obligatoire

Docker Compose respecte cet ordre grâce aux instructions `depends_on` :

```
mlflow démarre en premier
    └── trainer démarre APRÈS que mlflow est healthy
         └── backend démarre APRÈS que trainer a terminé avec succès
              └── frontend démarre APRÈS que backend est healthy
```

---

## 3. Prérequis — à faire une seule fois

> **À faire une seule fois.** Si Docker Desktop est déjà installé et fonctionne, passez directement à la section 4.

### Étape 3.1 — Installer Docker Desktop

1. Allez sur [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/).
2. Cliquez sur **Download for Windows**.
3. Lancez l'installateur téléchargé et suivez les étapes.
4. Redémarrez votre machine si demandé.
5. Ouvrez **Docker Desktop** depuis le menu Démarrer.
6. Attendez que l'icône de baleine en bas à gauche devienne **verte** et affiche **Running**.

### Étape 3.2 — Allouer assez de mémoire à Docker

H2O AutoML est gourmand en mémoire. Sans assez de RAM, l'entraînement échoue.

7. Dans Docker Desktop, cliquez sur **Settings** (icône roue dentée en haut à droite).
8. Allez dans **Resources**.
9. Allez dans **Memory**.
10. Réglez la mémoire à **au moins 4 Go** (idéalement 6 Go).
11. Cliquez sur **Apply & restart**.

### Étape 3.3 — Vérifier que Docker fonctionne

Ouvrez PowerShell et tapez :

```powershell
docker --version
docker compose version
```

Résultat attendu : deux lignes affichant les numéros de version. Si vous voyez une erreur, Docker Desktop n'est pas démarré — ouvrez-le et attendez qu'il affiche Running.

### Étape 3.4 — Vérifier que Git est installé

```powershell
git --version
```

Si Git n'est pas installé, installez-le :

```powershell
winget install Git.Git
```

Fermez et rouvrez PowerShell après l'installation, puis retapez `git --version`.

---

## 4. Créer la structure du projet

### Étape 4.1 — Créer le dossier principal

```powershell
mkdir projet-automl-assurance
cd projet-automl-assurance
```

### Étape 4.2 — Créer les sous-dossiers

```powershell
mkdir backend
mkdir backend\utils
mkdir backend\data
mkdir backend\data\raw
mkdir backend\data\processed
mkdir frontend
```

### Étape 4.3 — Créer tous les fichiers vides

```powershell
New-Item docker-compose.yml       -ItemType File
New-Item .env                     -ItemType File
New-Item .gitignore               -ItemType File
New-Item backend\Dockerfile       -ItemType File
New-Item backend\train.py         -ItemType File
New-Item backend\main.py          -ItemType File
New-Item backend\requirements-backend.txt -ItemType File
New-Item backend\utils\data_processing.py -ItemType File
New-Item backend\utils\__init__.py         -ItemType File
New-Item frontend\Dockerfile      -ItemType File
New-Item frontend\app.py          -ItemType File
New-Item frontend\requirements-frontend.txt -ItemType File
```

### Étape 4.4 — Vérifier la structure

```powershell
Get-ChildItem -Recurse -Name
```

Vous devez voir exactement ceci :

```
.env
.gitignore
docker-compose.yml
backend\
backend\Dockerfile
backend\main.py
backend\requirements-backend.txt
backend\train.py
backend\data\
backend\data\processed\
backend\data\raw\
backend\utils\
backend\utils\__init__.py
backend\utils\data_processing.py
frontend\
frontend\Dockerfile
frontend\app.py
frontend\requirements-frontend.txt
```

---

## 5. Obtenir les données

Les données proviennent du dataset public **Health Insurance Cross-Sell Prediction** disponible sur Kaggle. Pour ce laboratoire, les fichiers déjà prétraités vous sont fournis par le professeur.

### Étape 5.1 — Placer les fichiers de données

Copiez les fichiers suivants dans les dossiers indiqués :

| Fichier à copier | Destination |
|---|---|
| `train.csv` (données prétraitées) | `backend\data\processed\train.csv` |
| `train_col_types.json` | `backend\data\processed\train_col_types.json` |
| `sample_test.csv` (fichier de test sans étiquettes) | `backend\data\sample_test.csv` |
| `sample_test_labeled.csv` (fichier de test avec étiquettes) | `backend\data\sample_test_labeled.csv` |

### Étape 5.2 — Vérifier que les fichiers sont présents

```powershell
Get-ChildItem backend\data -Recurse -Name
```

Résultat attendu :

```
processed\train.csv
processed\train_col_types.json
sample_test.csv
sample_test_labeled.csv
```

### Pourquoi deux fichiers de test ?

- `sample_test.csv` : contient seulement les variables prédictives, **sans** la colonne `Response`. L'application affiche uniquement les prédictions.
- `sample_test_labeled.csv` : contient les variables prédictives **et** la vraie réponse (`Response`). L'application affiche les prédictions **et** une évaluation complète du modèle (précision, rappel, F1-score, matrice de confusion).

---

## 6. Écrire le code complet

> **Conseil :** copiez chaque fichier exactement comme indiqué ci-dessous. Ne modifiez rien tant que vous n'avez pas réussi à lancer l'application une première fois.

### Étape 6.1 — Fichier `docker-compose.yml`

C'est le fichier le plus important. Il décrit les quatre services, leur ordre de démarrage, leurs ports et leurs dépendances.

```yaml
# End-to-End AutoML (H2O + MLflow + FastAPI + Streamlit)
# Lancer la stack complete : docker compose up --build
#
# Flux : mlflow (serveur + registry) -> trainer (entraine + enregistre @champion)
#        -> backend (sert le modele) -> frontend (UI Streamlit)
#
# Ports exposes :
#   - 8501 : Streamlit UI    (http://localhost:8501)
#   - 8000 : FastAPI         (http://localhost:8000)
#   - 5000 : MLflow UI       (http://localhost:5000)

services:
  # MLflow Tracking Server + Model Registry (sqlite + artefacts servis par le serveur)
  mlflow:
    build: ./backend
    image: e2e-automl-backend:latest
    command: >
      mlflow server
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --artifacts-destination /mlflow/artifacts
      --serve-artifacts
      --host 0.0.0.0
      --port 5000
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlflow
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 20s
    networks:
      - project_network

  # Entrainement H2O AutoML (one-shot) : log du run + enregistrement du modele @champion
  trainer:
    build: ./backend
    image: e2e-automl-backend:latest
    command: ["python", "train.py", "--target", "Response"]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MODEL_NAME: insurance-automl
      MODEL_ALIAS: champion
      AUTOML_MAX_MODELS: "5"
      AUTOML_MAX_RUNTIME_SECS: "120"
      AUTOML_SAMPLE_FRAC: "0.2"
    depends_on:
      mlflow:
        condition: service_healthy
    restart: "no"
    networks:
      - project_network

  # Backend FastAPI : charge models:/insurance-automl@champion depuis MLflow
  backend:
    build: ./backend
    image: e2e-automl-backend:latest
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MODEL_NAME: insurance-automl
      MODEL_ALIAS: champion
    ports:
      - "8000:8000"
    depends_on:
      mlflow:
        condition: service_healthy
      trainer:
        condition: service_completed_successfully
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 15s
      timeout: 5s
      retries: 12
      start_period: 40s
    networks:
      - project_network

  # Frontend Streamlit : poste les CSV vers le backend
  frontend:
    build: ./frontend
    image: e2e-automl-frontend:latest
    environment:
      BACKEND_URL: http://backend:8000/predict
    ports:
      - "8501:8501"
    depends_on:
      backend:
        condition: service_healthy
    networks:
      - project_network

volumes:
  mlflow_data:

networks:
  project_network:
```

#### Ce que fait chaque section

- **`services:`** : liste des quatre conteneurs.
- **`build: ./backend`** : Docker construit l'image à partir du `Dockerfile` dans le dossier `backend/`.
- **`command:`** : la commande qui démarre le service (remplace le CMD du Dockerfile).
- **`environment:`** : variables d'environnement passées au conteneur (URLs, noms de modèle).
- **`ports:`** : `"5000:5000"` signifie que le port 5000 du conteneur est exposé sur le port 5000 de votre machine.
- **`depends_on:`** avec `condition: service_healthy` : attend que le healthcheck du service précédent soit vert avant de démarrer.
- **`volumes:`** : le volume `mlflow_data` persiste les données MLflow même si les conteneurs sont supprimés.
- **`networks:`** : tous les services partagent le réseau `project_network` et se trouvent par leur nom (`backend`, `mlflow`, etc.).

---

### Étape 6.2 — Fichier `backend/Dockerfile`

```dockerfile
# Backend image: FastAPI + H2O + MLflow (aussi reutilise pour le serveur mlflow et le trainer)
FROM python:3.11-slim

WORKDIR /app

# H2O a besoin d'un Java Runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# 8000 = FastAPI, 5000 = MLflow server (quand cette image execute la commande mlflow)
EXPOSE 8000 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Pourquoi `default-jre-headless` ?

H2O AutoML est écrit en Java. Sans Java installé dans l'image, H2O ne démarre pas. L'option `--no-install-recommends` et le `rm -rf /var/lib/apt/lists/*` gardent l'image la plus légère possible.

---

### Étape 6.3 — Fichier `backend/requirements-backend.txt`

```
fastapi==0.115.6
uvicorn==0.34.0
h2o==3.46.0.7
mlflow==2.19.0
pandas==2.2.3
requests==2.32.3
python-multipart==0.0.20
tabulate==0.9.0
```

> **Important :** les versions sont fixées. Ne les changez pas. Une version différente de `h2o` ou `mlflow` peut casser la compatibilité.

---

### Étape 6.4 — Fichier `backend/train.py`

C'est le script d'entraînement. Il s'exécute une seule fois dans le conteneur `trainer`, puis le conteneur s'arrête.

```python
# =========================================
# H2O AutoML Training with MLflow Tracking
# - Logs run + metrics to the MLflow Tracking Server
# - Registers the best model in the MLflow Model Registry under an alias
# =========================================
import argparse
import json
import os
import tempfile

import h2o
from h2o.automl import H2OAutoML, get_leaderboard

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient


def env(name, default):
    """Read an env var, falling back to a default (empty string treated as unset)."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


def parse_args():
    parser = argparse.ArgumentParser(description="H2O AutoML Training and MLflow Tracking")
    parser.add_argument('--name', '--experiment_name', metavar='',
                        default=env('EXPERIMENT_NAME', 'automl-insurance'),
                        help='Name of Experiment. Default is automl-insurance', type=str)
    parser.add_argument('--target', '--t', metavar='', required=True,
                        help='Name of Target Column (y)', type=str)
    parser.add_argument('--models', '--m', metavar='',
                        default=int(env('AUTOML_MAX_MODELS', '10')),
                        help='Number of AutoML models to train. Default is 10', type=int)
    parser.add_argument('--runtime', metavar='',
                        default=int(env('AUTOML_MAX_RUNTIME_SECS', '0')),
                        help='Max AutoML runtime in seconds (0 = no limit). Default is 0', type=int)
    parser.add_argument('--sample-frac', metavar='',
                        default=float(env('AUTOML_SAMPLE_FRAC', '1.0')),
                        help='Fraction of training rows to use (speeds up smoke tests). Default is 1.0', type=float)
    return parser.parse_args()


def main():
    args = parse_args()

    tracking_uri = env('MLFLOW_TRACKING_URI', None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = env('MODEL_NAME', 'insurance-automl')
    model_alias = env('MODEL_ALIAS', 'champion')

    h2o.init()

    client = MlflowClient()

    experiment = client.get_experiment_by_name(args.name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(args.name)
        experiment = client.get_experiment(experiment_id)
    mlflow.set_experiment(args.name)

    print(f"Name: {args.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tracking uri: {mlflow.get_tracking_uri()}")

    main_frame = h2o.import_file(path='data/processed/train.csv')

    if 0 < args.sample_frac < 1.0:
        main_frame = main_frame.split_frame(ratios=[args.sample_frac], seed=42)[0]
        print(f'Sampled training frame to {args.sample_frac:.0%} -> {main_frame.nrow} rows')

    with open('data/processed/train_col_types.json', 'w') as fp:
        json.dump(main_frame.types, fp)

    target = args.target
    predictors = [n for n in main_frame.col_names if n != target]

    main_frame[target] = main_frame[target].asfactor()

    with mlflow.start_run() as run:
        aml = H2OAutoML(
            max_models=args.models,
            max_runtime_secs=args.runtime,
            seed=42,
            balance_classes=True,
            sort_metric='logloss',
            verbosity='info',
            exclude_algos=['GLM', 'DRF'],
        )

        aml.train(x=predictors, y=target, training_frame=main_frame)

        mlflow.log_param("max_models", args.models)
        mlflow.log_param("max_runtime_secs", args.runtime)
        mlflow.log_param("sample_frac", args.sample_frac)
        mlflow.log_metric("log_loss", aml.leader.logloss())
        mlflow.log_metric("AUC", aml.leader.auc())

        mlflow.h2o.log_model(aml.leader, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")
        print(f'AutoML best model saved in {model_uri}')

        lb = get_leaderboard(aml, extra_columns='ALL').as_data_frame()
        with tempfile.TemporaryDirectory() as tmp:
            lb_path = os.path.join(tmp, 'leaderboard.csv')
            lb.to_csv(lb_path, index=False)
            mlflow.log_artifact(lb_path, artifact_path="model")
        print('Leaderboard logged as MLflow artifact')

    registered = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=model_name)
    client.set_registered_model_alias(name=model_name, alias=model_alias, version=registered.version)
    print(f'Registered model "{model_name}" v{registered.version} with alias @{model_alias}')


if __name__ == "__main__":
    main()
```

#### Ce que fait ce script, étape par étape

1. Lit les variables d'environnement (`MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_ALIAS`, etc.).
2. Démarre un cluster H2O local en mémoire.
3. Crée ou réutilise une expérience MLflow nommée `automl-insurance`.
4. Charge `train.csv` dans un H2OFrame.
5. Si `AUTOML_SAMPLE_FRAC` < 1.0, n'utilise qu'une fraction des données (pour aller plus vite en laboratoire).
6. Lance `H2OAutoML` : essaie plusieurs algorithmes automatiquement (XGBoost, GBM, Deep Learning, etc.), exclut GLM et DRF pour gagner du temps.
7. Logue les paramètres et métriques (log-loss, AUC) dans MLflow.
8. Enregistre le meilleur modèle dans le registre MLflow sous le nom `insurance-automl` avec l'alias `champion`.

---

### Étape 6.5 — Fichier `backend/main.py`

C'est l'API FastAPI. Elle reçoit les requêtes de prédiction du frontend.

```python
# ===========================
# Backend setup (H2O + MLflow + FastAPI)
# Loads the @champion model from the MLflow Model Registry and serves predictions.
# ===========================
import io
import os

import h2o
import mlflow
import mlflow.h2o
import pandas as pd
from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from utils.data_processing import match_col_types, separate_id_col

MODEL_NAME = os.getenv("MODEL_NAME", "insurance-automl")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

app = FastAPI(title="End-to-End AutoML - Insurance Cross-Sell")

h2o.init()
if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
print(f"Loading model from registry: {model_uri}")
best_model = mlflow.h2o.load_model(model_uri)
print("Model loaded successfully")


@app.post("/predict")
async def predict(file: bytes = File(...)):
    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
    test_h2o = h2o.H2OFrame(test_df)

    id_name, X_id, X_h2o = separate_id_col(test_h2o)
    X_h2o = match_col_types(X_h2o)
    preds = best_model.predict(X_h2o)

    if id_name is not None:
        preds_list = preds.as_data_frame()['predict'].tolist()
        id_list = X_id.as_data_frame()[id_name].tolist()
        preds_final = dict(zip(id_list, preds_list))
    else:
        preds_final = preds.as_data_frame()['predict'].tolist()

    json_compatible_item_data = jsonable_encoder(preds_final)
    return JSONResponse(content=json_compatible_item_data)


@app.get("/health")
async def health():
    return PlainTextResponse("OK")


@app.get("/")
async def main():
    content = """
    <body>
    <h2>Welcome to the End-to-End AutoML Pipeline for Insurance Cross-Sell</h2>
    <p>The H2O model and FastAPI instance have been set up successfully.</p>
    <p>Interactive API docs: <a href="/docs">/docs</a></p>
    <p>Open the Streamlit UI (http://localhost:8501) to submit prediction requests.</p>
    </body>
    """
    return HTMLResponse(content=content)
```

#### Ce que fait ce fichier, étape par étape

1. Au démarrage du conteneur `backend`, Python exécute ce fichier. H2O démarre, MLflow se connecte au serveur.
2. Le modèle `models:/insurance-automl@champion` est chargé depuis MLflow — c'est le modèle que `trainer` a enregistré.
3. La route `POST /predict` reçoit un fichier CSV en multipart, le convertit en H2OFrame, sépare la colonne ID si elle existe, aligne les types de colonnes avec ceux du jeu d'entraînement, génère les prédictions et renvoie un JSON.
4. La route `GET /health` renvoie simplement `OK` — c'est ce que le healthcheck de Docker vérifie.

---

### Étape 6.6 — Fichier `backend/utils/data_processing.py`

```python
import h2o
import json


def separate_id_col(h2o_frame):
    """Sépare la colonne ID (si présente) et retourne le dataset sans cette colonne."""
    possible_id_list = ['ID', 'Id', 'id']

    for i in possible_id_list:
        if i in h2o_frame.names:
            id_name = i
            X_id = h2o_frame[:, id_name]
            X_h2o = h2o_frame.drop(id_name)
            break
        else:
            id_name, X_id = None, None
            X_h2o = h2o_frame

    return id_name, X_id, X_h2o


def match_col_types(h2o_frame):
    """Aligne les types de colonnes du jeu de test avec ceux du jeu d'entraînement."""
    with open('data/processed/train_col_types.json') as f:
        train_col_types = json.load(f)

    for key in train_col_types.keys():
        try:
            if train_col_types[key] != h2o_frame.types[key]:
                if train_col_types[key] == 'real' and h2o_frame.types[key] == 'enum':
                    h2o_frame[key] = h2o_frame[key].ascharacter().asnumeric()
                elif train_col_types[key] == 'real':
                    h2o_frame[key] = h2o_frame[key].asnumeric()
                elif train_col_types[key] == 'int':
                    h2o_frame[key] = h2o_frame[key].asfactor()
                elif train_col_types[key] == 'str':
                    h2o_frame[key] = h2o_frame[key].ascharacter()
        except Exception:
            pass

    return h2o_frame
```

#### Pourquoi ces deux fonctions ?

- `separate_id_col` : si le CSV envoyé contient une colonne d'identifiant (`ID`, `Id`, ou `id`), elle la retire avant de prédire (on ne prédit pas à partir d'un ID) et la réutilise pour associer chaque prédiction à son client.
- `match_col_types` : H2O peut déduire des types différents entre le CSV d'entraînement et le CSV de test. Cette fonction force les types du test à correspondre à ceux du train — sans ça, le modèle peut rejeter les données.

---

### Étape 6.7 — Fichier `backend/utils/__init__.py`

Ce fichier doit exister mais peut rester vide. Il indique à Python que `utils/` est un module importable.

Laissez ce fichier vide.

---

### Étape 6.8 — Fichier `frontend/Dockerfile`

```dockerfile
# Frontend image: Streamlit UI
FROM python:3.11-slim

WORKDIR /app

COPY requirements-frontend.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

### Étape 6.9 — Fichier `frontend/requirements-frontend.txt`

```
streamlit==1.41.1
pandas==2.2.3
requests==2.32.3
```

---

### Étape 6.10 — Fichier `frontend/app.py`

C'est l'interface utilisateur. Elle s'affiche dans le navigateur.

```python
# =========================================
# Streamlit UI for the End-to-End AutoML project
# Posts an uploaded CSV to the FastAPI backend, displays predictions,
# explains the dataset, and (when ground-truth labels are present)
# shows an evaluation with a confusion matrix.
# =========================================
import io
import json
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AutoML Insurance Cross-Sell", page_icon="🚗", layout="wide")
st.title('End-to-End AutoML Project: Insurance Cross-Sell')
st.caption('H2O AutoML + MLflow + FastAPI + Streamlit')

ENDPOINT = os.getenv('BACKEND_URL', 'http://backend:8000/predict')

TARGET_COL = 'Response'
LABELS = {1: 'Interested in vehicle insurance', 0: 'Not interested'}


with st.expander('About this project and dataset', expanded=True):
    st.markdown(
        """
**Goal.** An insurance company that already sells **health insurance** wants to know
which of its existing customers are likely to also buy **vehicle insurance**. Targeting
only the interested customers makes a cross-sell campaign cheaper and more effective.

**Dataset.** Health Insurance Cross-Sell (Kaggle). Each row is a customer. Key features:

| Feature | Meaning |
|---|---|
| `Gender`, `Age` | Customer demographics |
| `Driving_License` | Whether the customer holds a driving licence (0/1) |
| `Region_Code` | Region of the customer (one-hot encoded) |
| `Previously_Insured` | Already has vehicle insurance (0/1) |
| `Vehicle_Age`, `Vehicle_Damage` | Age of the vehicle / past damage |
| `Annual_Premium` | Premium the customer pays for health insurance |
| `Policy_Sales_Channel` | Channel used to reach the customer (one-hot encoded) |
| `Vintage` | Number of days the customer has been with the company |
| **`Response`** (target) | **1 = interested** in vehicle insurance, **0 = not interested** |

**Model.** An H2O AutoML model was trained on the processed (one-hot encoded) data and
selected automatically based on log-loss. It outputs `1` (target this customer) or `0`.

**Two ways to use this app:**
- Upload a file **without** `Response` (e.g. `sample_test.csv`) -> get predictions only.
- Upload a file **with** `Response` (e.g. `sample_test_labeled.csv`) -> get predictions
  **plus** an evaluation (accuracy, precision, recall, F1) and a **confusion matrix**.
        """
    )

st.write(
    "Upload a test CSV (already in the processed/one-hot format, like the files in "
    "`backend/data/`) and click **Start Prediction**."
)

test_csv = st.file_uploader('Upload test dataset (CSV)', type=['csv'], accept_multiple_files=False)


def compute_metrics(y_true, y_pred):
    """Confusion-matrix counts and standard classification metrics."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return tp, tn, fp, fn, accuracy, precision, recall, f1


if test_csv:
    test_df = pd.read_csv(test_csv)
    st.subheader('Sample of Uploaded Dataset')
    st.write(test_df.head())
    has_labels = TARGET_COL in test_df.columns
    st.write(
        f"Rows: {len(test_df)} | Columns: {test_df.shape[1]} | "
        f"Ground-truth labels: {'yes (evaluation enabled)' if has_labels else 'no (prediction only)'}"
    )

    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)
    test_bytes_obj.seek(0)

    files = {"file": ('test_dataset.csv', test_bytes_obj, "multipart/form-data")}

    if st.button('Start Prediction'):
        if len(test_df) == 0:
            st.warning("Please upload a non-empty test dataset!")
        else:
            try:
                with st.spinner('Prediction in progress. Please wait...'):
                    response = requests.post(ENDPOINT, files=files, timeout=8000)
                response.raise_for_status()
                result = response.json()

                if isinstance(result, dict):
                    results_df = pd.DataFrame(
                        {'Customer ID': list(result.keys()), 'Prediction': list(result.values())}
                    )
                else:
                    results_df = pd.DataFrame(
                        {'Customer #': range(1, len(result) + 1), 'Prediction': result}
                    )

                results_df['Prediction'] = results_df['Prediction'].astype(int)
                results_df['Result'] = results_df['Prediction'].map(lambda v: LABELS.get(v, str(v)))

                n_total = len(results_df)
                n_interested = int((results_df['Prediction'] == 1).sum())
                n_not = n_total - n_interested
                pct = (n_interested / n_total * 100) if n_total else 0

                st.success(f'Done! {n_total} customers analysed.')

                st.subheader('Summary')
                c1, c2, c3 = st.columns(3)
                c1.metric('Customers analysed', n_total)
                c2.metric('Interested (to target)', n_interested, f'{pct:.0f}%')
                c3.metric('Not interested', n_not)
                st.caption(
                    f"The model predicts that {n_interested} of {n_total} customers "
                    f"({pct:.0f}%) are likely interested in additional vehicle insurance. "
                    "These are the customers to prioritise for the cross-sell campaign."
                )
                st.bar_chart(results_df['Result'].value_counts(), use_container_width=True)

                if has_labels:
                    y_true = test_df[TARGET_COL].astype(int).tolist()
                    y_pred = results_df['Prediction'].tolist()
                    tp, tn, fp, fn, acc, prec, rec, f1 = compute_metrics(y_true, y_pred)

                    st.subheader('Model Evaluation')
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric('Accuracy', f'{acc:.1%}')
                    m2.metric('Precision', f'{prec:.1%}')
                    m3.metric('Recall', f'{rec:.1%}')
                    m4.metric('F1-score', f'{f1:.1%}')

                    st.markdown('**Confusion Matrix** (rows = actual, columns = predicted)')
                    cm = pd.DataFrame(
                        [[tn, fp], [fn, tp]],
                        index=['Actual: Not interested', 'Actual: Interested'],
                        columns=['Predicted: Not interested', 'Predicted: Interested'],
                    )
                    st.table(cm)

                    st.markdown(
                        f"""
**How to read this:**
- **True Negatives ({tn})** - correctly identified as *not interested*.
- **True Positives ({tp})** - correctly identified as *interested* (good leads found).
- **False Positives ({fp})** - predicted *interested* but actually not (wasted outreach).
- **False Negatives ({fn})** - predicted *not interested* but actually interested (missed opportunities).

**Interpretation.** *Precision* ({prec:.0%}) = of all customers the model flags as interested,
how many really are. *Recall* ({rec:.0%}) = of all truly interested customers, how many the
model catches. For a cross-sell campaign you usually want **high recall** (don't miss
potential buyers), while keeping precision high enough to avoid wasting calls.
                        """
                    )

                st.subheader('Detailed results')
                show_cols = [results_df.columns[0], 'Result']
                if has_labels:
                    results_df['Actual'] = [LABELS.get(int(v), str(v)) for v in test_df[TARGET_COL]]
                    results_df['Correct'] = [
                        '✓' if int(a) == int(p) else '✗'
                        for a, p in zip(test_df[TARGET_COL], results_df['Prediction'])
                    ]
                    show_cols += ['Actual', 'Correct']
                st.dataframe(results_df[show_cols], use_container_width=True, hide_index=True)

                d1, d2 = st.columns(2)
                d1.download_button(
                    label='Download results (CSV)',
                    data=results_df.to_csv(index=False),
                    file_name='prediction_results.csv',
                    mime='text/csv',
                )
                d2.download_button(
                    label='Download raw (JSON)',
                    data=json.dumps(result),
                    file_name='automl_prediction_results.json',
                    mime='application/json',
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the prediction backend at {ENDPOINT}.")
                st.exception(exc)
```

---

### Étape 6.11 — Fichier `.gitignore`

```
__pycache__/
*.py[cod]
.env
*.log
.terraform/
mlruns/
mlflow_data/
```

---

## 7. Vérifier la structure finale avant de lancer

### Étape 7.1 — Compter les fichiers

```powershell
Get-ChildItem -Recurse -File | Measure-Object
```

Vous devez avoir au minimum **14 fichiers** (hors fichiers de données).

### Étape 7.2 — Vérifier que les données sont là

```powershell
Test-Path backend\data\processed\train.csv
Test-Path backend\data\processed\train_col_types.json
Test-Path backend\data\sample_test.csv
Test-Path backend\data\sample_test_labeled.csv
```

Chaque ligne doit afficher `True`. Si une ligne affiche `False`, vous avez oublié de copier ce fichier.

### Étape 7.3 — Vérifier que Docker est démarré

```powershell
docker info
```

Si vous voyez une longue liste d'informations sur Docker, c'est bon. Si vous voyez une erreur du type `error during connect`, ouvrez Docker Desktop et attendez qu'il soit **Running**.

---

## 8. Lancer l'application

### Étape 8.1 — Construire et démarrer les quatre conteneurs

```powershell
docker compose up --build
```

Cette commande fait plusieurs choses en séquence :

1. **Build** : Docker lit les deux `Dockerfile` et construit les images `e2e-automl-backend:latest` et `e2e-automl-frontend:latest`. Lors du premier lancement, cela télécharge Python, Java, installe tous les packages. **Comptez 5 à 10 minutes la première fois.**
2. **Start mlflow** : le serveur MLflow démarre, crée une base SQLite.
3. **Healthcheck mlflow** : Docker attend que `http://localhost:5000/health` réponde `200 OK`.
4. **Start trainer** : une fois mlflow healthy, le trainer démarre, entraîne H2O AutoML (2 à 5 minutes), enregistre le modèle dans MLflow, puis s'arrête avec le code 0.
5. **Start backend** : une fois le trainer terminé avec succès, FastAPI démarre, charge le modèle depuis MLflow.
6. **Healthcheck backend** : Docker attend que `http://localhost:8000/health` réponde `OK`.
7. **Start frontend** : une fois le backend healthy, Streamlit démarre.

### Ce que vous voyez dans le terminal pendant le démarrage

Les logs s'affichent en continu. Voici les messages clés à repérer :

```
mlflow-1   | [INFO] mlflow server started
trainer-1  | H2O cluster started
trainer-1  | Sampled training frame to 20% -> XXXX rows
trainer-1  | AutoML best model saved in ...
trainer-1  | Registered model "insurance-automl" v1 with alias @champion
trainer-1  | exited with code 0           <-- trainer a terminé avec succès
backend-1  | Loading model from registry: models:/insurance-automl@champion
backend-1  | Model loaded successfully
backend-1  | INFO: Application startup complete.
frontend-1 | You can now view your Streamlit app in your browser.
```

> **Si le terminal affiche** `trainer exited with code 1` : le trainer a échoué. Consultez les logs avec `docker compose logs trainer` pour voir l'erreur exacte.

### Étape 8.2 — Lancer en mode détaché (optionnel)

Si vous préférez récupérer votre terminal, ajoutez `-d` :

```powershell
docker compose up --build -d
```

Les conteneurs continuent de tourner en arrière-plan. Pour voir les logs :

```powershell
docker compose logs -f
```

Pour voir uniquement les logs du trainer :

```powershell
docker compose logs trainer
```

---

## 9. Vérifier que les quatre services fonctionnent

### Étape 9.1 — Vérifier l'état des conteneurs

Ouvrez un **deuxième PowerShell** (gardez le premier avec les logs) :

```powershell
docker compose ps
```

Résultat attendu :

```
NAME        IMAGE                        STATUS
mlflow      e2e-automl-backend:latest    Up (healthy)
trainer     e2e-automl-backend:latest    Exited (0)
backend     e2e-automl-backend:latest    Up (healthy)
frontend    e2e-automl-frontend:latest   Up
```

Points à vérifier :
- `mlflow` : doit être `Up (healthy)`, **pas** `Up (unhealthy)`.
- `trainer` : doit être `Exited (0)`. Le code `0` signifie succès. `Exited (1)` signifie échec.
- `backend` : doit être `Up (healthy)`.
- `frontend` : doit être `Up`.

### Étape 9.2 — Vérifier le serveur MLflow

```powershell
Invoke-WebRequest -Uri http://localhost:5000/health -UseBasicParsing
```

Résultat attendu : `StatusCode: 200`.

Ouvrez aussi `http://localhost:5000` dans votre navigateur. Vous devez voir le tableau de bord MLflow avec :
- L'expérience `automl-insurance`.
- Un run avec les métriques `log_loss` et `AUC`.
- Le modèle `insurance-automl` dans la section **Models** avec l'alias `@champion`.

### Étape 9.3 — Vérifier l'API FastAPI

```powershell
Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing
```

Résultat attendu : `StatusCode: 200`, contenu `OK`.

Ouvrez aussi `http://localhost:8000/docs` dans votre navigateur. Vous devez voir la documentation interactive Swagger de l'API avec les routes `/predict`, `/health` et `/`.

### Étape 9.4 — Vérifier l'interface Streamlit

Ouvrez `http://localhost:8501` dans votre navigateur. Vous devez voir :
- Le titre **End-to-End AutoML Project: Insurance Cross-Sell**.
- Une section dépliable **About this project and dataset**.
- Un bouton **Browse files** pour uploader un CSV.

---

## 10. Faire une prédiction

### Étape 10.1 — Test sans étiquettes (prédictions uniquement)

1. Allez sur `http://localhost:8501`.
2. Cliquez sur **Browse files**.
3. Naviguez jusqu'au fichier `backend\data\sample_test.csv` dans votre dossier de projet.
4. Sélectionnez le fichier. Streamlit affiche les premières lignes du CSV.
5. Cliquez sur le bouton **Start Prediction**.
6. Attendez quelques secondes (l'animation de chargement tourne).

Résultat attendu :
- Un message vert : `Done! X customers analysed.`
- Trois métriques : nombre total de clients, nombre de clients intéressés (avec pourcentage), nombre de clients non intéressés.
- Un graphique en barres.
- Un tableau détaillé avec une colonne **Result** indiquant `Interested in vehicle insurance` ou `Not interested`.
- Deux boutons de téléchargement : CSV et JSON.

### Étape 10.2 — Test avec étiquettes (prédictions + évaluation)

1. Répétez les étapes 1 à 5 mais choisissez `backend\data\sample_test_labeled.csv`.

Résultat attendu en plus des prédictions :
- Une section **Model Evaluation** avec les métriques Accuracy, Precision, Recall, F1-score.
- Une **Confusion Matrix** avec True Negatives, True Positives, False Positives, False Negatives.
- Une interprétation en texte de la matrice de confusion.
- Une colonne **Actual** et une colonne **Correct** (✓ ou ✗) dans le tableau détaillé.

### Étape 10.3 — Tester l'API directement avec PowerShell

```powershell
$file = "backend\data\sample_test.csv"
$uri  = "http://localhost:8000/predict"

$form = @{ file = Get-Item $file }
$response = Invoke-RestMethod -Uri $uri -Method Post -Form $form
$response | ConvertTo-Json | Select-Object -First 20
```

Résultat attendu : un tableau JSON de 0 et de 1, un par client.

---

## 11. Explorer le tableau de bord MLflow

### Étape 11.1 — Voir l'expérience et le run

1. Allez sur `http://localhost:5000`.
2. Cliquez sur **automl-insurance** dans la liste des expériences.
3. Cliquez sur le run (ligne dans le tableau).
4. Explorez les onglets :
   - **Parameters** : `max_models`, `max_runtime_secs`, `sample_frac`.
   - **Metrics** : `log_loss` (plus c'est bas, mieux c'est), `AUC` (plus c'est proche de 1, mieux c'est).
   - **Artifacts** : le modèle H2O sérialisé et le `leaderboard.csv`.

### Étape 11.2 — Voir le modèle enregistré

1. Dans MLflow, cliquez sur **Models** en haut à gauche.
2. Cliquez sur `insurance-automl`.
3. Vérifiez que la version 1 existe avec l'alias `champion`.

---

## 12. Lire les logs pour comprendre ce qui se passe

### Étape 12.1 — Voir tous les logs en direct

```powershell
docker compose logs -f
```

Appuyez sur `Ctrl+C` pour arrêter l'affichage des logs (les conteneurs continuent de tourner).

### Étape 12.2 — Voir les logs d'un seul service

```powershell
docker compose logs mlflow
docker compose logs trainer
docker compose logs backend
docker compose logs frontend
```

### Étape 12.3 — Voir les logs depuis le début du trainer

```powershell
docker compose logs trainer --no-log-prefix
```

Vous devez voir les étapes de l'entraînement H2O : initialisation du cluster, chargement des données, résultats AutoML, enregistrement dans MLflow.

### Étape 12.4 — Voir les logs du backend lors d'une prédiction

```powershell
docker compose logs backend -f
```

Ensuite, faites une prédiction dans l'interface. Vous verrez apparaître dans le terminal :
```
backend-1  | [+] Initiate Prediction
```

---

## 13. Arrêter proprement l'application

### Étape 13.1 — Arrêter les conteneurs

```powershell
docker compose down
```

Docker arrête et supprime les conteneurs. Le volume `mlflow_data` est conservé : si vous relancez `docker compose up`, le modèle MLflow est déjà là et le trainer ne s'exécute **pas** à nouveau.

### Étape 13.2 — Arrêter ET supprimer le volume MLflow

Si vous voulez repartir de zéro (forcer un nouvel entraînement) :

```powershell
docker compose down -v
```

L'option `-v` supprime aussi le volume `mlflow_data`. Au prochain `docker compose up --build`, le trainer s'exécutera à nouveau.

### Étape 13.3 — Vérifier que tout est arrêté

```powershell
docker compose ps
```

Résultat attendu : aucun conteneur listé, ou tous à l'état `Exited`.

### Étape 13.4 — Supprimer les images (nettoyage complet, optionnel)

```powershell
docker rmi e2e-automl-backend:latest e2e-automl-frontend:latest
```

Cela libère l'espace disque. Le prochain `docker compose up --build` téléchargera et reconstruira tout.

---

## 14. Captures à remettre

- Capture de `docker compose up --build` montrant les quatre services démarrés.
- Capture de `docker compose ps` montrant `mlflow` healthy, `trainer` exited 0, `backend` healthy, `frontend` up.
- Capture du tableau de bord MLflow (`http://localhost:5000`) avec l'expérience `automl-insurance` et les métriques.
- Capture de la page du modèle dans MLflow montrant l'alias `@champion`.
- Capture de l'interface Streamlit (`http://localhost:8501`) avec un CSV uploadé.
- Capture des résultats de prédiction avec le résumé (métriques, graphique).
- Capture de la matrice de confusion obtenue avec `sample_test_labeled.csv`.
- Capture de `docker compose down` réussi.

**Interdit dans les captures :** ne montrez jamais de clés d'accès, tokens ou mots de passe.

---

## 15. Erreurs fréquentes et solutions directes

### Erreur 15.1 — `error during connect: ... Is the docker daemon running?`

- **Cause :** Docker Desktop n'est pas démarré.
- **Solution :** ouvrez Docker Desktop et attendez qu'il affiche Running (baleine verte).
- **Vérification :** `docker info`

### Erreur 15.2 — `trainer exited with code 1`

- **Cause :** le trainer a échoué. Les raisons les plus fréquentes : manque de RAM, fichier `train.csv` manquant, problème de connexion avec MLflow.
- **Solution :**
  1. Consultez les logs : `docker compose logs trainer`
  2. Si l'erreur mentionne Java ou H2O : vérifiez que Docker a au moins 4 Go de RAM alloués (Settings > Resources > Memory).
  3. Si l'erreur mentionne `train.csv not found` : vérifiez que le fichier est bien dans `backend\data\processed\`.
  4. Si l'erreur mentionne la connexion MLflow : assurez-vous que le service mlflow est healthy avant que le trainer ne démarre (relancez avec `docker compose up`).

### Erreur 15.3 — `backend` reste en état `unhealthy`

- **Cause :** le backend n'arrive pas à charger le modèle depuis MLflow. Le plus souvent, c'est parce que le trainer n'a pas terminé correctement (voir erreur 15.2).
- **Solution :** résolvez d'abord le problème du trainer. Ensuite, relancez `docker compose down -v && docker compose up --build`.
- **Vérification :** `docker compose logs backend`

### Erreur 15.4 — `Could not reach the prediction backend`

- **Cause :** vous avez cliqué sur Start Prediction mais le backend n'est pas encore healthy.
- **Solution :** attendez quelques minutes supplémentaires. Vérifiez avec `docker compose ps` que `backend` est `Up (healthy)`. Rafraîchissez `http://localhost:8501` et réessayez.

### Erreur 15.5 — Port déjà utilisé (`Bind for 0.0.0.0:5000 failed`)

- **Cause :** un autre programme utilise le port 5000 ou 8000 ou 8501 sur votre machine.
- **Solution :** identifiez le programme et arrêtez-le. Sur Windows :

```powershell
netstat -ano | findstr :5000
```

Notez le PID, puis arrêtez le processus avec `Stop-Process -Id <PID>`.

### Erreur 15.6 — Les images se reconstruisent à chaque `docker compose up`

- **Cause :** comportement normal si vous avez modifié un fichier source. Docker compare les couches.
- **Solution :** si vous ne voulez pas reconstruire, omettez `--build` :

```powershell
docker compose up
```

### Erreur 15.7 — `No module named 'utils'`

- **Cause :** le fichier `backend/utils/__init__.py` est manquant.
- **Solution :** créez-le (même vide) :

```powershell
New-Item backend\utils\__init__.py -ItemType File
```

Ensuite reconstruisez : `docker compose up --build`.

---

## 16. Résumé final du projet

- Vous avez créé une structure de projet complète avec deux services Python distincts (backend et frontend).
- Vous avez écrit un `docker-compose.yml` qui orchestre quatre conteneurs avec des dépendances et des healthchecks.
- Vous avez utilisé **H2O AutoML** pour entraîner automatiquement plusieurs modèles et sélectionner le meilleur.
- Vous avez utilisé **MLflow** comme registre de modèles : le trainer dépose le modèle, le backend le récupère.
- Vous avez exposé les prédictions via une **API FastAPI** qui reçoit un CSV et renvoie un JSON.
- Vous avez construit une **interface Streamlit** qui permet à n'importe qui d'utiliser le modèle sans écrire de code.
- Vous avez appris à lire les logs Docker pour comprendre l'ordre de démarrage et détecter les erreurs.
- Vous avez arrêté proprement la stack avec `docker compose down`.

**Phrase à retenir :** un projet de Machine Learning en production, ce n'est pas seulement un notebook. C'est un pipeline complet : données prétraitées, entraînement automatisé, registre de modèles, API de prédiction et interface utilisateur — le tout orchestré et reproductible.

---

## 17. Sources officielles utiles

- [Documentation Docker Compose](https://docs.docker.com/compose/)
- [Documentation H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [Documentation MLflow — Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Dataset Kaggle — Health Insurance Cross-Sell](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)
