# End-to-End AutoML Insurance — Step-by-Step Practical Assignment
**Version for students — English**

**Purpose of this document:** This document tells you exactly what to do, in what order, what to type, and what to check. You will rebuild a complete Machine Learning application from scratch: an H2O AutoML model trained automatically, an MLflow server that stores the model, a FastAPI API that serves predictions, and a Streamlit web interface accessible in your browser. Everything runs with **Docker Compose** — one single command starts all four services.

**No cloud costs.** This project runs entirely on your local machine with Docker Desktop. No AWS account required. No billing.

**By the end of this assignment, you will be able to:** create a multi-container project structure → write every file by hand → launch Docker Compose → verify all four services → make a live prediction in the browser → read Docker logs → shut everything down cleanly.

---

## Two Assignment Options

Choose the one your instructor assigned to you. **Both options use the exact same codebase.** The difference is in the deliverables and the level of understanding you must demonstrate.

---

### ASSIGNMENT A — Reproduce and Verify

**Goal:** Follow this guide step by step, reproduce the entire application from scratch, verify every service, make predictions, and document what you did with screenshots.

**Deliverables:**
1. Your project folder (zipped) containing all the files you wrote.
2. A short PDF report (1 to 2 pages) with your 8 required screenshots.
3. A paragraph per screenshot explaining what it shows and why it proves the step worked.

**Grading criteria:**
- All four containers start in the correct order. (25 pts)
- Predictions work with both test files. (25 pts)
- Screenshots are complete and correctly annotated. (25 pts)
- The `docker compose down` screenshot proves a clean teardown. (25 pts)

---

### ASSIGNMENT B — Reproduce, Extend, and Explain

**Goal:** Follow this guide step by step, reproduce the application, then make two specific changes to the code (described at the end of this document), verify they work, and write a short technical explanation.

**Deliverables:**
1. Your modified project folder (zipped).
2. A PDF report (2 to 3 pages) with all screenshots plus a section explaining each change you made and why it works.
3. A `CHANGES.md` file in your project folder listing every line you changed, in which file, and what it does.

**Grading criteria:**
- Base application works identically to Assignment A. (40 pts)
- Both code changes are implemented correctly and verified. (40 pts)
- Written explanation is clear and technically accurate. (20 pts)

---

## 1. Expected Final Result

Before you start, read this list carefully. This is what you must have when you are done.

- A project folder with this structure:

```
projet-automl-insurance/
├── docker-compose.yml
├── .gitignore
├── backend/
│   ├── Dockerfile
│   ├── train.py
│   ├── main.py
│   ├── requirements-backend.txt
│   ├── data/
│   │   ├── processed/
│   │   │   ├── train.csv
│   │   │   └── train_col_types.json
│   │   ├── sample_test.csv
│   │   └── sample_test_labeled.csv
│   └── utils/
│       ├── __init__.py
│       └── data_processing.py
└── frontend/
    ├── Dockerfile
    ├── app.py
    └── requirements-frontend.txt
```

- Four Docker containers that start in the right order: `mlflow` → `trainer` → `backend` → `frontend`.
- The Streamlit web interface accessible at `http://localhost:8501`.
- The FastAPI API documentation accessible at `http://localhost:8000/docs`.
- The MLflow experiment dashboard accessible at `http://localhost:5000`.
- A successful prediction after uploading a CSV file in the interface.
- A clean shutdown with `docker compose down`.

---

## 2. What You Are Building

You are building a complete AutoML pipeline to predict which customers of a health insurance company are likely to also buy **vehicle insurance**. This is called **cross-selling**. The model outputs `1` (customer is interested) or `0` (customer is not interested). This is a binary classification problem.

### The four services and what each one does

| Service | Technology | Port | Role |
|---|---|---|---|
| `mlflow` | MLflow Server | 5000 | Stores trained models in a registry. Think of it as the "vault" for models. Runs a web UI so you can inspect runs and metrics. |
| `trainer` | H2O AutoML + Python | none | Runs exactly once. Automatically trains several ML algorithms, picks the best one (the "leader"), saves it to MLflow, then exits. |
| `backend` | FastAPI + H2O | 8000 | Waits for HTTP requests. Receives a CSV file, loads it into H2O, runs the model, returns a JSON list of predictions. |
| `frontend` | Streamlit | 8501 | The interactive web page. The user uploads a CSV, clicks a button, and sees the prediction results in the browser. |

### Full data flow diagram

```
USER (browser at http://localhost:8501)
      |
      |  Step 1. uploads a CSV file
      |  Step 2. clicks "Start Prediction"
      v
[ frontend ]  Streamlit container (port 8501)
      |
      |  Step 3. sends an HTTP POST to http://backend:8000/predict
      |          (the CSV is sent as multipart/form-data)
      v
[ backend ]   FastAPI container (port 8000)
      |
      |  Step 4. receives the CSV bytes
      |  Step 5. reads CSV -> pandas DataFrame -> H2OFrame
      |  Step 6. separates the ID column if one exists
      |  Step 7. aligns column types with the training set
      |  Step 8. calls best_model.predict(X_h2o)
      v
[ H2O model ] (loaded in memory inside the backend container)
      |
      |  Step 9. returns a column "predict" with 0 or 1 per row
      v
[ backend ]   converts to JSON, sends HTTP response
      |
      |  Step 10. returns {"1": 0, "2": 1, "3": 0, ...} or [0, 1, 0, ...]
      v
[ frontend ]  displays summary, bar chart, table, confusion matrix (if labels present)
      |
      v
USER sees the results in the browser
```

### Mandatory startup order

Docker Compose enforces this order using `depends_on` with health checks:

```
mlflow starts first
  └── trainer starts ONLY AFTER mlflow is healthy
       └── backend starts ONLY AFTER trainer exits with code 0
            └── frontend starts ONLY AFTER backend is healthy
```

**Why does this matter?** The trainer must be able to reach the MLflow server to save the model. The backend must load the model from MLflow before accepting prediction requests. The frontend must reach a live backend before it can forward CSV files. If you skip the health checks, services start in parallel and crash because their dependencies aren't ready.

---

## 3. Prerequisites — Do This Once

> If Docker Desktop is already installed, running, and you have verified it works with `docker info`, skip to Section 4.

### Step 3.1 — Install Docker Desktop

1. Open your browser.
2. Go to [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/).
3. Click **Download for Windows**.
4. Wait for the installer file (`Docker Desktop Installer.exe`) to download.
5. Double-click the installer.
6. Accept the license agreement.
7. Keep the default options checked. Click **OK**.
8. Wait for the installation to finish (2–5 minutes).
9. Click **Close and restart** when prompted.
10. After your machine restarts, open **Docker Desktop** from the Start menu.
11. Wait until you see the whale icon in the taskbar and the status says **Running** (green).

> **If Docker Desktop shows a yellow or red status:** wait 1–2 more minutes. If it stays red, try right-clicking the taskbar whale icon and choosing "Restart Docker Desktop".

### Step 3.2 — Give Docker enough memory

H2O AutoML is memory-intensive. If Docker gets less than 4 GB, the training will crash with an out-of-memory error.

12. In Docker Desktop, click the **Settings gear icon** in the top-right corner.
13. Click **Resources** in the left sidebar.
14. Click **Memory**.
15. Drag the slider to at least **4 GB** (ideally 6 GB if your machine allows it).
16. Click **Apply & restart**.
17. Wait for Docker Desktop to restart (about 30 seconds).

### Step 3.3 — Verify Docker works

Open **PowerShell** (press `Windows + R`, type `powershell`, press Enter).

Type this command exactly:

```powershell
docker --version
```

Expected result (your version number may differ, that is fine):

```
Docker version 27.0.3, build 7d4bcd8
```

> **If you see:** `'docker' is not recognized as an internal or external command` — Docker Desktop did not add itself to your PATH. Close PowerShell, reopen it, and try again. If it still fails, restart your machine.

Now verify Docker Compose is available:

```powershell
docker compose version
```

Expected result:

```
Docker Compose version v2.29.1
```

Now verify Docker can run a container:

```powershell
docker run hello-world
```

Expected result: a message starting with `Hello from Docker!`. If you see this, Docker is working correctly.

### Step 3.4 — Install Git (if not already installed)

```powershell
git --version
```

If you see a version number, Git is already installed. If you see an error:

```powershell
winget install Git.Git
```

After installation, close PowerShell, reopen it, and verify:

```powershell
git --version
```

---

## 4. Create the Project Folder Structure

> **Rule:** Do not copy files from somewhere else yet. Create the folders and empty files first. You will fill them in Section 6.

### Step 4.1 — Create the project root folder

Open PowerShell. Choose a location you will remember. For example, your Documents folder:

```powershell
cd $HOME\Documents
mkdir projet-automl-insurance
cd projet-automl-insurance
```

Verify you are in the right place:

```powershell
pwd
```

Expected result: the path ends with `projet-automl-insurance`.

### Step 4.2 — Create all subdirectories

```powershell
mkdir backend
mkdir backend\utils
mkdir backend\data
mkdir backend\data\raw
mkdir backend\data\processed
mkdir frontend
```

### Step 4.3 — Create all empty files

```powershell
New-Item docker-compose.yml                       -ItemType File
New-Item .gitignore                               -ItemType File
New-Item backend\Dockerfile                       -ItemType File
New-Item backend\train.py                         -ItemType File
New-Item backend\main.py                          -ItemType File
New-Item backend\requirements-backend.txt         -ItemType File
New-Item backend\utils\data_processing.py         -ItemType File
New-Item backend\utils\__init__.py                -ItemType File
New-Item frontend\Dockerfile                      -ItemType File
New-Item frontend\app.py                          -ItemType File
New-Item frontend\requirements-frontend.txt       -ItemType File
```

### Step 4.4 — Verify the full structure

```powershell
Get-ChildItem -Recurse -Name
```

You must see exactly this (the `data/` subfolders will appear empty for now):

```
.gitignore
docker-compose.yml
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
frontend\Dockerfile
frontend\app.py
frontend\requirements-frontend.txt
```

> **If a file or folder is missing:** re-run the `New-Item` or `mkdir` command for that specific file. Do not proceed to the next section until all files exist.

---

## 5. Copy the Data Files

The data files are provided by your instructor. You do not need a Kaggle account for this lab.

### Step 5.1 — What files you need and where they go

| File to copy | Destination inside your project |
|---|---|
| `train.csv` (preprocessed training set) | `backend\data\processed\train.csv` |
| `train_col_types.json` (column type metadata) | `backend\data\processed\train_col_types.json` |
| `sample_test.csv` (test set, no labels) | `backend\data\sample_test.csv` |
| `sample_test_labeled.csv` (test set with labels) | `backend\data\sample_test_labeled.csv` |

### Step 5.2 — Verify the data files are in place

After copying, run:

```powershell
Test-Path backend\data\processed\train.csv
Test-Path backend\data\processed\train_col_types.json
Test-Path backend\data\sample_test.csv
Test-Path backend\data\sample_test_labeled.csv
```

Each line must print `True`. If any line prints `False`, the file is missing — copy it before continuing.

### Step 5.3 — Check the data is not empty

```powershell
(Get-Item backend\data\processed\train.csv).Length
(Get-Item backend\data\sample_test.csv).Length
```

Both numbers must be greater than 0. If either is 0, the file is empty and the copy failed.

### What are these two test files?

- `sample_test.csv` — contains only the predictor columns, **without** the `Response` column. When you upload this file, the app returns predictions only.
- `sample_test_labeled.csv` — contains the predictor columns **plus** the true `Response` column. When you upload this file, the app returns predictions **and** evaluates the model against the true labels (accuracy, precision, recall, F1-score, and a confusion matrix).

---

## 6. Write Every File by Hand

> **Critical rule:** Copy each file exactly as shown. Do not skip a single line. Do not change variable names, indentation, or spacing unless instructed. After you get the application working, you can experiment.

Open your project folder in VS Code or any text editor:

```powershell
code .
```

If VS Code is not installed, you can use Notepad. To open a file in Notepad:

```powershell
notepad docker-compose.yml
```

---

### Step 6.1 — File: `docker-compose.yml`

This is the most important file in the entire project. It defines all four containers, their startup order, their environment variables, their exposed ports, and their health checks.

Open `docker-compose.yml` and write:

```yaml
# End-to-End AutoML (H2O + MLflow + FastAPI + Streamlit)
# Start the full stack: docker compose up --build
#
# Flow: mlflow (server + registry) -> trainer (trains + registers @champion)
#       -> backend (serves the model) -> frontend (Streamlit UI)
#
# Exposed ports:
#   - 8501 : Streamlit UI    (http://localhost:8501)
#   - 8000 : FastAPI         (http://localhost:8000)
#   - 5000 : MLflow UI       (http://localhost:5000)

services:

  # ----------------------------------------------------------------
  # Service 1: MLflow Tracking Server + Model Registry
  # Uses a SQLite database stored in the mlflow_data Docker volume.
  # The --serve-artifacts flag means MLflow also serves the model
  # binary files, so the backend does not need a shared filesystem.
  # ----------------------------------------------------------------
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

  # ----------------------------------------------------------------
  # Service 2: H2O AutoML Trainer (one-shot)
  # Trains models, logs to MLflow, registers the best as @champion,
  # then exits. restart: "no" ensures it never restarts automatically.
  # ----------------------------------------------------------------
  trainer:
    build: ./backend
    image: e2e-automl-backend:latest
    command: ["python", "train.py", "--target", "Response"]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MODEL_NAME: insurance-automl
      MODEL_ALIAS: champion
      # These three settings make training fast enough for a lab session.
      # AUTOML_MAX_MODELS=5 means AutoML tries 5 algorithms max.
      # AUTOML_MAX_RUNTIME_SECS=120 stops after 2 minutes no matter what.
      # AUTOML_SAMPLE_FRAC=0.2 uses only 20% of training rows (faster).
      AUTOML_MAX_MODELS: "5"
      AUTOML_MAX_RUNTIME_SECS: "120"
      AUTOML_SAMPLE_FRAC: "0.2"
    depends_on:
      mlflow:
        condition: service_healthy
    restart: "no"
    networks:
      - project_network

  # ----------------------------------------------------------------
  # Service 3: FastAPI prediction backend
  # Waits for the trainer to finish (service_completed_successfully),
  # then loads the @champion model from MLflow and listens on port 8000.
  # ----------------------------------------------------------------
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

  # ----------------------------------------------------------------
  # Service 4: Streamlit frontend
  # Only starts after the backend is healthy (model is loaded and
  # the /health endpoint returns OK).
  # ----------------------------------------------------------------
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

# Named volume so MLflow data survives container restarts.
# Run "docker compose down -v" to delete it and force retraining.
volumes:
  mlflow_data:

# Private Docker network. Services talk to each other by service name,
# e.g. "backend" resolves to the backend container's IP automatically.
networks:
  project_network:
```

#### Line-by-line explanation of the key concepts

**`build: ./backend`** — Docker reads `backend/Dockerfile` and builds an image from it. The same image is reused for `mlflow`, `trainer`, and `backend` (they all run Python + H2O + MLflow).

**`command:`** — This overrides the default `CMD` in the Dockerfile. That is how the same image runs three different roles: the MLflow server command, the training script, or the FastAPI server.

**`environment:`** — These key-value pairs become environment variables inside the container. The Python code reads them with `os.getenv("MLFLOW_TRACKING_URI")`.

**`ports: "5000:5000"`** — Maps port 5000 inside the container to port 5000 on your machine. Format is `HOST:CONTAINER`. This is why you can open `http://localhost:5000` in your browser.

**`depends_on` with `condition: service_healthy`** — Docker does not start this service until the referenced service passes its health check. This is what enforces the startup order.

**`depends_on` with `condition: service_completed_successfully`** — Docker does not start the backend until the trainer exits with exit code 0 (success). If the trainer crashes (exit code 1), the backend never starts.

**`healthcheck`** — Docker runs this command periodically. `interval: 10s` = run every 10 seconds. `retries: 12` = try 12 times before marking as unhealthy. `start_period: 20s` = don't count failures during the first 20 seconds (the service is still starting up).

**`volumes: mlflow_data:/mlflow`** — The directory `/mlflow` inside the container is stored in the named Docker volume `mlflow_data`. Volumes persist even after containers are deleted. This means if you stop and restart the stack with `docker compose up` (without `-v`), the trained model is still there.

**`networks: project_network`** — All services share this virtual network. Docker automatically creates a DNS entry for each service: the container named `mlflow` is reachable at `http://mlflow:5000` from inside any other container on the same network.

---

### Step 6.2 — File: `backend/Dockerfile`

Open `backend/Dockerfile` and write:

```dockerfile
# Backend image: FastAPI + H2O + MLflow
# This single image is used for three different roles:
#   - the MLflow tracking server (service: mlflow)
#   - the AutoML trainer (service: trainer)
#   - the prediction API (service: backend)
# The role is determined by the "command:" in docker-compose.yml.

FROM python:3.11-slim

WORKDIR /app

# H2O AutoML requires a Java Runtime Environment (JRE).
# default-jre-headless installs OpenJDK without a graphical display.
# --no-install-recommends keeps the image lean (avoids pulling docs, etc.)
# Cleaning apt cache at the end reduces the final image size.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first (separate layer).
# Docker caches this layer. If requirements.txt does not change,
# Docker skips the pip install on rebuilds — much faster.
COPY requirements-backend.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code into the container.
COPY . /app

# Document which ports this container uses.
# 8000 = FastAPI prediction API
# 5000 = MLflow server (when running the mlflow command)
EXPOSE 8000 5000

# Default command: start FastAPI.
# Overridden by docker-compose.yml for the mlflow and trainer services.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Why `python:3.11-slim` and not just `python:3.11`?

The `slim` variant strips out a lot of development tools, documentation, and locale data that are not needed to run a Python application. The resulting Docker image is roughly 3× smaller, which means faster builds and faster downloads.

#### Why install Java?

H2O is a Java-based framework. When you call `h2o.init()` in Python, it actually starts a Java Virtual Machine (JVM) process in the background. Without Java installed in the container, `h2o.init()` crashes immediately with a `Java not found` error.

---

### Step 6.3 — File: `backend/requirements-backend.txt`

Open `backend/requirements-backend.txt` and write:

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

> **Do not change the version numbers.** These specific versions are tested together. A different version of `h2o` or `mlflow` can introduce incompatibilities that are hard to debug.

What each package does:

| Package | Role |
|---|---|
| `fastapi` | The web framework for the prediction API. |
| `uvicorn` | The ASGI server that runs the FastAPI app. |
| `h2o` | The AutoML framework. Requires Java. |
| `mlflow` | Experiment tracking and model registry. |
| `pandas` | Reads the uploaded CSV into a DataFrame. |
| `requests` | Not used by the backend directly, but required by mlflow. |
| `python-multipart` | Allows FastAPI to receive file uploads (multipart/form-data). Without this, the `/predict` endpoint cannot receive a CSV file. |
| `tabulate` | Required by H2O to print the leaderboard table to the logs. |

---

### Step 6.4 — File: `backend/train.py`

This is the training script. It runs once inside the `trainer` container, then the container exits.

Open `backend/train.py` and write:

```python
# =========================================
# H2O AutoML Training with MLflow Tracking
#
# What this script does:
#   1. Connects to the MLflow tracking server.
#   2. Starts an H2O cluster (Java process inside the container).
#   3. Loads the preprocessed training data from data/processed/train.csv.
#   4. Optionally samples a fraction of rows (to speed up lab sessions).
#   5. Saves the column data types to train_col_types.json (needed during prediction).
#   6. Runs H2O AutoML: trains up to N models, picks the best one (the "leader").
#   7. Logs parameters and metrics to MLflow.
#   8. Saves the best model as an MLflow artifact.
#   9. Registers the model in the MLflow Model Registry as "insurance-automl".
#  10. Tags the registered model with the alias "champion".
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
    """Read an environment variable. If it is not set or is empty, return the default."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


def parse_args():
    parser = argparse.ArgumentParser(description="H2O AutoML Training and MLflow Tracking")
    parser.add_argument(
        '--name', '--experiment_name',
        metavar='',
        default=env('EXPERIMENT_NAME', 'automl-insurance'),
        help='Name of the MLflow experiment. Default: automl-insurance',
        type=str,
    )
    parser.add_argument(
        '--target', '--t',
        metavar='',
        required=True,
        help='Name of the target column (the column to predict). Example: Response',
        type=str,
    )
    parser.add_argument(
        '--models', '--m',
        metavar='',
        default=int(env('AUTOML_MAX_MODELS', '10')),
        help='Maximum number of AutoML models to train. Default: 10',
        type=int,
    )
    parser.add_argument(
        '--runtime',
        metavar='',
        default=int(env('AUTOML_MAX_RUNTIME_SECS', '0')),
        help='Maximum AutoML runtime in seconds. 0 means no time limit. Default: 0',
        type=int,
    )
    parser.add_argument(
        '--sample-frac',
        metavar='',
        default=float(env('AUTOML_SAMPLE_FRAC', '1.0')),
        help='Fraction of training rows to use. Values < 1.0 speed up the lab. Default: 1.0',
        type=float,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Point MLflow at the tracking server running in the mlflow container.
    # If MLFLOW_TRACKING_URI is not set, mlflow logs locally to ./mlruns/
    tracking_uri = env('MLFLOW_TRACKING_URI', None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = env('MODEL_NAME', 'insurance-automl')
    model_alias = env('MODEL_ALIAS', 'champion')

    # Start the H2O cluster (a Java process).
    # Inside a Docker container, H2O uses the container's CPU and RAM.
    h2o.init()

    client = MlflowClient()

    # Create the MLflow experiment if it does not already exist.
    # If you rerun the trainer, it reuses the existing experiment.
    experiment = client.get_experiment_by_name(args.name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(args.name)
        experiment = client.get_experiment(experiment_id)
    mlflow.set_experiment(args.name)

    print(f"Experiment name:     {args.name}")
    print(f"Experiment ID:       {experiment.experiment_id}")
    print(f"Artifact location:   {experiment.artifact_location}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Load the preprocessed training data.
    # The path is relative to /app (the WORKDIR in the Dockerfile).
    main_frame = h2o.import_file(path='data/processed/train.csv')
    print(f"Loaded training frame: {main_frame.nrow} rows x {main_frame.ncol} columns")

    # Optionally sample a fraction of rows to make the lab session faster.
    # AUTOML_SAMPLE_FRAC=0.2 in docker-compose.yml uses 20% of the data.
    if 0 < args.sample_frac < 1.0:
        main_frame = main_frame.split_frame(ratios=[args.sample_frac], seed=42)[0]
        print(f"Sampled to {args.sample_frac:.0%} of original -> {main_frame.nrow} rows")

    # Save the column data types of the training frame.
    # During prediction, match_col_types() in data_processing.py reads this file
    # and ensures the test CSV has the same types as the training CSV.
    # Without this step, H2O may reject the test data due to type mismatches.
    with open('data/processed/train_col_types.json', 'w') as fp:
        json.dump(main_frame.types, fp)
    print("Column types saved to data/processed/train_col_types.json")

    # Define predictor columns (everything except the target).
    target = args.target
    predictors = [n for n in main_frame.col_names if n != target]
    print(f"Target column: {target}")
    print(f"Number of predictor columns: {len(predictors)}")

    # Convert the target column to a categorical (factor) type.
    # This tells H2O AutoML to solve a classification problem (not regression).
    main_frame[target] = main_frame[target].asfactor()

    # Run H2O AutoML inside an MLflow run context.
    # Everything logged inside the "with mlflow.start_run()" block is attached
    # to this single run in the MLflow experiment.
    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        aml = H2OAutoML(
            max_models=args.models,
            max_runtime_secs=args.runtime,
            seed=42,
            balance_classes=True,     # Handles the class imbalance in the insurance dataset
            sort_metric='logloss',    # AutoML picks the model with the lowest log-loss
            verbosity='info',
            exclude_algos=['GLM', 'DRF'],  # Skip slower algorithms to finish faster in labs
        )

        # This is where the actual training happens.
        # H2O tries up to max_models different algorithm configurations,
        # evaluates each one with cross-validation, and builds a leaderboard.
        aml.train(x=predictors, y=target, training_frame=main_frame)

        print(f"AutoML complete. Best model: {aml.leader.model_id}")
        print(f"Best model log-loss: {aml.leader.logloss():.6f}")
        print(f"Best model AUC:      {aml.leader.auc():.6f}")

        # Log training parameters to MLflow so you can reproduce this run.
        mlflow.log_param("max_models", args.models)
        mlflow.log_param("max_runtime_secs", args.runtime)
        mlflow.log_param("sample_frac", args.sample_frac)
        mlflow.log_param("target_column", target)
        mlflow.log_param("n_predictors", len(predictors))
        mlflow.log_param("best_model_id", aml.leader.model_id)

        # Log evaluation metrics to MLflow.
        mlflow.log_metric("log_loss", aml.leader.logloss())
        mlflow.log_metric("AUC", aml.leader.auc())

        # Save the best model as an MLflow artifact.
        # mlflow.h2o.log_model() serializes the H2O model binary
        # and uploads it to the MLflow artifact store.
        mlflow.h2o.log_model(aml.leader, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")
        print(f"Best model saved at: {model_uri}")

        # Save the AutoML leaderboard as a CSV artifact.
        # The leaderboard shows all models tried, sorted by log-loss.
        lb = get_leaderboard(aml, extra_columns='ALL').as_data_frame()
        with tempfile.TemporaryDirectory() as tmp:
            lb_path = os.path.join(tmp, 'leaderboard.csv')
            lb.to_csv(lb_path, index=False)
            mlflow.log_artifact(lb_path, artifact_path="model")
        print("Leaderboard logged as MLflow artifact.")

    # Register the model in the MLflow Model Registry.
    # This gives the model a name and a version number.
    registered = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name=model_name,
    )

    # Set the alias "champion" on this model version.
    # The backend loads the model using: models:/insurance-automl@champion
    # This alias can be moved to a new version without changing the backend code.
    client.set_registered_model_alias(
        name=model_name,
        alias=model_alias,
        version=registered.version,
    )
    print(f'Registered: "{model_name}" version {registered.version} with alias @{model_alias}')
    print("Training complete. Trainer container will now exit.")


if __name__ == "__main__":
    main()
```

---

### Step 6.5 — File: `backend/main.py`

This is the FastAPI prediction API.

Open `backend/main.py` and write:

```python
# ===========================
# FastAPI Prediction Backend
#
# What this file does:
#   - At startup: initializes H2O, connects to MLflow, loads the @champion model.
#   - POST /predict: receives a CSV, runs the model, returns JSON predictions.
#   - GET /health: returns "OK" — used by Docker's healthcheck.
#   - GET /: returns a welcome HTML page with a link to the interactive API docs.
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

# Read model configuration from environment variables.
# These are set in docker-compose.yml under the "backend" service.
MODEL_NAME  = os.getenv("MODEL_NAME",  "insurance-automl")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

app = FastAPI(title="End-to-End AutoML - Insurance Cross-Sell")

# -----------------------------------------------------------------------
# STARTUP: this code runs once when uvicorn starts the FastAPI application.
# It runs before any requests are accepted.
# -----------------------------------------------------------------------

# Start the H2O cluster inside this container.
h2o.init()

# Tell MLflow where the tracking server is.
if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)

# Load the best model from the MLflow Model Registry.
# "models:/insurance-automl@champion" means:
#   - Look up the model named "insurance-automl"
#   - Load the version tagged with the alias "champion"
# This is the model that train.py saved and tagged.
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
print(f"[startup] Loading model from: {model_uri}")
best_model = mlflow.h2o.load_model(model_uri)
print("[startup] Model loaded successfully. Ready to accept requests.")


# -----------------------------------------------------------------------
# ROUTE: POST /predict
# -----------------------------------------------------------------------
@app.post("/predict")
async def predict(file: bytes = File(...)):
    """
    Receive a CSV file, run the H2O model, return a list (or dict) of predictions.

    Input:  CSV file sent as multipart/form-data
    Output: JSON array [0, 1, 0, ...] or dict {"ID_1": 0, "ID_2": 1, ...}
    """
    print("[predict] Received prediction request")

    # Convert raw bytes to a file-like object so pandas can read it.
    file_obj = io.BytesIO(file)
    test_df   = pd.read_csv(file_obj)
    print(f"[predict] Loaded DataFrame: {len(test_df)} rows, {test_df.shape[1]} columns")

    # Convert the pandas DataFrame to an H2O frame for prediction.
    test_h2o = h2o.H2OFrame(test_df)

    # Remove the ID column if present and save it for the response.
    # We do not want the model to use the ID as a predictor.
    id_name, X_id, X_h2o = separate_id_col(test_h2o)

    # Make the test column types match the training column types.
    # Without this, H2O may throw a type mismatch error.
    X_h2o = match_col_types(X_h2o)

    # Run the model. preds is an H2OFrame with a "predict" column.
    preds = best_model.predict(X_h2o)
    print(f"[predict] Predictions generated: {preds.nrow} rows")

    # Build the response.
    if id_name is not None:
        # If there was an ID column, return a dict: {id: prediction}
        preds_list = preds.as_data_frame()['predict'].tolist()
        id_list    = X_id.as_data_frame()[id_name].tolist()
        preds_final = dict(zip(id_list, preds_list))
    else:
        # No ID column: return a plain list: [0, 1, 0, ...]
        preds_final = preds.as_data_frame()['predict'].tolist()

    json_compatible = jsonable_encoder(preds_final)
    return JSONResponse(content=json_compatible)


# -----------------------------------------------------------------------
# ROUTE: GET /health
# Used by Docker's healthcheck to determine if the service is ready.
# Returns plain text "OK" with HTTP 200 when the model is loaded.
# -----------------------------------------------------------------------
@app.get("/health")
async def health():
    return PlainTextResponse("OK")


# -----------------------------------------------------------------------
# ROUTE: GET /
# A simple welcome page. Open http://localhost:8000/ in your browser.
# -----------------------------------------------------------------------
@app.get("/")
async def root():
    content = """
    <html>
    <body style="font-family: Arial, sans-serif; padding: 40px;">
    <h2>End-to-End AutoML Pipeline — Insurance Cross-Sell</h2>
    <p>The H2O model is loaded and the FastAPI server is running.</p>
    <p><a href="/docs">Interactive API docs (Swagger UI)</a></p>
    <p><a href="/health">Health check</a></p>
    <p>Open <a href="http://localhost:8501">http://localhost:8501</a> for the Streamlit UI.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=content)
```

---

### Step 6.6 — File: `backend/utils/data_processing.py`

Open `backend/utils/data_processing.py` and write:

```python
# ===========================
# Data Processing Utilities
#
# Two helper functions used by main.py during prediction:
#   - separate_id_col: removes the ID column before prediction
#   - match_col_types: aligns test set column types with training set
# ===========================
import h2o
import json


def separate_id_col(h2o_frame):
    """
    Look for an ID column in the H2O frame.
    If found, remove it and return it separately.
    The model must not see the ID as a predictor.

    Args:
        h2o_frame: H2OFrame — the full test dataset including possible ID column.

    Returns:
        id_name (str or None): name of the ID column found, or None.
        X_id (H2OFrame or None): the ID column as its own frame, or None.
        X_h2o (H2OFrame): the dataset WITHOUT the ID column (or original if no ID).
    """
    possible_id_names = ['ID', 'Id', 'id']

    for name in possible_id_names:
        if name in h2o_frame.names:
            id_name = name
            X_id    = h2o_frame[:, id_name]       # keep the ID column separately
            X_h2o   = h2o_frame.drop(id_name)     # drop the ID from the feature set
            return id_name, X_id, X_h2o

    # No ID column found: return the original frame unchanged.
    return None, None, h2o_frame


def match_col_types(h2o_frame):
    """
    Align the column types of the test H2OFrame with those from the training set.

    When the model was trained, the training CSV had specific data types for each
    column (int, real, enum, etc.). When a user uploads a test CSV, H2O may infer
    slightly different types (e.g., a column of 0s and 1s might be read as 'enum'
    instead of 'int'). This mismatch causes H2O to refuse to make predictions.

    This function reads the saved training column types from train_col_types.json
    and converts any mismatched columns in the test frame.

    Args:
        h2o_frame: H2OFrame — the test dataset (without ID column).

    Returns:
        h2o_frame: H2OFrame — the same data with corrected column types.
    """
    # Load the column type reference saved during training.
    with open('data/processed/train_col_types.json') as f:
        train_col_types = json.load(f)

    for col_name in train_col_types:
        try:
            expected_type = train_col_types[col_name]
            actual_type   = h2o_frame.types.get(col_name)

            if actual_type is None:
                # Column not present in the test set — skip it.
                continue

            if expected_type == actual_type:
                # Types already match — nothing to do.
                continue

            # Types differ: convert the test column to match the training type.
            if expected_type == 'real' and actual_type == 'enum':
                h2o_frame[col_name] = h2o_frame[col_name].ascharacter().asnumeric()
            elif expected_type == 'real':
                h2o_frame[col_name] = h2o_frame[col_name].asnumeric()
            elif expected_type == 'int':
                h2o_frame[col_name] = h2o_frame[col_name].asfactor()
            elif expected_type == 'str':
                h2o_frame[col_name] = h2o_frame[col_name].ascharacter()

        except Exception:
            # If a column conversion fails, skip it and continue.
            # H2O will raise a clearer error at prediction time if it matters.
            pass

    return h2o_frame
```

---

### Step 6.7 — File: `backend/utils/__init__.py`

This file must exist but is empty. It tells Python that `utils/` is a package so you can write `from utils.data_processing import ...` in `main.py`.

Leave `backend/utils/__init__.py` empty.

---

### Step 6.8 — File: `frontend/Dockerfile`

Open `frontend/Dockerfile` and write:

```dockerfile
# Frontend image: Streamlit web interface
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (cached layer — avoids reinstalling on every build
# if only app.py changed but requirements did not).
COPY requirements-frontend.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all frontend source files.
COPY . /app

# Streamlit listens on this port.
EXPOSE 8501

# Start Streamlit on port 8501, bound to all network interfaces
# so Docker can forward traffic to it.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

### Step 6.9 — File: `frontend/requirements-frontend.txt`

Open `frontend/requirements-frontend.txt` and write:

```
streamlit==1.41.1
pandas==2.2.3
requests==2.32.3
```

---

### Step 6.10 — File: `frontend/app.py`

This is the entire Streamlit web interface.

Open `frontend/app.py` and write:

```python
# =========================================
# Streamlit Web Interface
#
# What this file does:
#   - Shows a description of the project and dataset.
#   - Lets the user upload a CSV file.
#   - Sends the CSV to the FastAPI backend via HTTP POST.
#   - Displays a summary, bar chart, detailed table, and optional evaluation.
#   - If the CSV contains a "Response" column, shows accuracy, precision,
#     recall, F1-score, and a confusion matrix.
# =========================================
import io
import json
import os

import pandas as pd
import requests
import streamlit as st

# Page configuration: sets the browser tab title and icon.
st.set_page_config(
    page_title="AutoML Insurance Cross-Sell",
    page_icon="🚗",
    layout="wide",
)

st.title('End-to-End AutoML Project: Insurance Cross-Sell')
st.caption('H2O AutoML + MLflow + FastAPI + Streamlit')

# The backend URL is read from an environment variable.
# In Docker, BACKEND_URL = "http://backend:8000/predict" (set in docker-compose.yml).
# Locally (outside Docker), it defaults to localhost.
ENDPOINT   = os.getenv('BACKEND_URL', 'http://localhost:8000/predict')
TARGET_COL = 'Response'
LABELS     = {1: 'Interested in vehicle insurance', 0: 'Not interested'}


# -----------------------------------------------------------------------
# Dataset description (shown in an expandable section)
# -----------------------------------------------------------------------
with st.expander('About this project and dataset', expanded=True):
    st.markdown(
        """
**Goal.** An insurance company that already sells **health insurance** wants to know
which of its existing customers are likely to also buy **vehicle insurance**.
Targeting only the interested customers makes a cross-sell campaign cheaper and
more effective.

**Dataset.** Health Insurance Cross-Sell (Kaggle). Each row is one customer.

| Feature | What it means |
|---|---|
| `Gender`, `Age` | Customer demographics |
| `Driving_License` | Does the customer hold a driving licence? (0 = No, 1 = Yes) |
| `Region_Code` | Region of the customer (one-hot encoded into many binary columns) |
| `Previously_Insured` | Does the customer already have vehicle insurance? (0 / 1) |
| `Vehicle_Age`, `Vehicle_Damage` | Age of the vehicle, whether it was damaged before |
| `Annual_Premium` | How much the customer pays for health insurance per year |
| `Policy_Sales_Channel` | Channel used to reach the customer (one-hot encoded) |
| `Vintage` | Number of days the customer has been with the company |
| **`Response`** (target) | **1** = customer is interested in vehicle insurance, **0** = not interested |

**Model.** An H2O AutoML model was trained on one-hot encoded data. It was selected
automatically based on the lowest log-loss among all candidate algorithms.

**Two ways to use this app:**
- Upload `sample_test.csv` (no `Response` column) → **predictions only**.
- Upload `sample_test_labeled.csv` (has `Response` column) → **predictions + evaluation**
  (accuracy, precision, recall, F1-score, confusion matrix).
        """
    )

st.write(
    "Upload a preprocessed test CSV (the format must match the training data, "
    "like the files in `backend/data/`) and click **Start Prediction**."
)

# -----------------------------------------------------------------------
# File uploader widget
# -----------------------------------------------------------------------
test_csv = st.file_uploader(
    'Upload test dataset (CSV)',
    type=['csv'],
    accept_multiple_files=False,
)


# -----------------------------------------------------------------------
# Metric computation (no sklearn needed — computed manually)
# -----------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """
    Compute confusion matrix counts and classification metrics from two lists.

    Args:
        y_true: list of actual labels (integers 0 or 1)
        y_pred: list of predicted labels (integers 0 or 1)

    Returns:
        tp, tn, fp, fn: confusion matrix counts
        accuracy, precision, recall, f1: classification metrics (floats 0.0–1.0)
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total           if total           else 0
    precision = tp / (tp + fp)              if (tp + fp)       else 0
    recall    = tp / (tp + fn)              if (tp + fn)       else 0
    f1        = (2 * precision * recall /
                 (precision + recall))      if (precision + recall) else 0

    return tp, tn, fp, fn, accuracy, precision, recall, f1


# -----------------------------------------------------------------------
# Main prediction logic — only runs when a file has been uploaded
# -----------------------------------------------------------------------
if test_csv:
    test_df = pd.read_csv(test_csv)

    st.subheader('Preview of Uploaded Dataset')
    st.write(test_df.head())

    has_labels = TARGET_COL in test_df.columns
    st.write(
        f"Rows: **{len(test_df)}** | "
        f"Columns: **{test_df.shape[1]}** | "
        f"Ground-truth labels: **{'yes — evaluation enabled' if has_labels else 'no — prediction only'}**"
    )

    # Convert the DataFrame back to CSV bytes to send to the backend.
    # We re-serialize from the DataFrame (not the raw uploaded file) to
    # ensure consistent encoding and formatting.
    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)
    test_bytes_obj.seek(0)   # Reset the read pointer to the beginning.

    files = {"file": ('test_dataset.csv', test_bytes_obj, "multipart/form-data")}

    if st.button('Start Prediction'):
        if len(test_df) == 0:
            st.warning("The uploaded file is empty. Please upload a non-empty CSV.")
        else:
            try:
                with st.spinner('Sending data to the prediction backend. Please wait...'):
                    response = requests.post(ENDPOINT, files=files, timeout=8000)

                # Raise an exception if the server returned an error status code.
                response.raise_for_status()
                result = response.json()

                # Build a results DataFrame from the JSON response.
                # The backend returns either a list or a dict depending on whether
                # an ID column was present.
                if isinstance(result, dict):
                    results_df = pd.DataFrame({
                        'Customer ID': list(result.keys()),
                        'Prediction':  list(result.values()),
                    })
                else:
                    results_df = pd.DataFrame({
                        'Customer #': range(1, len(result) + 1),
                        'Prediction': result,
                    })

                results_df['Prediction'] = results_df['Prediction'].astype(int)
                results_df['Result']     = results_df['Prediction'].map(
                    lambda v: LABELS.get(v, str(v))
                )

                n_total      = len(results_df)
                n_interested = int((results_df['Prediction'] == 1).sum())
                n_not        = n_total - n_interested
                pct          = (n_interested / n_total * 100) if n_total else 0

                st.success(f'Prediction complete! {n_total} customers analysed.')

                # --- Summary metrics ---
                st.subheader('Summary')
                col1, col2, col3 = st.columns(3)
                col1.metric('Customers analysed',     n_total)
                col2.metric('Interested (to target)', n_interested, f'{pct:.0f}%')
                col3.metric('Not interested',         n_not)

                st.caption(
                    f"The model predicts that **{n_interested}** of **{n_total}** customers "
                    f"(**{pct:.0f}%**) are likely interested in additional vehicle insurance. "
                    "These are the customers to prioritize for the cross-sell campaign."
                )

                # --- Bar chart ---
                st.bar_chart(results_df['Result'].value_counts(), use_container_width=True)

                # --- Evaluation (only available when ground-truth labels are present) ---
                if has_labels:
                    y_true = test_df[TARGET_COL].astype(int).tolist()
                    y_pred = results_df['Prediction'].tolist()

                    tp, tn, fp, fn, acc, prec, rec, f1 = compute_metrics(y_true, y_pred)

                    st.subheader('Model Evaluation')
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric('Accuracy',  f'{acc:.1%}')
                    m2.metric('Precision', f'{prec:.1%}')
                    m3.metric('Recall',    f'{rec:.1%}')
                    m4.metric('F1-score',  f'{f1:.1%}')

                    st.markdown('**Confusion Matrix** (rows = actual label, columns = predicted label)')
                    cm = pd.DataFrame(
                        [[tn, fp], [fn, tp]],
                        index=['Actual: Not interested (0)',   'Actual: Interested (1)'],
                        columns=['Predicted: Not interested (0)', 'Predicted: Interested (1)'],
                    )
                    st.table(cm)

                    st.markdown(f"""
**How to read the confusion matrix:**

| Cell | Count | What it means |
|---|---|---|
| True Negatives (TN) | {tn} | Correctly predicted as *not interested*. No wasted outreach. |
| True Positives (TP) | {tp} | Correctly predicted as *interested*. These are the leads the campaign should target. |
| False Positives (FP) | {fp} | Predicted *interested* but actually not. Wasted marketing calls. |
| False Negatives (FN) | {fn} | Predicted *not interested* but actually interested. Missed opportunities. |

**Precision** ({prec:.0%}): of all customers the model flags as *interested*, {prec:.0%} really are.
A low precision means many false alarms — the campaign wastes effort on uninterested customers.

**Recall** ({rec:.0%}): of all customers who truly are *interested*, the model catches {rec:.0%} of them.
A low recall means many missed opportunities — interested customers are never contacted.

**For a cross-sell campaign:** you usually want **high recall** (don't miss potential buyers)
while keeping precision high enough to avoid wasting too many calls.
                    """)

                # --- Detailed results table ---
                st.subheader('Detailed Results')
                show_cols = [results_df.columns[0], 'Result']
                if has_labels:
                    results_df['Actual']  = [LABELS.get(int(v), str(v)) for v in test_df[TARGET_COL]]
                    results_df['Correct'] = [
                        '✓' if int(a) == int(p) else '✗'
                        for a, p in zip(test_df[TARGET_COL], results_df['Prediction'])
                    ]
                    show_cols += ['Actual', 'Correct']

                st.dataframe(results_df[show_cols], use_container_width=True, hide_index=True)

                # --- Download buttons ---
                dl1, dl2 = st.columns(2)
                dl1.download_button(
                    label='Download results (CSV)',
                    data=results_df.to_csv(index=False),
                    file_name='prediction_results.csv',
                    mime='text/csv',
                )
                dl2.download_button(
                    label='Download raw predictions (JSON)',
                    data=json.dumps(result),
                    file_name='automl_predictions.json',
                    mime='application/json',
                )

            except requests.exceptions.RequestException as exc:
                st.error(
                    f"Could not reach the prediction backend at `{ENDPOINT}`. "
                    "Make sure all Docker containers are running (`docker compose ps`)."
                )
                st.exception(exc)
```

---

### Step 6.11 — File: `.gitignore`

Open `.gitignore` and write:

```
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Environment
.env
*.env

# Docker artifacts
mlruns/

# Secrets (never commit these)
*.pem
*.key
credentials.json
```

---

## 7. Final Verification Before Launch

### Step 7.1 — Count all files

```powershell
Get-ChildItem -Recurse -File | Measure-Object
```

You need at least **15 files** (11 code files + 4 data files).

### Step 7.2 — Verify every data file exists

```powershell
Test-Path backend\data\processed\train.csv            ; `
Test-Path backend\data\processed\train_col_types.json ; `
Test-Path backend\data\sample_test.csv                ; `
Test-Path backend\data\sample_test_labeled.csv
```

Every line must print `True`.

### Step 7.3 — Verify Docker is running

```powershell
docker info | Select-String "Server Version"
```

Expected: a line like `Server Version: 27.x.x`. If you see an error, open Docker Desktop and wait for it to say Running.

### Step 7.4 — Verify no ports are already in use

```powershell
netstat -ano | findstr ":5000 "
netstat -ano | findstr ":8000 "
netstat -ano | findstr ":8501 "
```

If any of these commands returns output, another program is using that port. Stop it before continuing (see Error 15.5).

---

## 8. Launch the Application

### Step 8.1 — Build the Docker images and start all containers

Make sure you are in your project folder:

```powershell
pwd
```

The path must end with `projet-automl-insurance`. If not:

```powershell
cd $HOME\Documents\projet-automl-insurance
```

Now build and start everything:

```powershell
docker compose up --build
```

**What Docker does step by step:**

| Step | What happens | Approximate time |
|---|---|---|
| 1 | Reads `backend/Dockerfile`, pulls `python:3.11-slim` if not cached | 1–3 min (first run only) |
| 2 | Installs `default-jre-headless` inside the backend image | 1–2 min (first run only) |
| 3 | Runs `pip install` for all backend packages | 2–5 min (first run only) |
| 4 | Reads `frontend/Dockerfile`, runs `pip install` for frontend packages | 1 min (first run only) |
| 5 | Starts the `mlflow` container | 5–10 sec |
| 6 | Waits for mlflow healthcheck to pass | 10–30 sec |
| 7 | Starts the `trainer` container | immediately after mlflow healthy |
| 8 | Trainer runs H2O AutoML (training) | 2–5 min |
| 9 | Trainer registers the model in MLflow, exits with code 0 | a few sec |
| 10 | Starts the `backend` container, loads the @champion model | 30–60 sec |
| 11 | Waits for backend healthcheck to pass | 15–60 sec |
| 12 | Starts the `frontend` container | immediately after backend healthy |

**Total first-run time: approximately 10–20 minutes.**

On subsequent runs (images already built, volume already has the MLflow data): **2–5 minutes**.

### Step 8.2 — Key log messages to watch for

The terminal shows logs from all four containers mixed together. Here are the messages that confirm each step succeeded:

```
mlflow-1   | [INFO] Starting gunicorn 23.x.x
mlflow-1   | ... Listening at: http://0.0.0.0:5000

trainer-1  | H2O cluster started.
trainer-1  | Loaded training frame: XXXX rows x YYY columns
trainer-1  | Sampled to 20% of original -> XXXX rows
trainer-1  | AutoML complete. Best model: XGBoost_1_AutoML_...
trainer-1  | Best model log-loss: 0.XXXXXX
trainer-1  | Best model AUC:      0.XXXXXX
trainer-1  | Registered: "insurance-automl" version 1 with alias @champion
trainer-1  | Training complete. Trainer container will now exit.

backend-1  | [startup] Loading model from: models:/insurance-automl@champion
backend-1  | [startup] Model loaded successfully. Ready to accept requests.
backend-1  | INFO:     Application startup complete.

frontend-1 | You can now view your Streamlit app in your browser.
frontend-1 |   URL: http://0.0.0.0:8501
```

> **If you see** `trainer-1 exited with code 1` — training failed. See Section 15 for solutions.

> **If you see** `trainer-1 exited with code 0` — training succeeded. 

### Step 8.3 — Run in detached mode (optional)

If you want to get your terminal back while containers keep running:

```powershell
docker compose up --build -d
```

View all logs:

```powershell
docker compose logs -f
```

View logs for only one service:

```powershell
docker compose logs trainer
docker compose logs backend
```

---

## 9. Verify All Four Services Are Running

Open a **new PowerShell window** (keep the first one showing logs).

### Step 9.1 — Check container status

```powershell
docker compose ps
```

Expected output (your image versions may differ):

```
NAME        IMAGE                        COMMAND                  STATUS
mlflow      e2e-automl-backend:latest    "mlflow server ..."      Up (healthy)
trainer     e2e-automl-backend:latest    "python train.py ..."    Exited (0)
backend     e2e-automl-backend:latest    "uvicorn main:app ..."   Up (healthy)
frontend    e2e-automl-frontend:latest   "streamlit run app..."   Up
```

Check each container:
- `mlflow` — must be `Up (healthy)`. **Never** `Up (unhealthy)`.
- `trainer` — must be `Exited (0)`. Code 0 = success. Code 1 = failure.
- `backend` — must be `Up (healthy)`.
- `frontend` — must be `Up`.

### Step 9.2 — Verify the MLflow server responds

```powershell
Invoke-WebRequest -Uri http://localhost:5000/health -UseBasicParsing
```

Expected: `StatusCode : 200`.

Now open `http://localhost:5000` in your browser. You should see:
- The MLflow dashboard.
- An experiment named `automl-insurance` in the left sidebar.
- A run inside that experiment.

### Step 9.3 — Verify the FastAPI backend responds

```powershell
Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing
```

Expected: `StatusCode : 200` and content `OK`.

Open `http://localhost:8000/docs` in your browser. You should see:
- The Swagger UI with three routes: `POST /predict`, `GET /health`, `GET /`.

### Step 9.4 — Verify the Streamlit frontend is up

Open `http://localhost:8501` in your browser.

You should see:
- The page title: **End-to-End AutoML Project: Insurance Cross-Sell**.
- An expandable section with the project description.
- A **Browse files** button.

---

## 10. Make Predictions

### Step 10.1 — Prediction without labels (output only)

1. Open `http://localhost:8501` in your browser.
2. Click **Browse files**.
3. Navigate to your project folder, then to `backend\data\`.
4. Select `sample_test.csv`.
5. The preview shows the first 5 rows of the file.
6. Click **Start Prediction**.
7. Wait for the spinner to finish.

Expected results on the screen:
- A green success message: `Prediction complete! X customers analysed.`
- Three metric boxes: total customers, interested count (with percentage), not-interested count.
- A bar chart showing the split between interested and not interested.
- A detailed table with a `Result` column per customer.
- Two download buttons.

### Step 10.2 — Prediction with labels (full evaluation)

1. Click **Browse files** again.
2. Select `sample_test_labeled.csv` this time.
3. Notice the page now says: `Ground-truth labels: yes — evaluation enabled`.
4. Click **Start Prediction**.

Expected extra results:
- A **Model Evaluation** section with Accuracy, Precision, Recall, F1-score.
- A **Confusion Matrix** table.
- A detailed explanation of each cell in the confusion matrix.
- The table now has `Actual` and `Correct` (✓ or ✗) columns.

### Step 10.3 — Test the API directly from PowerShell

```powershell
$file    = "backend\data\sample_test.csv"
$uri     = "http://localhost:8000/predict"

$form     = @{ file = Get-Item $file }
$response = Invoke-RestMethod -Uri $uri -Method Post -Form $form
$response | ConvertTo-Json | Select-Object -First 30
```

Expected: a JSON array of 0s and 1s.

---

## 11. Explore the MLflow Dashboard

### Step 11.1 — Find the training run

1. Open `http://localhost:5000` in your browser.
2. Click on `automl-insurance` in the experiment list on the left.
3. Click the run in the table (there should be one row).

### Step 11.2 — Explore the run details

On the run page, check these tabs:

**Parameters tab:**
- `max_models`: 5
- `max_runtime_secs`: 120
- `sample_frac`: 0.2
- `target_column`: Response
- `best_model_id`: the name of the winning algorithm (e.g., `XGBoost_1_AutoML_...`)

**Metrics tab:**
- `log_loss`: a decimal between 0 and 1. Lower is better.
- `AUC`: a decimal between 0.5 and 1.0. Closer to 1 is better. 0.5 means the model is no better than random guessing.

**Artifacts tab:**
- `model/`: the saved H2O model binary.
- `model/leaderboard.csv`: the comparison of all models AutoML tried.

### Step 11.3 — Find the registered model

1. Click **Models** in the top navigation bar of the MLflow UI.
2. Click `insurance-automl`.
3. You should see **Version 1** with the alias `@champion`.

---

## 12. Read the Logs

Reading logs is an essential skill for debugging and understanding what your application does.

### Step 12.1 — Follow all logs live

```powershell
docker compose logs -f
```

Press `Ctrl+C` to stop the live tail (containers keep running).

### Step 12.2 — See logs for one service only

```powershell
docker compose logs mlflow
docker compose logs trainer
docker compose logs backend
docker compose logs frontend
```

### Step 12.3 — See what the backend logs when a prediction runs

In one PowerShell window, start following backend logs:

```powershell
docker compose logs backend -f
```

In your browser, go to `http://localhost:8501`, upload `sample_test.csv`, and click **Start Prediction**.

You will see these lines appear in the log:

```
backend-1  | [predict] Received prediction request
backend-1  | [predict] Loaded DataFrame: 50 rows, 183 columns
backend-1  | [predict] Predictions generated: 50 rows
backend-1  | INFO:     127.0.0.1:XXXXX - "POST /predict HTTP/1.1" 200 OK
```

### Step 12.4 — Check that the trainer finished successfully

```powershell
docker compose logs trainer --no-log-prefix
```

Look for the last line: `Training complete. Trainer container will now exit.`

---

## 13. Stop the Application

### Step 13.1 — Stop all containers (keep the MLflow data)

```powershell
docker compose down
```

This stops and removes the containers. The `mlflow_data` volume is kept on disk. Next time you run `docker compose up` (without `--build`), the model is already registered and the trainer **will not** run again.

### Step 13.2 — Stop and delete the MLflow volume (force retraining)

```powershell
docker compose down -v
```

The `-v` flag deletes the `mlflow_data` volume. Next time you run `docker compose up --build`, the trainer runs again from scratch and registers a new model version.

### Step 13.3 — Confirm everything stopped

```powershell
docker compose ps
```

Expected: no containers listed, or all at `Exited` status.

### Step 13.4 — Remove the built images (full cleanup, optional)

```powershell
docker rmi e2e-automl-backend:latest e2e-automl-frontend:latest
```

This frees disk space. The next `docker compose up --build` will rebuild everything from scratch.

---

## 14. Required Screenshots

Capture these screenshots for your assignment report.

| # | What to capture | What it must show |
|---|---|---|
| 1 | `docker compose up --build` in progress | The terminal with log output from all four services starting |
| 2 | `docker compose ps` | All four containers, `trainer` at `Exited (0)`, `mlflow` and `backend` at `Up (healthy)` |
| 3 | MLflow dashboard at `http://localhost:5000` | The `automl-insurance` experiment with a run |
| 4 | MLflow run detail page | The Parameters and Metrics tabs with values |
| 5 | MLflow Models page | `insurance-automl` model with `@champion` alias |
| 6 | Streamlit UI at `http://localhost:8501` | The app loaded with the file upload widget visible |
| 7 | Prediction results with `sample_test_labeled.csv` | Summary metrics, bar chart, confusion matrix |
| 8 | `docker compose down` | The terminal output confirming all containers stopped |

**Forbidden in screenshots:** never show an AWS secret key, API token, password, or any credential. Blur or crop them if visible.

---

## 15. Common Errors and Direct Solutions

### Error 15.1 — `error during connect: ... Is the docker daemon running?`

**Cause:** Docker Desktop is not running.

**Solution:**
1. Open Docker Desktop from the Start menu.
2. Wait until the whale icon is green and the status says **Running**.
3. Retry your command.

**Verify:** `docker info | Select-String "Server Version"`

---

### Error 15.2 — `trainer-1 exited with code 1`

**Cause:** The training script crashed. Most common reasons: not enough RAM, missing data file, or MLflow server not reachable.

**Solution:**

Step 1 — Read the trainer logs:

```powershell
docker compose logs trainer
```

Step 2 — If you see `java.lang.OutOfMemoryError` or `H2O cluster is too unhealthy`:
- Open Docker Desktop → Settings → Resources → Memory.
- Increase to at least 4 GB (6 GB recommended). Click Apply & restart.
- Then: `docker compose down -v && docker compose up --build`

Step 3 — If you see `FileNotFoundError: data/processed/train.csv`:
- Verify the file exists: `Test-Path backend\data\processed\train.csv`
- If it prints `False`, copy the file to `backend\data\processed\` and retry.

Step 4 — If you see `ConnectionRefusedError` or `Connection refused` for MLflow:
- The trainer started before MLflow was ready. This should not happen with the health check, but if it does: `docker compose down && docker compose up --build`

---

### Error 15.3 — `backend` stays `Up (unhealthy)` or never starts

**Cause:** The backend cannot load the model from MLflow. Most common reason: the trainer failed (see Error 15.2).

**Solution:**
1. Fix the trainer failure first.
2. Then: `docker compose down -v && docker compose up --build`

**Verify:**

```powershell
docker compose logs backend | Select-String -Pattern "Loading model|Error|Exception"
```

---

### Error 15.4 — Streamlit shows `Could not reach the prediction backend`

**Cause:** You uploaded a file and clicked Start Prediction, but the backend is not yet healthy or crashed.

**Solution:**
1. Check `docker compose ps` — is `backend` listed as `Up (healthy)`?
2. If it is not healthy yet, wait 1–2 more minutes and try again.
3. If it crashed, read the logs: `docker compose logs backend`

---

### Error 15.5 — `Bind for 0.0.0.0:5000 failed: port is already allocated`

**Cause:** Another program on your machine is already using that port.

**Solution:**

Find what is using port 5000 (or 8000 or 8501):

```powershell
netstat -ano | findstr ":5000 "
```

The last column is the Process ID (PID). Stop that process:

```powershell
Stop-Process -Id <PID> -Force
```

Then retry `docker compose up --build`.

---

### Error 15.6 — `No module named 'utils'` in backend logs

**Cause:** The file `backend/utils/__init__.py` is missing.

**Solution:**

```powershell
New-Item backend\utils\__init__.py -ItemType File
docker compose up --build
```

---

### Error 15.7 — Images rebuild from scratch every time

**Cause:** Normal behavior when a source file changed. Docker invalidates the cache when any layer input changes.

**If you want to skip the rebuild** (use existing images):

```powershell
docker compose up
```

**If you want to force a full rebuild:**

```powershell
docker compose build --no-cache
docker compose up
```

---

### Error 15.8 — Trainer runs again even though the model was already trained

**Cause:** You used `docker compose down -v` (which deletes the volume) or the volume was never created.

**How the caching works:** when you run `docker compose down` (without `-v`), the `mlflow_data` volume is kept. On the next `docker compose up`, the trainer starts but the MLflow server already has the registered model. The trainer creates a **new version** but registers it as `@champion`.

To avoid retraining: use `docker compose down` (no `-v`).

---

## 16. Assignment B — Two Code Changes to Implement

> This section is for students assigned **Assignment B** only.

After you have the base application working exactly as described above, implement these two changes.

---

### Change 1 — Add a new metric: mean prediction score

**Location:** `frontend/app.py`

**What to change:** After the bar chart, add a new line that shows the average of all predictions as a percentage. For example: `Average prediction score: 0.27 (27% of customers are flagged as interested)`.

**Hint:** `results_df['Prediction'].mean()` gives you the fraction of predictions equal to 1.

**How to verify:** After making the change, run `docker compose up --build`. Upload either test file. Below the bar chart, a new line must appear showing the mean score.

**Screenshot required:** Capture the Streamlit results page showing your new mean score line.

---

### Change 2 — Increase the number of models trained

**Location:** `docker-compose.yml`

**What to change:** In the `trainer` service, change:

```yaml
AUTOML_MAX_MODELS: "5"
AUTOML_MAX_RUNTIME_SECS: "120"
AUTOML_SAMPLE_FRAC: "0.2"
```

to:

```yaml
AUTOML_MAX_MODELS: "10"
AUTOML_MAX_RUNTIME_SECS: "300"
AUTOML_SAMPLE_FRAC: "0.5"
```

**Why this matters:** AutoML now tries 10 models instead of 5, runs for up to 5 minutes instead of 2, and uses 50% of the training data instead of 20%. The resulting model should have a better AUC.

**How to verify:**
1. Stop the stack and delete the volume: `docker compose down -v`
2. Rebuild and restart: `docker compose up --build`
3. Open `http://localhost:5000`.
4. Check the AUC metric in the new run. It should be equal to or higher than the previous run.

**Screenshot required:** Capture the MLflow metrics page showing the new AUC value alongside the `max_models = 10` parameter.

**In your CHANGES.md file, write:**
- Which file you changed.
- Which lines exactly you changed.
- What the original values were and what the new values are.
- Why a higher `AUTOML_SAMPLE_FRAC` produces a better (or worse) model, and why.

---

## 17. Final Summary

- You installed Docker Desktop and verified it runs.
- You created a complete multi-service project structure from scratch.
- You wrote ten files by hand, including two Dockerfiles, a training script, a FastAPI API, Streamlit UI, and Docker Compose configuration.
- You understood the role of each container and why the startup order matters.
- You launched four containers with a single command.
- You verified each service using PowerShell commands and your browser.
- You made predictions in two modes: without labels (predictions only) and with labels (predictions + evaluation).
- You read Docker logs to understand what happens inside the containers.
- You stopped the application cleanly with `docker compose down`.

**Phrase to remember:** A Machine Learning project is not just a notebook. It is a pipeline: preprocessed data, automated training, model registry, prediction API, and a user interface — all orchestrated, reproducible, and running with a single command.

---

## 18. Official References

- [Docker Compose documentation](https://docs.docker.com/compose/)
- [H2O AutoML documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [MLflow Model Registry documentation](https://mlflow.org/docs/latest/model-registry.html)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [python-multipart (required for file uploads in FastAPI)](https://pypi.org/project/python-multipart/)
- [Kaggle — Health Insurance Cross-Sell Prediction dataset](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)
