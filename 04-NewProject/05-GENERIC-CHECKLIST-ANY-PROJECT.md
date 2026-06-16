# Generic Checklist — Build Any AutoML Project With This Stack
**Use this for any dataset. Follow the steps in order. Do not skip.**

---

## PHASE 1 — Define the Problem

**Step 1 — Write the business question in one sentence.**

Answer these three questions before touching any code:

```
What am I predicting?       Example: "Will this customer churn?"
What does class 1 mean?     Example: "Customer cancels their subscription"
What does class 0 mean?     Example: "Customer stays active"
```

Write it down. You will reuse these exact sentences in your Streamlit interface.

---

**Step 2 — Choose your dataset and download it.**

- Find a dataset on [Kaggle](https://www.kaggle.com/datasets) or [UCI ML Repository](https://archive.ics.uci.edu/).
- Download the CSV file(s).
- Check that it has a clear binary target column (values must be 0/1 or Yes/No or two distinct categories).
- Check that it has at least 500 rows (less than that and AutoML cannot learn reliably).

---

**Step 3 — Identify your target column.**

Open the CSV in Excel or a notebook and find:

```
Target column name:   _______________
Value that means YES (class 1):  _______________
Value that means NO  (class 0):  _______________
Number of rows:       _______________
Number of columns:    _______________
```

Write these down. You will use them in `train.py` (`--target`) and `app.py` (`LABELS`).

---

## PHASE 2 — Prepare the Data

**Step 4 — Open a Jupyter notebook and load the raw CSV.**

```python
import pandas as pd
df = pd.read_csv('your_raw_file.csv')
df.shape          # check rows and columns
df.dtypes         # check column types
df.isnull().sum() # check missing values
df.head()         # see the first 5 rows
```

---

**Step 5 — Clean the data.**

Do these in order:

- [ ] Drop identifier columns (columns like `ID`, `customerID`, `patient_nbr` — not predictors).
- [ ] Handle missing values — either drop rows, fill with median (numbers), or fill with mode (categories).
- [ ] Fix wrong types — e.g., `TotalCharges` stored as string instead of float.
- [ ] Remove extreme outliers if needed (e.g., ratio columns with values > 1000).

```python
df = df.drop(columns=['ID', 'CustomerID'])          # drop identifiers
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
```

---

**Step 6 — Encode the target column as 0 and 1.**

```python
# If target is Yes/No:
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# If target is a number already (e.g., 0 and 1):
# nothing to do

# If you need to create the target from another column:
df['pass'] = (df['G3'] >= 10).astype(int)
df = df.drop(columns=['G3'])   # remove the source column so it can't leak
```

---

**Step 7 — One-hot encode all categorical columns.**

H2O can handle categoricals, but one-hot encoding is safer and more explicit.

```python
categorical_cols = ['gender', 'Contract', 'PaymentMethod', 'InternetService']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
```

After encoding, verify:

```python
df.dtypes.value_counts()   # should be mostly int64 and float64
df.shape                   # columns count increases after one-hot encoding
```

---

**Step 8 — Split into train and test sets and save.**

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['YOUR_TARGET'])

# Save training set (used by the trainer container)
train_df.to_csv('train.csv', index=False)

# Save test set without labels (used for predictions in the UI)
test_df.drop(columns=['YOUR_TARGET']).to_csv('sample_test.csv', index=False)

# Save test set with labels (used for evaluation in the UI)
test_df.to_csv('sample_test_labeled.csv', index=False)

print(f"Train: {len(train_df)} rows")
print(f"Test:  {len(test_df)} rows")
```

---

**Step 9 — Copy the three CSV files to your project.**

```
train.csv               → backend/data/processed/train.csv
sample_test.csv         → backend/data/sample_test.csv
sample_test_labeled.csv → backend/data/sample_test_labeled.csv
```

Do NOT copy `train_col_types.json` — it is generated automatically by `train.py`.

---

## PHASE 3 — Create the Project Structure

**Step 10 — Create the folder structure.**

```powershell
mkdir my-project-name
cd my-project-name
mkdir backend
mkdir backend\utils
mkdir backend\data
mkdir backend\data\processed
mkdir frontend
```

---

**Step 11 — Create all empty files.**

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

---

## PHASE 4 — Write the Files That Never Change

> These four files are **identical** for every project. Copy them as-is.

---

**Step 12 — Write `backend/Dockerfile`.**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

**Step 13 — Write `backend/requirements-backend.txt`.**

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

---

**Step 14 — Write `frontend/Dockerfile`.**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements-frontend.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

**Step 15 — Write `frontend/requirements-frontend.txt`.**

```
streamlit==1.41.1
pandas==2.2.3
requests==2.32.3
```

---

**Step 16 — Write `backend/utils/__init__.py`.**

Leave it empty. It just needs to exist.

---

**Step 17 — Write `backend/utils/data_processing.py`.**

```python
import h2o
import json


def separate_id_col(h2o_frame):
    possible_id_names = ['ID', 'Id', 'id']
    for name in possible_id_names:
        if name in h2o_frame.names:
            X_id  = h2o_frame[:, name]
            X_h2o = h2o_frame.drop(name)
            return name, X_id, X_h2o
    return None, None, h2o_frame


def match_col_types(h2o_frame):
    with open('data/processed/train_col_types.json') as f:
        train_col_types = json.load(f)
    for col in train_col_types:
        try:
            if train_col_types[col] == h2o_frame.types.get(col):
                continue
            if train_col_types[col] == 'real' and h2o_frame.types.get(col) == 'enum':
                h2o_frame[col] = h2o_frame[col].ascharacter().asnumeric()
            elif train_col_types[col] == 'real':
                h2o_frame[col] = h2o_frame[col].asnumeric()
            elif train_col_types[col] == 'int':
                h2o_frame[col] = h2o_frame[col].asfactor()
            elif train_col_types[col] == 'str':
                h2o_frame[col] = h2o_frame[col].ascharacter()
        except Exception:
            pass
    return h2o_frame
```

---

**Step 18 — Write `backend/main.py`.**

```python
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

MODEL_NAME   = os.getenv("MODEL_NAME",  "my-automl-model")
MODEL_ALIAS  = os.getenv("MODEL_ALIAS", "champion")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

app = FastAPI(title="AutoML Prediction API")

h2o.init()
if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)

model_uri  = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
best_model = mlflow.h2o.load_model(model_uri)


@app.post("/predict")
async def predict(file: bytes = File(...)):
    file_obj = io.BytesIO(file)
    test_df  = pd.read_csv(file_obj)
    test_h2o = h2o.H2OFrame(test_df)

    id_name, X_id, X_h2o = separate_id_col(test_h2o)
    X_h2o = match_col_types(X_h2o)
    preds = best_model.predict(X_h2o)

    if id_name is not None:
        preds_list  = preds.as_data_frame()['predict'].tolist()
        id_list     = X_id.as_data_frame()[id_name].tolist()
        preds_final = dict(zip(id_list, preds_list))
    else:
        preds_final = preds.as_data_frame()['predict'].tolist()

    return JSONResponse(content=jsonable_encoder(preds_final))


@app.get("/health")
async def health():
    return PlainTextResponse("OK")


@app.get("/")
async def root():
    return HTMLResponse("<h2>AutoML API is running. <a href='/docs'>API docs</a></h2>")
```

---

**Step 19 — Write `backend/train.py`.**

```python
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
    value = os.getenv(name)
    return value if value not in (None, "") else default


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',        default=env('EXPERIMENT_NAME', 'automl-experiment'), type=str)
    parser.add_argument('--target',      required=True, type=str)
    parser.add_argument('--models',      default=int(env('AUTOML_MAX_MODELS', '10')),         type=int)
    parser.add_argument('--runtime',     default=int(env('AUTOML_MAX_RUNTIME_SECS', '0')),    type=int)
    parser.add_argument('--sample-frac', default=float(env('AUTOML_SAMPLE_FRAC', '1.0')),     type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    tracking_uri = env('MLFLOW_TRACKING_URI', None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name  = env('MODEL_NAME',  'my-automl-model')
    model_alias = env('MODEL_ALIAS', 'champion')

    h2o.init()
    client = MlflowClient()

    experiment = client.get_experiment_by_name(args.name)
    if experiment is None:
        mlflow.create_experiment(args.name)
    mlflow.set_experiment(args.name)

    frame = h2o.import_file(path='data/processed/train.csv')

    if 0 < args.sample_frac < 1.0:
        frame = frame.split_frame(ratios=[args.sample_frac], seed=42)[0]
        print(f"Sampled to {args.sample_frac:.0%} -> {frame.nrow} rows")

    with open('data/processed/train_col_types.json', 'w') as fp:
        json.dump(frame.types, fp)

    target     = args.target
    predictors = [c for c in frame.col_names if c != target]
    frame[target] = frame[target].asfactor()

    with mlflow.start_run() as run:
        aml = H2OAutoML(
            max_models=args.models,
            max_runtime_secs=args.runtime,
            seed=42,
            balance_classes=True,
            sort_metric='logloss',
            exclude_algos=['GLM', 'DRF'],
        )
        aml.train(x=predictors, y=target, training_frame=frame)

        mlflow.log_param("max_models",        args.models)
        mlflow.log_param("max_runtime_secs",  args.runtime)
        mlflow.log_param("sample_frac",       args.sample_frac)
        mlflow.log_param("target_column",     target)
        mlflow.log_metric("log_loss",         aml.leader.logloss())
        mlflow.log_metric("AUC",              aml.leader.auc())

        mlflow.h2o.log_model(aml.leader, artifact_path="model")
        model_uri_run = mlflow.get_artifact_uri("model")

        lb = get_leaderboard(aml, extra_columns='ALL').as_data_frame()
        with tempfile.TemporaryDirectory() as tmp:
            lb_path = os.path.join(tmp, 'leaderboard.csv')
            lb.to_csv(lb_path, index=False)
            mlflow.log_artifact(lb_path, artifact_path="model")

    registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
    client.set_registered_model_alias(model_name, model_alias, registered.version)
    print(f'Done. Registered "{model_name}" v{registered.version} @{model_alias}')


if __name__ == "__main__":
    main()
```

---

## PHASE 5 — Write the Two Files That Change Per Project

---

**Step 20 — Write `docker-compose.yml`.**

Copy the template below. Change only the **four marked lines**.

```yaml
services:

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

  trainer:
    build: ./backend
    image: e2e-automl-backend:latest
    command: ["python", "train.py", "--target", "YOUR_TARGET_COLUMN"]  # <-- CHANGE THIS
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MODEL_NAME: your-project-automl                                  # <-- CHANGE THIS
      MODEL_ALIAS: champion
      AUTOML_MAX_MODELS: "5"                                           # <-- CHANGE IF NEEDED
      AUTOML_MAX_RUNTIME_SECS: "120"                                   # <-- CHANGE IF NEEDED
      AUTOML_SAMPLE_FRAC: "0.2"
    depends_on:
      mlflow:
        condition: service_healthy
    restart: "no"
    networks:
      - project_network

  backend:
    build: ./backend
    image: e2e-automl-backend:latest
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MODEL_NAME: your-project-automl                                  # <-- SAME AS ABOVE
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

---

**Step 21 — Write `frontend/app.py`.**

Copy the template below. Change only the **three marked lines**.

```python
import io
import json
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AutoML Prediction App", page_icon="🤖", layout="wide")
st.title('YOUR PROJECT TITLE HERE')                        # <-- CHANGE THIS
st.caption('H2O AutoML + MLflow + FastAPI + Streamlit')

ENDPOINT   = os.getenv('BACKEND_URL', 'http://localhost:8000/predict')
TARGET_COL = 'YOUR_TARGET_COLUMN'                          # <-- CHANGE THIS
LABELS     = {1: 'Your label for YES', 0: 'Your label for NO'}  # <-- CHANGE THIS


with st.expander('About this project', expanded=True):
    st.markdown("""
**Goal.** Describe your business problem here in 2-3 sentences.

**Dataset.** Describe your dataset here.

**Two ways to use this app:**
- Upload a file **without** the target column → predictions only.
- Upload a file **with** the target column → predictions + full evaluation.
    """)

st.write("Upload a preprocessed CSV and click **Start Prediction**.")

test_csv = st.file_uploader('Upload dataset (CSV)', type=['csv'])


def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total                        if total           else 0
    precision = tp / (tp + fp)                           if (tp + fp)       else 0
    recall    = tp / (tp + fn)                           if (tp + fn)       else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return tp, tn, fp, fn, accuracy, precision, recall, f1


if test_csv:
    test_df    = pd.read_csv(test_csv)
    has_labels = TARGET_COL in test_df.columns

    st.subheader('Preview')
    st.write(test_df.head())
    st.write(f"Rows: **{len(test_df)}** | Columns: **{test_df.shape[1]}** | "
             f"Labels present: **{'yes' if has_labels else 'no'}**")

    buf = io.BytesIO()
    test_df.to_csv(buf, index=False)
    buf.seek(0)
    files = {"file": ('data.csv', buf, "multipart/form-data")}

    if st.button('Start Prediction'):
        try:
            with st.spinner('Running prediction...'):
                r = requests.post(ENDPOINT, files=files, timeout=8000)
            r.raise_for_status()
            result = r.json()

            if isinstance(result, dict):
                results_df = pd.DataFrame({'ID': list(result.keys()), 'Prediction': list(result.values())})
            else:
                results_df = pd.DataFrame({'#': range(1, len(result) + 1), 'Prediction': result})

            results_df['Prediction'] = results_df['Prediction'].astype(int)
            results_df['Result']     = results_df['Prediction'].map(lambda v: LABELS.get(v, str(v)))

            n_total = len(results_df)
            n_pos   = int((results_df['Prediction'] == 1).sum())
            n_neg   = n_total - n_pos
            pct     = (n_pos / n_total * 100) if n_total else 0

            st.success(f'Done! {n_total} records analysed.')

            c1, c2, c3 = st.columns(3)
            c1.metric('Total',    n_total)
            c2.metric(LABELS[1],  n_pos, f'{pct:.0f}%')
            c3.metric(LABELS[0],  n_neg)

            st.bar_chart(results_df['Result'].value_counts(), use_container_width=True)

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

                st.markdown('**Confusion Matrix**')
                cm = pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=[f'Actual: {LABELS[0]}', f'Actual: {LABELS[1]}'],
                    columns=[f'Predicted: {LABELS[0]}', f'Predicted: {LABELS[1]}'],
                )
                st.table(cm)

            st.subheader('Detailed Results')
            show = [results_df.columns[0], 'Result']
            if has_labels:
                results_df['Actual']  = [LABELS.get(int(v), str(v)) for v in test_df[TARGET_COL]]
                results_df['Correct'] = ['✓' if int(a) == int(p) else '✗'
                                         for a, p in zip(test_df[TARGET_COL], results_df['Prediction'])]
                show += ['Actual', 'Correct']
            st.dataframe(results_df[show], use_container_width=True, hide_index=True)

            d1, d2 = st.columns(2)
            d1.download_button('Download CSV', results_df.to_csv(index=False), 'results.csv', 'text/csv')
            d2.download_button('Download JSON', json.dumps(result), 'results.json', 'application/json')

        except requests.exceptions.RequestException as exc:
            st.error(f"Cannot reach the backend at {ENDPOINT}. Check that all containers are running.")
            st.exception(exc)
```

---

## PHASE 6 — Launch and Verify

**Step 22 — Build and start all containers.**

```powershell
docker compose up --build
```

Wait for these four messages (in any order):

```
mlflow  → ... Listening at: http://0.0.0.0:5000
trainer → Done. Registered "your-project-automl" v1 @champion
backend → Application startup complete.
frontend → You can now view your Streamlit app in your browser.
```

---

**Step 23 — Check container status.**

Open a second PowerShell window:

```powershell
docker compose ps
```

Expected:

```
mlflow    Up (healthy)
trainer   Exited (0)       ← must be 0, not 1
backend   Up (healthy)
frontend  Up
```

---

**Step 24 — Check each URL in your browser.**

| URL | What you should see |
|---|---|
| `http://localhost:5000` | MLflow dashboard with your experiment and run |
| `http://localhost:8000/docs` | Swagger UI with `/predict` and `/health` routes |
| `http://localhost:8501` | Your Streamlit app with the title you set |

---

**Step 25 — Make a test prediction.**

1. Open `http://localhost:8501`.
2. Click **Browse files**.
3. Upload `backend/data/sample_test.csv`.
4. Click **Start Prediction**.
5. Confirm you see results (summary, bar chart, table).
6. Upload `backend/data/sample_test_labeled.csv`.
7. Click **Start Prediction**.
8. Confirm you see the confusion matrix and evaluation metrics.

---

**Step 26 — Verify the MLflow model registry.**

1. Open `http://localhost:5000`.
2. Click **Models** in the top menu.
3. Confirm your model name appears (e.g., `your-project-automl`).
4. Click it and confirm version 1 has the alias `@champion`.

---

## PHASE 7 — Stop and Clean Up

**Step 27 — Stop the application (keep the trained model).**

```powershell
docker compose down
```

The `mlflow_data` volume is kept. Next `docker compose up` reuses the trained model — no retraining.

---

**Step 28 — Stop and reset everything (force retraining next time).**

```powershell
docker compose down -v
```

The `-v` flag deletes the volume. Next `docker compose up --build` retrains from scratch.

---

## Full Summary — All 28 Steps in Order

No step is skipped. The earlier version of this checklist grouped some steps, which made it look like numbers were missing. Here is the **complete list, in order**. The right column tells you whether the step changes per project or is always identical.

```
PHASE 1 — DEFINE THE PROBLEM
[ ] Step 1   Write the business question                                  (changes)
[ ] Step 2   Download the dataset                                         (changes)
[ ] Step 3   Identify the target column name                             (changes)

PHASE 2 — PREPARE THE DATA
[ ] Step 4   Load and explore the data in a notebook                     (changes)
[ ] Step 5   Clean the data (drop IDs, handle nulls)                     (changes)
[ ] Step 6   Encode the target as 0 / 1                                  (changes)
[ ] Step 7   One-hot encode categorical columns                         (changes)
[ ] Step 8   Split and save train / sample_test / sample_test_labeled    (changes)
[ ] Step 9   Copy the 3 CSV files into backend/data/                     (changes)

PHASE 3 — CREATE THE PROJECT STRUCTURE
[ ] Step 10  Create the folder structure (mkdir)                         (identical)
[ ] Step 11  Create all empty files (New-Item)                           (identical)

PHASE 4 — WRITE THE FILES THAT NEVER CHANGE
[ ] Step 12  backend/Dockerfile                                          (identical)
[ ] Step 13  backend/requirements-backend.txt                           (identical)
[ ] Step 14  frontend/Dockerfile                                         (identical)
[ ] Step 15  frontend/requirements-frontend.txt                         (identical)
[ ] Step 16  backend/utils/__init__.py  (empty)                         (identical)
[ ] Step 17  backend/utils/data_processing.py                           (identical)
[ ] Step 18  backend/main.py                                            (identical)
[ ] Step 19  backend/train.py                                           (identical)

PHASE 5 — WRITE THE TWO FILES THAT CHANGE PER PROJECT
[ ] Step 20  docker-compose.yml: change --target, MODEL_NAME            (changes)
[ ] Step 21  frontend/app.py: change TITLE, TARGET_COL, LABELS          (changes)

PHASE 6 — LAUNCH AND VERIFY
[ ] Step 22  docker compose up --build
[ ] Step 23  docker compose ps   ->  trainer Exited (0)
[ ] Step 24  Open the 3 URLs in the browser (8501 / 8000 / 5000)
[ ] Step 25  Upload test CSV, confirm predictions
[ ] Step 26  Open MLflow, confirm the @champion model

PHASE 7 — STOP AND CLEAN UP
[ ] Step 27  docker compose down        (keeps the trained model)
[ ] Step 28  docker compose down -v     (optional: full reset, forces retraining)
```

### Why some steps were "missing" before

They were not actually missing — the previous summary **grouped** them to highlight only what you change per project:

- **Steps 10–11** (create folders/files) and **Steps 12–19** (write the 8 unchanged files) are the same for every project, so they were collapsed into one line.
- **Step 28** (`down -v`) is optional, so it was left off the short list.

For a **brand-new project**, you do all 28 steps once. For a **second project reusing this stack**, you only redo the steps marked `(changes)` — Steps 1–9, 20, 21 — and reuse everything marked `(identical)` by copying the files from your first project.

**That is it. 28 steps. Same stack. Any binary classification problem.**
