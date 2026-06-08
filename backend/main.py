# ===========================
# Module: Backend setup (H2O + MLflow + FastAPI)
# Loads the @champion model from the MLflow Model Registry and serves predictions.
# Original author: Kenneth Leung (modernised for MLflow 2.x + Model Registry)
# ===========================
# Run locally: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
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

# Initiate H2O instance and point MLflow at the tracking server
h2o.init()
if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)

# Load the best model from the MLflow Model Registry (alias-based reference)
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

    # Separate ID column (if any)
    id_name, X_id, X_h2o = separate_id_col(test_h2o)

    # Match test set column types with train set
    X_h2o = match_col_types(X_h2o)

    # Generate predictions with best model (output is H2O frame)
    preds = best_model.predict(X_h2o)

    # Apply processing if dataset has an ID column
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
