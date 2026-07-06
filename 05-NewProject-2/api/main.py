"""
API FastAPI de service (serving) pour le projet Wine Quality MLOps.

Cette API sert d'intermediaire entre l'interface Streamlit et MLflow :
- elle interroge le serveur de tracking MLflow pour lister experiences et runs,
- elle charge les modeles enregistres et expose une route de prediction,
- elle calcule des statistiques sur le dataset pour construire des sliders realistes.

Elle N'entraine aucun modele : l'entrainement reste la responsabilite du service `trainer`.
Cette separation entrainement / service est une bonne pratique MLOps.
"""

import os
from functools import lru_cache
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DATA_PATH = os.getenv("DATA_PATH", "data/red-wine-quality.csv")
# Nom sous lequel le trainer enregistre le modele (voir trainer/train.py).
MODEL_ARTIFACT_NAME = os.getenv("MODEL_ARTIFACT_NAME", "my_new_model_1")

mlflow.set_tracking_uri(TRACKING_URI)

# Colonnes exactes du CSV (avec espaces). L'ordre est important : le modele
# scikit-learn a ete entraine sur ces colonnes, dans cet ordre.
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Correspondance clef d'API (snake_case, compatible JSON) -> nom de colonne CSV.
API_KEY_TO_COLUMN = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "chlorides": "chlorides",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
    "density": "density",
    "ph": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}

app = FastAPI(
    title="Wine Quality MLOps API",
    description=(
        "API de service des modeles ElasticNet / Ridge / Lasso entraines "
        "et enregistres dans MLflow. Sert l'interface Streamlit."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Modeles Pydantic
# --------------------------------------------------------------------------- #


class WineFeatures(BaseModel):
    """Les 11 caracteristiques physico-chimiques d'un vin rouge."""

    fixed_acidity: float = Field(..., description="Acidite fixe (g/dm3)")
    volatile_acidity: float = Field(..., description="Acidite volatile (g/dm3)")
    citric_acid: float = Field(..., description="Acide citrique (g/dm3)")
    residual_sugar: float = Field(..., description="Sucre residuel (g/dm3)")
    chlorides: float = Field(..., description="Chlorures (g/dm3)")
    free_sulfur_dioxide: float = Field(..., description="SO2 libre (mg/dm3)")
    total_sulfur_dioxide: float = Field(..., description="SO2 total (mg/dm3)")
    density: float = Field(..., description="Densite (g/cm3)")
    ph: float = Field(..., description="pH")
    sulphates: float = Field(..., description="Sulfates (g/dm3)")
    alcohol: float = Field(..., description="Degre d'alcool (% vol)")


class PredictRequest(BaseModel):
    run_id: str = Field(..., description="Identifiant du run MLflow a utiliser")
    features: WineFeatures


class PredictResponse(BaseModel):
    predicted_quality: float
    run_id: str
    algo: str
    params: dict[str, Any]


# --------------------------------------------------------------------------- #
# Fonctions utilitaires (avec cache)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=16)
def load_model(run_id: str):
    """Charge (et met en cache) un modele MLflow a partir de son run_id."""
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
    return mlflow.pyfunc.load_model(model_uri)


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Charge le CSV du dataset une seule fois."""
    return pd.read_csv(DATA_PATH)


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=TRACKING_URI)


def _run_to_dict(run) -> dict[str, Any]:
    metrics = run.data.metrics or {}
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name or run.data.tags.get("mlflow.runName", ""),
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "params": dict(run.data.params or {}),
        "metrics": {
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
        },
        "tags": {
            k: v for k, v in (run.data.tags or {}).items()
            if not k.startswith("mlflow.")
        },
    }


def _guess_algo(run: dict[str, Any]) -> str:
    """Deduit la famille de modele a partir du nom d'experience ou des params."""
    name = run.get("experiment_name", "") or ""
    lname = name.lower()
    if "ridge" in lname:
        return "Ridge"
    if "lasso" in lname:
        return "Lasso"
    if "el" in lname or "elastic" in lname:
        return "ElasticNet"
    # repli : ElasticNet a un l1_ratio, pas Ridge/Lasso
    if "l1_ratio" in run.get("params", {}):
        return "ElasticNet"
    return "unknown"


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict[str, Any]:
    """Verifie la connexion au serveur MLflow."""
    try:
        experiments = _client().search_experiments()
        return {
            "status": "ok",
            "tracking_uri": TRACKING_URI,
            "experiment_count": len(experiments),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "degraded",
            "tracking_uri": TRACKING_URI,
            "error": str(exc),
        }


@app.get("/experiments")
def list_experiments() -> list[dict[str, Any]]:
    """Liste les experiences MLflow avec leur nombre de runs."""
    client = _client()
    result = []
    for exp in client.search_experiments():
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        result.append(
            {
                "name": exp.name,
                "experiment_id": exp.experiment_id,
                "run_count": len(runs),
            }
        )
    return result


@app.get("/runs")
def list_runs(
    experiment_name: str = Query(..., description="Nom de l'experience MLflow")
) -> list[dict[str, Any]]:
    """Liste les runs d'une experience, tries par RMSE croissant."""
    client = _client()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise HTTPException(
            status_code=404, detail=f"Experience introuvable : {experiment_name}"
        )

    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    result = []
    for run in runs:
        d = _run_to_dict(run)
        d["experiment_name"] = experiment_name
        d["algo"] = _guess_algo(d)
        result.append(d)

    result.sort(
        key=lambda r: (r["metrics"]["rmse"] is None, r["metrics"]["rmse"] or 0.0)
    )
    return result


@app.get("/features")
def features() -> dict[str, Any]:
    """Renvoie les statistiques (min/max/mean/std) de chaque feature."""
    df = load_dataset()
    stats = {}
    for api_key, column in API_KEY_TO_COLUMN.items():
        series = df[column]
        stats[api_key] = {
            "column": column,
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
        }
    return {"features": stats, "order": list(API_KEY_TO_COLUMN.keys())}


@app.get("/presets")
def presets() -> dict[str, Any]:
    """Valeurs medianes des vins pour differents niveaux de qualite (presets UI)."""
    df = load_dataset()
    result = {}
    for quality in sorted(df["quality"].unique()):
        subset = df[df["quality"] == quality]
        medians = {}
        for api_key, column in API_KEY_TO_COLUMN.items():
            medians[api_key] = float(subset[column].median())
        result[str(int(quality))] = {
            "count": int(len(subset)),
            "features": medians,
        }
    return result


@app.get("/model/{run_id}/coefficients")
def coefficients(run_id: str) -> dict[str, Any]:
    """Renvoie les coefficients du modele lineaire (interpretabilite)."""
    try:
        sk_model = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    coef = getattr(sk_model, "coef_", None)
    intercept = getattr(sk_model, "intercept_", None)
    if coef is None:
        raise HTTPException(status_code=400, detail="Modele sans coefficients lineaires")

    coef_flat = list(map(float, coef.ravel()))
    pairs = sorted(
        zip(FEATURE_COLUMNS, coef_flat),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    return {
        "run_id": run_id,
        "intercept": float(intercept.ravel()[0]) if hasattr(intercept, "ravel") else float(intercept),
        "coefficients": [{"feature": f, "coef": c} for f, c in pairs],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predit la qualite d'un vin a partir des features et d'un run MLflow."""
    try:
        model = load_model(request.run_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=404,
            detail=f"Impossible de charger le modele du run {request.run_id} : {exc}",
        ) from exc

    feature_dict = request.features.model_dump()
    row = {column: feature_dict[api_key] for api_key, column in API_KEY_TO_COLUMN.items()}
    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    try:
        prediction = model.predict(frame)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Echec de la prediction : {exc}") from exc

    value = float(prediction[0]) if hasattr(prediction, "__getitem__") else float(prediction)

    # Recupere params + algo pour enrichir la reponse.
    run = _client().get_run(request.run_id)
    run_dict = _run_to_dict(run)
    exp = _client().get_experiment(run.info.experiment_id)
    run_dict["experiment_name"] = exp.name if exp else ""
    algo = _guess_algo(run_dict)

    return PredictResponse(
        predicted_quality=round(value, 4),
        run_id=request.run_id,
        algo=algo,
        params=run_dict["params"],
    )
