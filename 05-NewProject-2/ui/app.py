"""
Interface Streamlit pedagogique pour le projet Wine Quality MLOps.

Cette application consomme UNIQUEMENT l'API FastAPI (jamais MLflow directement).
Elle est organisee en 5 onglets :
    1. Accueil          - vue d'ensemble et sante du systeme
    2. Exploration      - analyse exploratoire du dataset (EDA)
    3. Theorie          - explication des modeles Ridge / Lasso / ElasticNet
    4. Comparaison      - comparaison des runs MLflow
    5. Prediction       - prediction interactive de la qualite d'un vin
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

import pages_content as content

# --------------------------------------------------------------------------- #
# Configuration generale
# --------------------------------------------------------------------------- #

API_URL_DEFAULT = os.getenv("API_URL", "http://api:8000")
DATA_PATH = os.getenv("DATA_PATH", "data/red-wine-quality.csv")
TARGET = "quality"

st.set_page_config(
    page_title="Wine Quality MLOps",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Clients API (avec cache)
# --------------------------------------------------------------------------- #


def api_url() -> str:
    return st.session_state.get("api_url", API_URL_DEFAULT).rstrip("/")


def api_get(path: str, **params):
    resp = requests.get(f"{api_url()}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, payload: dict):
    resp = requests.post(f"{api_url()}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False, ttl=60)
def get_health():
    return api_get("/health")


@st.cache_data(show_spinner=False, ttl=60)
def get_experiments():
    return api_get("/experiments")


@st.cache_data(show_spinner=False, ttl=60)
def get_runs(experiment_name: str):
    return api_get("/runs", experiment_name=experiment_name)


@st.cache_data(show_spinner=False, ttl=300)
def get_features():
    return api_get("/features")


@st.cache_data(show_spinner=False, ttl=300)
def get_presets():
    return api_get("/presets")


@st.cache_data(show_spinner=False, ttl=300)
def get_coefficients(run_id: str):
    return api_get(f"/model/{run_id}/coefficients")


@st.cache_data(show_spinner=False)
def load_local_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def all_runs() -> pd.DataFrame:
    """Aggrege tous les runs de toutes les experiences dans un DataFrame."""
    rows = []
    for exp in get_experiments():
        if exp["run_count"] == 0:
            continue
        for run in get_runs(exp["name"]):
            rows.append(
                {
                    "experiment": exp["name"],
                    "algo": run.get("algo", ""),
                    "run_name": run.get("run_name", ""),
                    "run_id": run["run_id"],
                    "alpha": _to_float(run["params"].get("alpha")),
                    "l1_ratio": _to_float(run["params"].get("l1_ratio")),
                    "rmse": run["metrics"].get("rmse"),
                    "mae": run["metrics"].get("mae"),
                    "r2": run["metrics"].get("r2"),
                }
            )
    return pd.DataFrame(rows)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Barre laterale
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.title("🍇 Wine MLOps")
    st.caption("Educational interface — MLflow + FastAPI + Streamlit")

    st.session_state.setdefault("api_url", API_URL_DEFAULT)
    st.session_state["api_url"] = st.text_input(
        "FastAPI URL", value=st.session_state["api_url"]
    )

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    try:
        health = get_health()
        if health.get("status") == "ok":
            st.success(f"API + MLflow OK\n\n{health.get('experiment_count', 0)} experiment(s)")
        else:
            st.warning(f"API reachable, MLflow degraded: {health.get('error', '?')}")
    except requests.RequestException:
        st.error("API unreachable. Make sure the `api` service is running.")

    st.divider()
    st.caption(
        "Recommended order:\n\n"
        "1. `docker compose up -d --build mlflow`\n"
        "2. `docker compose run --rm trainer`\n"
        "3. `docker compose up -d --build api ui`"
    )


# --------------------------------------------------------------------------- #
# Onglets
# --------------------------------------------------------------------------- #

tab_home, tab_eda, tab_theory, tab_compare, tab_predict = st.tabs(
    [
        "🏠 Home",
        "📊 Data exploration",
        "📚 Theory",
        "🏆 MLflow comparison",
        "🔮 Prediction",
    ]
)


# ------------------------------ Tab 1: Home -------------------------------- #
with tab_home:
    st.title("Predict red wine quality 🍷")
    st.markdown(content.INTRO)

    st.subheader("The MLOps flow")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fontname="Helvetica"];
            CSV [label="Data\\n(CSV)"];
            Trainer [label="Trainer\\n(scikit-learn)"];
            MLflow [label="MLflow\\n(tracking)"];
            API [label="FastAPI\\n(serving)"];
            UI [label="Streamlit\\n(this app)"];
            CSV -> Trainer -> MLflow -> API -> UI;
        }
        """,
        use_container_width=True,
    )

    st.subheader("System status")
    try:
        runs_df = all_runs()
        c1, c2, c3 = st.columns(3)
        c1.metric("Experiments", len(get_experiments()))
        c2.metric("Recorded runs", len(runs_df))
        best_rmse = runs_df["rmse"].min() if not runs_df.empty else None
        c3.metric("Best RMSE", f"{best_rmse:.4f}" if best_rmse is not None else "—")

        if runs_df.empty:
            st.info(
                "No run found. Start training with "
                "`docker compose run --rm trainer` then refresh."
            )
    except requests.RequestException:
        st.error("Cannot reach the API. Check the `api` service.")


# --------------------------- Tab 2: Data exploration ----------------------- #
with tab_eda:
    st.title("📊 Data exploration")
    try:
        df = load_local_dataset()
    except FileNotFoundError:
        st.error(f"File not found: {DATA_PATH}")
        df = None

    if df is not None:
        st.markdown(
            f"The dataset contains **{len(df)} wines** and "
            f"**{df.shape[1]} columns** ({df.shape[1] - 1} features + the `quality` target)."
        )

        with st.expander("Data preview (head)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        with st.expander("Descriptive statistics (describe)"):
            st.dataframe(df.describe().T, use_container_width=True)

        st.subheader("Distribution of a variable")
        col = st.selectbox("Choose a variable", df.columns, index=len(df.columns) - 1)
        fig_hist = px.histogram(df, x=col, nbins=30, marginal="box", title=f"Distribution of {col}")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Correlation matrix")
        corr = df.corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation between variables",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
        top3 = target_corr.head(3)
        st.markdown(
            "**Variables most correlated with `quality`**: "
            + ", ".join(f"`{k}` ({v:+.2f})" for k, v in top3.items())
        )

        st.subheader("Boxplots by quality level")
        default_feats = [c for c in ["alcohol", "volatile acidity", "sulphates"] if c in df.columns]
        feats = st.multiselect(
            "Variables to compare", [c for c in df.columns if c != TARGET], default=default_feats
        )
        for feat in feats:
            fig_box = px.box(df, x=TARGET, y=feat, title=f"{feat} by quality", points="outliers")
            st.plotly_chart(fig_box, use_container_width=True)

        st.info(content.EDA_HELP)


# ----------------------------- Tab 3: Theory ------------------------------- #
with tab_theory:
    st.title("📚 Understanding the models")
    st.markdown(content.THEORY_INTRO)

    with st.expander("Ridge (L2 penalty)", expanded=True):
        st.markdown(content.RIDGE_MD)
    with st.expander("Lasso (L1 penalty)"):
        st.markdown(content.LASSO_MD)
    with st.expander("ElasticNet (L1 + L2)"):
        st.markdown(content.ELASTICNET_MD)

    st.subheader("Bias-variance trade-off vs alpha")
    st.caption(
        "Illustration: as `alpha` increases, the model gets simpler. "
        "The bias error goes up, the variance error goes down. "
        "The total error has a minimum: the right `alpha`."
    )
    alpha = st.slider("alpha (regularization strength)", 0.0, 2.0, 0.5, 0.05)
    grid = np.linspace(0.01, 2.0, 100)
    bias = 0.15 + 0.35 * grid              # bias grows with alpha
    variance = 0.6 / (1.0 + 4.0 * grid)     # variance shrinks with alpha
    total = bias + variance
    curve = pd.DataFrame(
        {"alpha": grid, "Bias^2": bias, "Variance": variance, "Total error": total}
    )
    fig_bv = px.line(
        curve,
        x="alpha",
        y=["Bias^2", "Variance", "Total error"],
        title="Illustrative decomposition of the error",
    )
    fig_bv.add_vline(x=alpha, line_dash="dash", annotation_text=f"alpha = {alpha:.2f}")
    st.plotly_chart(fig_bv, use_container_width=True)

    st.info(content.CHOICE_GUIDE)


# -------------------------- Tab 4: MLflow comparison ----------------------- #
with tab_compare:
    st.title("🏆 MLflow experiment comparison")
    try:
        runs_df = all_runs()
    except requests.RequestException:
        st.error("Cannot reach the API.")
        runs_df = pd.DataFrame()

    if runs_df.empty:
        st.info(
            "No run to compare. Run `docker compose run --rm trainer` "
            "then refresh the data."
        )
    else:
        display = runs_df[
            ["algo", "run_name", "alpha", "l1_ratio", "rmse", "mae", "r2"]
        ].sort_values("rmse")
        st.subheader("All runs")
        st.dataframe(
            display.style.format(
                {"alpha": "{:.2f}", "l1_ratio": "{:.2f}", "rmse": "{:.4f}", "mae": "{:.4f}", "r2": "{:.4f}"}
            ),
            use_container_width=True,
        )

        champion = runs_df.sort_values("rmse").iloc[0]
        st.session_state["champion_run_id"] = champion["run_id"]
        st.success(
            f"🏆 Global champion: **{champion['algo']}** "
            f"(`{champion['run_name']}`, alpha={champion['alpha']}) "
            f"— RMSE = {champion['rmse']:.4f}"
        )

        st.subheader("RMSE by algorithm and alpha")
        fig_bar = px.bar(
            runs_df,
            x="algo",
            y="rmse",
            color="alpha",
            barmode="group",
            title="RMSE (lower = better)",
            hover_data=["run_name", "mae", "r2"],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Radar of champions per family")
        champions = runs_df.loc[runs_df.groupby("algo")["rmse"].idxmin()]
        metrics = ["rmse", "mae", "r2"]
        norm = champions.copy()
        for m in ["rmse", "mae"]:
            rng = runs_df[m].max() - runs_df[m].min()
            norm[m] = 1 - (champions[m] - runs_df[m].min()) / rng if rng else 1.0
        rng_r2 = runs_df["r2"].max() - runs_df["r2"].min()
        norm["r2"] = (champions["r2"] - runs_df["r2"].min()) / rng_r2 if rng_r2 else 1.0

        fig_radar = go.Figure()
        labels = ["RMSE (norm.)", "MAE (norm.)", "R2 (norm.)"]
        for _, row in norm.iterrows():
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[row["rmse"], row["mae"], row["r2"]],
                    theta=labels,
                    fill="toself",
                    name=row["algo"],
                )
            )
        fig_radar.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            title="Normalized champions (higher = better)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.info(content.METRICS_HELP)


# --------------------------- Tab 5: Prediction ----------------------------- #
with tab_predict:
    st.title("🔮 Interactive prediction")
    st.markdown(content.PREDICT_HELP)

    try:
        feats_info = get_features()
        runs_df = all_runs()
    except requests.RequestException:
        st.error("Cannot reach the API.")
        feats_info, runs_df = None, pd.DataFrame()

    if feats_info is None or runs_df.empty:
        st.info(
            "Prediction unavailable: make sure the API responds and the "
            "trainer has created runs."
        )
    else:
        features = feats_info["features"]
        order = feats_info["order"]

        # --- Presets ---
        st.subheader("Presets")
        try:
            presets = get_presets()
        except requests.RequestException:
            presets = {}

        preset_cols = st.columns(len(presets) + 1) if presets else [st]
        if presets:
            for idx, (quality, info) in enumerate(sorted(presets.items())):
                label = f"Quality {quality} ({info['count']} wines)"
                if preset_cols[idx].button(label, use_container_width=True):
                    for key, val in info["features"].items():
                        st.session_state[f"feat_{key}"] = float(val)
                    st.rerun()
            if preset_cols[-1].button("Reset (mean)", use_container_width=True):
                for key in order:
                    st.session_state[f"feat_{key}"] = float(features[key]["mean"])
                st.rerun()

        # --- Sliders ---
        st.subheader("Wine characteristics")
        values = {}
        slider_cols = st.columns(3)
        for i, key in enumerate(order):
            info = features[key]
            default = float(st.session_state.get(f"feat_{key}", info["mean"]))
            lo, hi = float(info["min"]), float(info["max"])
            default = min(max(default, lo), hi)
            step = max((hi - lo) / 100.0, 1e-4)
            values[key] = slider_cols[i % 3].slider(
                info["column"],
                min_value=lo,
                max_value=hi,
                value=default,
                step=step,
                key=f"feat_{key}",
            )

        # --- Model choice ---
        st.subheader("Model to use")
        runs_df = runs_df.sort_values("rmse").reset_index(drop=True)
        options = runs_df["run_id"].tolist()

        def _fmt(rid):
            r = runs_df[runs_df["run_id"] == rid].iloc[0]
            return f"{r['algo']} | {r['run_name']} | alpha={r['alpha']} | RMSE={r['rmse']:.4f}"

        champ = st.session_state.get("champion_run_id", options[0])
        default_index = options.index(champ) if champ in options else 0
        run_id = st.selectbox("MLflow run", options, index=default_index, format_func=_fmt)

        if st.button("🔮 Predict quality", type="primary", use_container_width=True):
            try:
                result = api_post("/predict", {"run_id": run_id, "features": values})
            except requests.RequestException as exc:
                st.error(f"Prediction failed: {exc}")
                result = None

            if result is not None:
                quality = result["predicted_quality"]
                c1, c2 = st.columns([1, 2])
                c1.metric("Predicted quality", f"{quality:.2f}")

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=quality,
                        title={"text": f"{result['algo']}"},
                        gauge={
                            "axis": {"range": [3, 8]},
                            "bar": {"color": "darkred"},
                            "steps": [
                                {"range": [3, 5], "color": "#f2c9c0"},
                                {"range": [5, 6.5], "color": "#f7e2b5"},
                                {"range": [6.5, 8], "color": "#bfe3b5"},
                            ],
                        },
                    )
                )
                gauge.update_layout(height=280, margin={"t": 40, "b": 10})
                c2.plotly_chart(gauge, use_container_width=True)

                with st.expander("API response details"):
                    st.json(result)

                st.subheader("Why this value? (model coefficients)")
                try:
                    coefs = get_coefficients(run_id)
                    coef_df = pd.DataFrame(coefs["coefficients"])
                    fig_coef = px.bar(
                        coef_df,
                        x="coef",
                        y="feature",
                        orientation="h",
                        title="Weight of each variable (coefficients)",
                    )
                    fig_coef.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_coef, use_container_width=True)
                    st.caption(
                        "A positive coefficient pushes quality up, a negative one "
                        "pushes it down. Zero coefficients (Lasso) mark variables "
                        "ignored by the model."
                    )
                except requests.RequestException:
                    st.caption("Coefficients not available for this model.")
