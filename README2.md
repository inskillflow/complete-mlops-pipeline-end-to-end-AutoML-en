### Allez à 04-NewProject

## References # 1,  Credits and source articles


- https://github.com/inskillflow/complete-mlops-pipeline-end-to-end-AutoML-en 
- https://docs.google.com/document/d/1kPL1UA4sMRyVEwfjHA6ZMR_KmvjS2er1JQ_KzFFJ7ks/edit?usp=sharing

## References # 2,  Credits and source articles


This project is adapted and modernised from Kenneth Leung's excellent work. The original
design (H2O AutoML + MLflow + FastAPI + Streamlit for insurance cross-sell) and the
Dockerization approach are described in these two articles:

- [End-to-End AutoML Pipeline with H2O AutoML, MLflow, FastAPI, and Streamlit](https://towardsdatascience.com/end-to-end-automl-train-and-serve-with-h2o-mlflow-fastapi-and-streamlit-5d36eedfe606)
- [How to Dockerize Machine Learning Applications Built with H2O, MLflow, FastAPI, and Streamlit](https://towardsdatascience.com/how-to-dockerize-machine-learning-applications-built-with-h2o-mlflow-fastapi-and-streamlit-a56221035eb5/)
- Original repository: [kennethleungty/End-to-End-AutoML-Insurance](https://github.com/kennethleungty/End-to-End-AutoML-Insurance/)

What changed here vs. the original:
- A dedicated **MLflow Tracking Server + Model Registry** (instead of a committed local `mlruns/`).
- Automatic, self-contained training via a one-shot `trainer` service + `depends_on` conditions.
- Modernised dependencies (Python 3.11, MLflow 2.x, recent H2O/pandas/FastAPI/Streamlit).
- Backend loads the model by registry alias `models:/insurance-automl@champion`.
- Richer Streamlit UI: dataset description, plain-language summary, and a confusion matrix.

## References # 3,  Credits and source articles
- https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
- https://mlflow.org/docs/latest/model-registry.html
- https://fastapi.tiangolo.com/
- https://docs.streamlit.io/

