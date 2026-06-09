# Complete Prediction Workflow — Questions Only

## Student Name: _______________________________

## Date: _______________________________

## Course: _______________________________

---

# Question 1 — CSV Upload in Streamlit

Explain what happens when the user uploads a CSV file in the Streamlit interface.

Answer:

---

---

---

---

---

---

# Question 2 — Role of Streamlit

What is the role of the Streamlit frontend in this AutoML application?

Answer:

---

---

---

---

---

---

# Question 3 — File Preview

Why does Streamlit read the uploaded CSV file with pandas before sending it to the backend?

Answer:

---

---

---

---

---

---

# Question 4 — DataFrame Concept

Explain what a pandas DataFrame represents in this workflow.

Answer:

---

---

---

---

---

---

# Question 5 — Start Prediction Button

Explain what happens when the user clicks the **Start Prediction** button.

Answer:

---

---

---

---

---

---

# Question 6 — Preparing the File Before Sending

How does Streamlit prepare the uploaded file before sending it to the FastAPI backend?

Answer:

---

---

---

---

---

---

# Question 7 — Importance of `seek(0)`

Why is `seek(0)` important when Streamlit sends an in-memory CSV file to the backend?

Answer:

---

---

---

---

---

---

# Question 8 — HTTP POST Request

Explain the role of the HTTP POST request in the prediction workflow.

Answer:

---

---

---

---

---

---

# Question 9 — Backend URL

Why does the frontend use the internal Docker URL `http://backend:8000/predict`?

Answer:

---

---

---

---

---

---

# Question 10 — `localhost` Problem

Why would `http://localhost:8000/predict` be incorrect from inside the frontend container?

Answer:

---

---

---

---

---

---

# Question 11 — Docker Service Names

Explain how Docker Compose service names allow containers to communicate with each other.

Answer:

---

---

---

---

---

---

# Question 12 — FastAPI Backend

What is the role of the FastAPI backend in this application?

Answer:

---

---

---

---

---

---

# Question 13 — `/predict` Endpoint

Explain what happens inside the backend when the `/predict` endpoint receives the uploaded CSV file.

Answer:

---

---

---

---

---

---

# Question 14 — Bytes to DataFrame

Why does the backend convert the uploaded file from bytes into a pandas DataFrame?

Answer:

---

---

---

---

---

---

# Question 15 — H2OFrame Conversion

Why does the backend convert the pandas DataFrame into an H2OFrame before prediction?

Answer:

---

---

---

---

---

---

# Question 16 — ID Column

Why should an ID column be separated before sending the data to the machine learning model?

Answer:

---

---

---

---

---

---

# Question 17 — Column Type Matching

Why is it important to match the prediction dataset column types with the training dataset column types?

Answer:

---

---

---

---

---

---

# Question 18 — MLflow Model Registry

Explain the role of the MLflow Model Registry in this application.

Answer:

---

---

---

---

---

---

# Question 19 — Model URI

Explain the meaning of the following model URI:

`models:/insurance-automl@champion`

Answer:

---

---

---

---

---

---

# Question 20 — Champion Alias

Why is the alias `champion` useful in the MLflow model loading process?

Answer:

---

---

---

---

---

---

# Question 21 — Environment Variables

Explain the role of environment variables such as `MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_ALIAS`, and `BACKEND_URL`.

Answer:

---

---

---

---

---

---

# Question 22 — Prediction Execution

Explain how the backend uses the trained H2O AutoML model to generate predictions.

Answer:

---

---

---

---

---

---

# Question 23 — JSON Response

Why does the backend return the prediction results as JSON?

Answer:

---

---

---

---

---

---

# Question 24 — Displaying Results in Streamlit

Explain how Streamlit displays the prediction results after receiving the backend response.

Answer:

---

---

---

---

---

---

# Question 25 — Optional Evaluation with `Response`

If the uploaded CSV file contains a `Response` column, what additional analysis can Streamlit perform?

Answer:

---

---

---

---

---
