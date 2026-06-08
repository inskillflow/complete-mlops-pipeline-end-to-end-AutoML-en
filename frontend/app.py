# =========================================
# Streamlit UI for the End-to-End AutoML project
# Posts an uploaded CSV to the FastAPI backend, displays predictions,
# explains the dataset, and (when ground-truth labels are present)
# shows an evaluation with a confusion matrix.
# Original author: Kenneth Leung
# =========================================
# Run locally: streamlit run app.py
import io
import json
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AutoML Insurance Cross-Sell", page_icon="🚗", layout="wide")
st.title('End-to-End AutoML Project: Insurance Cross-Sell')
st.caption('H2O AutoML + MLflow + FastAPI + Streamlit')

# Backend endpoint (service name on the Docker network, overridable via env)
ENDPOINT = os.getenv('BACKEND_URL', 'http://backend:8000/predict')

TARGET_COL = 'Response'
LABELS = {1: 'Interested in vehicle insurance', 0: 'Not interested'}


# ---------------------------------------------------------------------------
# Dataset description (English)
# ---------------------------------------------------------------------------
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
    """Confusion-matrix counts and standard classification metrics (no sklearn needed)."""
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

    # Convert dataframe to BytesIO object (for parsing as file into FastAPI later)
    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)
    test_bytes_obj.seek(0)  # Reset pointer to avoid EmptyDataError

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

                # Build a readable results table (handles both list and {id: pred} dict)
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

                # --- Plain-language summary ---
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

                # --- Evaluation + confusion matrix (only if ground truth provided) ---
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

                # --- Detailed, readable table ---
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

                # --- Downloads (readable CSV + raw JSON) ---
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
