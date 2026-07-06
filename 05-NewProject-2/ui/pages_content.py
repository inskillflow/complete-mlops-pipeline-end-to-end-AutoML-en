"""
Long educational texts for the Streamlit interface.
Kept here to keep app.py readable.
"""

INTRO = """
### Welcome to the red wine MLOps lab

This application illustrates a **complete, end-to-end MLOps cycle**:

1. **Data**: a dataset of Portuguese red wines (Vinho Verde), with 11
   physico-chemical measurements and a quality score from 3 to 8.
2. **Training** (`trainer`): three families of regularized linear regression
   models (ElasticNet, Ridge, Lasso) are trained with several values of the
   `alpha` parameter.
3. **Tracking** (`MLflow`): each training run is recorded with its parameters,
   metrics and model.
4. **Serving** (`FastAPI`): an API loads the registered models and answers
   prediction requests.
5. **Interface** (`Streamlit`): this application, which only talks to the API.
   It never touches MLflow directly.

Go through the tabs from left to right: you follow the same path as a data
point, from exploration all the way to a production prediction.
"""

EDA_HELP = """
**How to read these charts?**

- The **histogram** shows the distribution of a variable. A narrow distribution
  means most wines look alike on that criterion.
- The **correlation matrix** measures the linear link between two variables,
  from -1 (opposite) to +1 (identical). A value close to 0 means no linear
  relationship.
- The variables most correlated with `quality` are the best candidates to
  predict quality. In this dataset, `alcohol` is usually the most strongly
  (positively) correlated, and `volatile acidity` negatively.
"""

THEORY_INTRO = """
The three models used are **regularized linear regressions**. A linear
regression looks for coefficients (beta) such that:

$$\\hat{y} = \\beta_0 + \\beta_1 x_1 + \\dots + \\beta_{11} x_{11}$$

**Regularization** adds a penalty on the size of the coefficients to avoid
overfitting. The `alpha` parameter controls the strength of that penalty: the
larger `alpha`, the more the coefficients are constrained (a simpler model,
more bias, less variance).
"""

RIDGE_MD = """
#### Ridge - L2 penalty

$$\\min_{\\beta}\\ \\lVert y - X\\beta \\rVert_2^2 \\ +\\ \\alpha \\lVert \\beta \\rVert_2^2$$

- Penalizes the **sum of squares** of the coefficients.
- **Shrinks** all coefficients toward zero, but **never exactly to zero**.
- Great when variables are **correlated with each other** (multicollinearity).
- Does not perform variable selection: it keeps all features.
"""

LASSO_MD = """
#### Lasso - L1 penalty

$$\\min_{\\beta}\\ \\lVert y - X\\beta \\rVert_2^2 \\ +\\ \\alpha \\lVert \\beta \\rVert_1$$

- Penalizes the **sum of absolute values** of the coefficients.
- Can set some coefficients **exactly to zero**: it therefore performs
  **automatic variable selection**.
- Useful when you suspect only a few variables really matter.
"""

ELASTICNET_MD = """
#### ElasticNet - L1 + L2 combined

$$\\min_{\\beta}\\ \\lVert y - X\\beta \\rVert_2^2 \\ +\\ \\alpha \\big( \\rho \\lVert \\beta \\rVert_1 + (1-\\rho)\\lVert \\beta \\rVert_2^2 \\big)$$

- Mixes Ridge and Lasso. The `l1_ratio` parameter corresponds to `rho`.
- `l1_ratio = 0` -> pure Ridge. `l1_ratio = 1` -> pure Lasso.
- A good compromise: variable selection **and** stability under correlation.
"""

CHOICE_GUIDE = """
**Which one to choose?**

- Many correlated variables, you want to keep them all -> **Ridge**.
- You want a simple model that keeps only the important variables -> **Lasso**.
- You are not sure, you want a robust compromise -> **ElasticNet**.
"""

METRICS_HELP = """
**How to read these metrics?**

- **RMSE** (Root Mean Squared Error): average error expressed in the unit of the
  target (here, quality points). **The smaller, the better.**
- **MAE** (Mean Absolute Error): mean absolute error, less sensitive to extreme
  values than RMSE. **The smaller, the better.**
- **R2** (coefficient of determination): share of variance explained by the
  model, from 0 to 1 (can be negative if the model is very poor).
  **The closer to 1, the better.**

The **champion** is the run with the smallest RMSE.
"""

PREDICT_HELP = """
Move the sliders to describe a wine, pick a model (run) then run the
prediction. The slider ranges come from the real values of the dataset, via the
API. The **presets** fill the sliders with the median profile of a wine of a
given quality.
"""
