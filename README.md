# Bank Marketing Prediction — ML Pipeline

## Problem Statement
Predicted whether a bank customer will subscribe to a term deposit, using
the UCI Bank Marketing dataset (45,211 customers, 17 features).

## Pipeline Overview
1. **Data Cleaning** — Handled disguised missing values (categorical
   columns using "unknown" instead of NaN). Imputed low-missing columns
   (job: 0.6%, education: 4.1%) with mode; kept high-missing columns
   (contact: 28.8%, poutcome: 81.7%) as their own category since
   "unknown" itself carries meaning (e.g., poutcome=unknown mostly means
   first-time contact).
2. **Feature Engineering** — Converted the pdays placeholder (-1) into a
   clean binary flag. One-hot encoded 6 nominal categorical columns.
   Standardized numeric features (age, balance, duration, etc.).
3. **Class Imbalance** — Target was imbalanced (88% no / 12% yes).
   Used class_weight='balanced' in both models to avoid a lazy
   "always predict no" model, and evaluated using precision/recall/F1
   instead of relying on accuracy alone.
4. **Modeling** — Trained and compared Logistic Regression vs Decision
   Tree (max_depth=6, to avoid overfitting).
5. **Evaluation** — Logistic Regression outperformed on precision (42%
   vs 35%) and F1-score (55% vs 50%), making it the better choice for
   this business case — fewer wasted follow-up calls while still
   catching 81.5% of actual subscribers.

## Results
| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| Accuracy | 84.6% | 80% |
| Precision (yes) | 41.8% | 35% |
| Recall (yes) | 81.5% | 85% |
| F1-score (yes) | 55% | 50% |

## Dashboard
An interactive Streamlit dashboard (`dashboard.py`) lets you input customer
details (age, job, balance, contact history, etc.) and get a live
subscription prediction from the trained Logistic Regression model, along
with the predicted probability.

## Tech Stack
Python, pandas, scikit-learn, Streamlit, Jupyter Notebook

## Limitations & Next Steps
- Dataset size (~45K rows) suits pandas; a larger-scale version could be
  re-implemented in PySpark to demonstrate distributed processing.
- Could try further tuning (GridSearchCV) or ensemble models
  (Random Forest, XGBoost) to improve precision further.

## How to Run

**To explore the full pipeline (cleaning, modeling, evaluation):**
1. `pip install -r requirements.txt`
2. Open `project.ipynb` in VS Code/Jupyter and run all cells

**To use the interactive dashboard:**
1. `pip install -r requirements.txt`
2. `streamlit run dashboard.py`
