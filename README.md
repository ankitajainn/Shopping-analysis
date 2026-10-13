# Shopping Analysis | Data Engineering & ML Analytics Pipeline

An end-to-end customer analytics project: a Python ETL pipeline cleans ~3,900 transaction records, a SQL layer engineers features, two scikit-learn models (Ridge Regression and K-Means) extract insight, and a Streamlit app serves both models for real-time use.

**Live app:** [Open the Streamlit app](https://shopping-analysis.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.x-blue) ![SQL](https://img.shields.io/badge/SQL-MySQL%20%7C%20SQLite-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Highlights

- **ETL pipeline (Python):** processes 3,900+ records through missing-value validation, normalization and data-quality auditing.
- **SQL analytics layer:** CTEs, window functions and `CASE` statements across 10+ queries for business analysis and feature engineering.
- **Machine learning:** Ridge Regression (spend prediction) and K-Means (4 customer personas), evaluated with 5-fold cross-validation and Silhouette scoring.
- **Deployment:** Streamlit app that loads serialized Joblib (`.pkl`) pipelines for real-time inference.

---

## Architecture

```
Raw CSV  ->  Python ETL  ->  Cleaned CSV  ->  SQL feature layer  ->  scikit-learn models  ->  Joblib artifacts  ->  Streamlit app
```

| Stage | What happens | Where |
|---|---|---|
| Extract / Transform | Validate, impute, standardize, export | `Project.ipynb` |
| Business analysis | 10 analytical SQL questions | `customer_behavior_sql_queries.sql` |
| Feature engineering + ML | In-memory SQLite features, Ridge, K-Means | `ml_pipeline.py` |
| Serialization | Full pipelines saved with Joblib | `artifacts/` |
| Serving | Interactive predictions and segmentation | `app.py` |

---

## Repository structure

```
Shopping-analysis/
├── artifacts/
│   ├── model.pkl                  # Ridge pipeline (preprocessing + model)
│   ├── kmeans_model.pkl           # Scaler + K-Means pipeline
│   └── model_metadata.pkl         # Feature lists, alpha, k, metrics, sklearn version
├── Project.ipynb                  # Data cleaning / EDA notebook
├── ml_pipeline.py                 # SQL feature layer + Ridge + K-Means + export (Colab-ready)
├── app.py                         # Streamlit application
├── customer_behavior_sql_queries.sql
├── customer_shopping_behavior.csv # Raw data
├── my_data_cleaned_file.csv       # Cleaned data
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 1. Data cleaning (ETL)

Performed in `Project.ipynb`:

1. **Assessed dataset structure:** inspected data types, null counts and summary statistics.
2. **Imputed missing values:** filled 37 null entries in `Review Rating` using category-specific medians.
3. **Standardized schema:** lowercase column names, spaces replaced with underscores.
4. **Renamed target attribute:** `purchase_amount_(usd)` to `purchase_amount`.
5. **Removed redundant variable:** dropped `promo_code_used` after verifying it perfectly matched `discount_applied`.
6. **Binned age:** four quantile-based age groups.
7. **Mapped frequencies to days:** text cadence (Weekly, Fortnightly, ...) to numeric `purchase_frequency_days`.
8. **Exported** the normalized dataset to `my_data_cleaned_file.csv` without an index.

---

## 2. SQL analysis

`customer_behavior_sql_queries.sql` answers:

1. Total revenue by male vs. female customers
2. Customers who used a discount but still spent more than the average purchase amount
3. Top 5 products by average review rating
4. Average purchase amount: Standard vs. Express shipping
5. Do subscribers spend more? (average spend and total revenue)
6. Top 5 products by percentage of purchases with a discount
7. Customer segments (New / Returning / Loyal) by previous purchases
8. Top 3 most purchased products within each category
9. Are repeat buyers (more than 5 previous purchases) more likely to subscribe?
10. Revenue contribution of each age group

### Feature engineering in SQL (inside `ml_pipeline.py`)

The cleaned data is loaded into an **in-memory SQLite** database and features are built with CTEs, window functions and `CASE` statements:

| Feature | Logic |
|---|---|
| `subscription_flag`, `discount_flag` | Yes/No to 1/0 |
| `purchase_frequency_days` | Text cadence to days between purchases |
| `annual_purchase_rate` | `365 / purchase_frequency_days` (purchasing velocity) |
| `loyalty_segment` | New / Returning / Loyal from `previous_purchases` |
| `age_group` | Age bands |
| `rating_vs_category_avg` | Review rating minus category average (window function) |

---

## 3. Machine learning

### Ridge Regression: predicting purchase amount
- Preprocessing (`StandardScaler` for numeric, `OneHotEncoder` for categorical) lives **inside** a scikit-learn `Pipeline`, so it is fit on training data only and reused identically at inference.
- Regularization strength (alpha) chosen with `RidgeCV` using **5-fold cross-validation**.
- Compared against plain Linear Regression and a naive mean-prediction baseline.
- Reported metrics: RMSE, MAE, R² on a held-out 20% test set.



### K-Means: customer personas
- Features: age, purchase amount, review rating, previous purchases, purchase frequency, subscription flag, standardized before clustering.
- `k = 4` selected using the **elbow method** and **Silhouette score**, and for interpretability.
- Silhouette score at k = 4: 



---

## 4. Streamlit app

`app.py` loads `artifacts/model.pkl`, `artifacts/kmeans_model.pkl` and `artifacts/model_metadata.pkl` with Joblib, takes customer inputs from the sidebar, and returns a predicted purchase amount and customer segment.

**Run locally**

```bash
git clone https://github.com/ankitajainn/Shopping-analysis.git
cd Shopping-analysis
pip install -r requirements.txt
streamlit run app.py
```

> The saved `.pkl` files were created with the scikit-learn version recorded in `model_metadata.pkl`. Use the version pinned in `requirements.txt` to avoid pickle incompatibilities.

**Retrain the models**

```bash
python ml_pipeline.py
```

This regenerates everything in `artifacts/`.

---

## Tech stack

Python, pandas, NumPy, scikit-learn, SQLite, MySQL, Joblib, Matplotlib, Streamlit

## Possible improvements

- Compare tree-based models (Random Forest, Gradient Boosting) against the Ridge baseline
- Compute group-level features (e.g. category averages) from training data only to avoid leakage
- Add unit tests for the ETL steps and schedule the pipeline
- Add a churn / customer-risk score from the segmentation

## Author

**Ankita Jain**, B.Tech Electrical Engineering, NIT Bhopal
[GitHub](https://github.com/ankitajainn)

## License

Released under the [MIT License](LICENSE).










