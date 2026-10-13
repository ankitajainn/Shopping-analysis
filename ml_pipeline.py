# ==========================================================================================
# CUSTOMER SHOPPING TRENDS — SQL → SCIKIT-LEARN PIPELINE
# Regression (spend prediction) + K-Means (customer segmentation) + joblib export
#
# Designed to run top-to-bottom in Google Colab. Each "# %% [markdown]" block can be
# pasted into a text cell and each "# %%" block into a code cell, or you can simply run
# the whole file as a script (`!python shopping_analysis_ml_colab.py`).
#
# Repo: https://github.com/ankitajainn/Shopping-analysis
# ==========================================================================================


# %% [markdown]
# # 0. Environment setup
# Colab already ships with pandas / scikit-learn / matplotlib / joblib. `sqlite3` is in the
# Python standard library, so there is nothing to install for the SQL layer.

# %%
# Uncomment if a package is ever missing in your runtime:
# !pip install -q pandas scikit-learn matplotlib seaborn joblib

import os
import sqlite3
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)
RANDOM_STATE = 42

print("Libraries loaded.")


# %% [markdown]
# # 1. Data loading
# Pulls the CSV straight from the GitHub **raw** endpoint so the notebook is reproducible
# with no manual upload. Two candidates are tried in order:
#
# 1. `my_data_cleaned_file.csv` — the cleaned export from the data-cleaning notebook
#    (snake_case columns, imputed review ratings, `purchase_frequency_days` already mapped).
# 2. `customer_shopping_behavior.csv` — the original raw file.
#
# A local-file fallback is included for offline runs. Column names are normalised
# immediately afterwards so the rest of the script works with either file.

# %%
GITHUB_USER = "ankitajainn"
GITHUB_REPO = "Shopping-analysis"
GITHUB_BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}"

CANDIDATE_FILES = [
    f"{RAW_BASE}/my_data_cleaned_file.csv",
    f"{RAW_BASE}/customer_shopping_behavior.csv",
    # Kaggle mirror of the same dataset, if you prefer:
    # "https://raw.githubusercontent.com/<you>/<repo>/main/shopping_trends.csv",
    "my_data_cleaned_file.csv",              # local fallback (e.g. after files.upload())
    "customer_shopping_behavior.csv",
]


def load_dataset(candidates):
    """Return the first CSV that loads successfully, plus the source used."""
    last_error = None
    for src in candidates:
        try:
            frame = pd.read_csv(src)
            print(f"Loaded from: {src}")
            return frame, src
        except Exception as exc:          # noqa: BLE001 - we genuinely want to try the next one
            last_error = exc
            print(f"  could not load {src} ({type(exc).__name__})")
    raise RuntimeError(f"No dataset could be loaded. Last error: {last_error}")


df_raw, source_used = load_dataset(CANDIDATE_FILES)
print(f"\nShape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
df_raw.head()


# %% [markdown]
# ## 1a. Normalise the schema
# Makes the script tolerant of both the raw file (`Purchase Amount (USD)`) and the cleaned
# file (`purchase_amount`). Everything downstream uses snake_case names.

# %%
def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = (
        frame.columns.str.strip()
        .str.lower()
        .str.replace(r"[()\[\]]", "", regex=True)
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    renames = {
        "purchase_amount_usd": "purchase_amount",
        "purchase_amount_in_usd": "purchase_amount",
        "item_purchased": "item_purchased",
        "review_rating": "review_rating",
        "subscription_status": "subscription_status",
        "discount_applied": "discount_applied",
        "promo_code_used": "promo_code_used",
        "previous_purchases": "previous_purchases",
        "frequency_of_purchases": "frequency_of_purchases",
        "shipping_type": "shipping_type",
        "payment_method": "payment_method",
        "customer_id": "customer_id",
    }
    return frame.rename(columns={k: v for k, v in renames.items() if k in frame.columns})


df = normalise_columns(df_raw)
print("Columns:", list(df.columns))
print("\nNulls per column:\n", df.isna().sum()[lambda s: s > 0] or "none")
print("\nDtypes:\n", df.dtypes)


# %% [markdown]
# # 2. SQL prep layer (`sqlite3`)
# The DataFrame is pushed into an **in-memory SQLite database** and all filtering, cleaning
# and feature engineering is expressed as SQL — mirroring how this would work against a real
# warehouse. Scikit-Learn only ever sees the query result.
#
# Features engineered in SQL:
# * `subscription_flag`, `discount_flag` — Yes/No → 1/0
# * `purchase_frequency_days` — text cadence → numeric days between purchases
# * `annual_purchase_rate` — 365 / cadence, i.e. expected purchases per year
# * `loyalty_segment` — New / Returning / Loyal buckets from `previous_purchases`
# * `age_group` — quantile-style age bands
# * `rating_vs_category_avg` — review rating centred on the category mean (window function)

# %%
conn = sqlite3.connect(":memory:")
df.to_sql("shopping", conn, index=False, if_exists="replace")

FEATURE_QUERY = """
WITH base AS (
    SELECT
        CAST(customer_id AS INTEGER)                          AS customer_id,
        CAST(age AS INTEGER)                                  AS age,
        TRIM(gender)                                          AS gender,
        TRIM(category)                                        AS category,
        TRIM(item_purchased)                                  AS item_purchased,
        TRIM(location)                                        AS location,
        TRIM(season)                                          AS season,
        TRIM(shipping_type)                                   AS shipping_type,
        TRIM(payment_method)                                  AS payment_method,
        CAST(purchase_amount AS REAL)                         AS purchase_amount,
        CAST(review_rating AS REAL)                           AS review_rating,
        CAST(previous_purchases AS INTEGER)                   AS previous_purchases,
        TRIM(frequency_of_purchases)                          AS frequency_of_purchases,

        -- Yes/No (or already-boolean) flags -> 0/1
        CASE WHEN LOWER(TRIM(CAST(subscription_status AS TEXT))) IN ('yes','true','1')
             THEN 1 ELSE 0 END                                AS subscription_flag,
        CASE WHEN LOWER(TRIM(CAST(discount_applied AS TEXT)))   IN ('yes','true','1')
             THEN 1 ELSE 0 END                                AS discount_flag,

        -- Purchase cadence -> days between purchases
        CASE LOWER(TRIM(frequency_of_purchases))
            WHEN 'weekly'          THEN 7
            WHEN 'bi-weekly'       THEN 14
            WHEN 'fortnightly'     THEN 14
            WHEN 'every 3 months'  THEN 90
            WHEN 'quarterly'       THEN 90
            WHEN 'monthly'         THEN 30
            WHEN 'annually'        THEN 365
            ELSE NULL
        END                                                   AS purchase_frequency_days
    FROM shopping
    WHERE purchase_amount IS NOT NULL
      AND CAST(purchase_amount AS REAL) > 0
      AND age IS NOT NULL
),
enriched AS (
    SELECT
        b.*,
        ROUND(365.0 / b.purchase_frequency_days, 3)           AS annual_purchase_rate,
        CASE
            WHEN b.previous_purchases <= 10 THEN 'New'
            WHEN b.previous_purchases <= 30 THEN 'Returning'
            ELSE 'Loyal'
        END                                                   AS loyalty_segment,
        CASE
            WHEN b.age < 30 THEN '18-29'
            WHEN b.age < 45 THEN '30-44'
            WHEN b.age < 60 THEN '45-59'
            ELSE '60+'
        END                                                   AS age_group,
        ROUND(
            b.review_rating - AVG(b.review_rating) OVER (PARTITION BY b.category),
            3
        )                                                     AS rating_vs_category_avg
    FROM base b
)
SELECT *
FROM enriched
WHERE purchase_frequency_days IS NOT NULL
  AND review_rating IS NOT NULL;
"""

data = pd.read_sql_query(FEATURE_QUERY, conn)
print(f"Rows after SQL prep: {len(data)}")
data.head()


# %% [markdown]
# ## 2a. A couple of SQL sanity checks
# Quick aggregate queries so you can eyeball the engineered columns before modelling.

# %%
data.to_sql("features", conn, index=False, if_exists="replace")

print("Average spend by subscription status:")
print(pd.read_sql_query("""
    SELECT CASE subscription_flag WHEN 1 THEN 'Subscriber' ELSE 'Non-subscriber' END AS status,
           COUNT(*)                        AS customers,
           ROUND(AVG(purchase_amount), 2)  AS avg_spend,
           ROUND(SUM(purchase_amount), 2)  AS total_revenue
    FROM features
    GROUP BY subscription_flag
    ORDER BY avg_spend DESC;
""", conn).to_string(index=False))

print("\nRevenue contribution by age group:")
print(pd.read_sql_query("""
    SELECT age_group,
           COUNT(*)                       AS customers,
           ROUND(SUM(purchase_amount), 2) AS revenue,
           ROUND(100.0 * SUM(purchase_amount) /
                 (SELECT SUM(purchase_amount) FROM features), 2) AS pct_of_revenue
    FROM features
    GROUP BY age_group
    ORDER BY revenue DESC;
""", conn).to_string(index=False))

print("\nAverage spend by loyalty segment:")
print(pd.read_sql_query("""
    SELECT loyalty_segment,
           COUNT(*)                       AS customers,
           ROUND(AVG(purchase_amount), 2) AS avg_spend
    FROM features
    GROUP BY loyalty_segment
    ORDER BY avg_spend DESC;
""", conn).to_string(index=False))


# %% [markdown]
# # 3. Supervised model — predicting `purchase_amount`
# Target: `purchase_amount` (USD).
# Predictors: age, review rating, previous purchases, subscription flag, discount flag,
# purchase cadence, annual purchase rate, rating-vs-category, plus categorical context
# (gender, category, season, shipping type, payment method).
#
# Everything lives inside one `Pipeline` so that scaling and one-hot encoding are fitted on
# training folds only, and so the exported artefact accepts raw feature rows at inference.

# %%
TARGET = "purchase_amount"

NUMERIC_FEATURES = [
    "age",
    "review_rating",
    "previous_purchases",
    "subscription_flag",
    "discount_flag",
    "purchase_frequency_days",
    "annual_purchase_rate",
    "rating_vs_category_avg",
]
CATEGORICAL_FEATURES = [
    "gender",
    "category",
    "season",
    "shipping_type",
    "payment_method",
    "loyalty_segment",
]

# Keep only features that actually exist in this dataset version.
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in data.columns]
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c in data.columns]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = data[FEATURES].copy()
y = data[TARGET].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

try:                                   # scikit-learn >= 1.2
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
except TypeError:                      # older versions
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", ohe, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)


# %% [markdown]
# ## 3a. Baseline: Multiple Linear Regression (OLS)

# %%
ols_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", LinearRegression()),
])
ols_pipeline.fit(X_train, y_train)


# %% [markdown]
# ## 3b. Ridge Regression with cross-validated alpha
# Ridge shrinks coefficients, which matters here because one-hot encoding produces many
# correlated dummy columns.

# %%
alphas = np.logspace(-3, 3, 25)
ridge_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RidgeCV(alphas=alphas, cv=5)),
])
ridge_pipeline.fit(X_train, y_train)

best_alpha = ridge_pipeline.named_steps["model"].alpha_
print(f"Best Ridge alpha (5-fold CV): {best_alpha:.4f}")

# Refit a plain Ridge at that alpha — cleaner object to persist.
final_model = Pipeline([
    ("prep", preprocessor),
    ("model", Ridge(alpha=best_alpha, random_state=RANDOM_STATE)),
])
final_model.fit(X_train, y_train)


# %% [markdown]
# # 4. Evaluation & interpretation
# RMSE, MAE and R² on the held-out test set, plus 5-fold CV R² on the training set.

# %%
def evaluate(name, pipeline, X_te, y_te):
    preds = pipeline.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    mae = float(mean_absolute_error(y_te, preds))
    r2 = float(r2_score(y_te, preds))
    print(f"{name:<28} RMSE={rmse:8.3f}   MAE={mae:8.3f}   R²={r2:8.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


print("Hold-out test performance")
print("-" * 70)
results = [
    evaluate("Linear Regression (OLS)", ols_pipeline, X_test, y_test),
    evaluate(f"Ridge (alpha={best_alpha:.3g})", final_model, X_test, y_test),
]

# Naive benchmark: always predict the training mean.
baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
print(f"{'Baseline (train mean)':<28} "
      f"RMSE={np.sqrt(mean_squared_error(y_test, baseline_pred)):8.3f}   "
      f"MAE={mean_absolute_error(y_test, baseline_pred):8.3f}   "
      f"R²={r2_score(y_test, baseline_pred):8.4f}")

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_r2 = cross_val_score(final_model, X_train, y_train, cv=cv, scoring="r2")
print(f"\n5-fold CV R² on training data: mean={cv_r2.mean():.4f}  std={cv_r2.std():.4f}")

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))


# %% [markdown]
# ## 4a. Feature coefficients — what actually drives spend?
# Numeric features were standardised, so their coefficients read as **"dollars of change in
# purchase amount per one standard deviation of the feature"** and are directly comparable.
# Categorical coefficients are dollars relative to the dropped reference level.

# %%
feature_names = final_model.named_steps["prep"].get_feature_names_out()
coefs = final_model.named_steps["model"].coef_
intercept = final_model.named_steps["model"].intercept_

coef_df = (
    pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    .assign(abs_coefficient=lambda d: d["coefficient"].abs())
    .sort_values("abs_coefficient", ascending=False)
    .reset_index(drop=True)
)

print(f"Intercept: {intercept:.3f} USD\n")
print("Top 15 drivers by absolute effect:")
print(coef_df.head(15).to_string(index=False))

top_n = coef_df.head(15).sort_values("coefficient")
plt.figure(figsize=(9, 6))
plt.barh(top_n["feature"], top_n["coefficient"],
         color=np.where(top_n["coefficient"] >= 0, "#2a9d8f", "#e76f51"))
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Ridge coefficients — effect on purchase amount (USD)")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### How to read the numbers
# Interpret the printed table above, not a pre-written story. Practical guidance:
#
# * **Compare R² to the naive baseline.** This dataset is a well-known synthetic Kaggle set
#   in which `purchase_amount` is close to uniformly random, so R² near zero (or slightly
#   negative) is a real finding, not a bug: the recorded attributes do not explain spend.
#   Report that honestly rather than over-claiming.
# * **RMSE vs MAE.** RMSE punishes large misses; a gap much wider than MAE means a few
#   customers are badly mispredicted.
# * **Coefficient sign and size.** A positive `num__previous_purchases` coefficient of, say,
#   1.8 means one SD more purchase history is associated with ~$1.80 more spend — an
#   association within this sample, not a causal effect of loyalty on spend.
# * **Small coefficients on `subscription_flag` / `discount_flag`** mean the promo levers in
#   this dataset are not moving basket size, which is itself a useful business answer.

# %%
if results_df.loc[results_df["model"].str.startswith("Ridge"), "r2"].iloc[0] < 0.05:
    print(
        "NOTE: R² is very low. The available attributes carry almost no linear signal about\n"
        "      spend in this dataset. Before reaching for a fancier model, check whether the\n"
        "      target is genuinely predictable here — for this Kaggle set, it largely isn't.\n"
        "      Segmentation (below) is the more informative analysis."
    )


# %% [markdown]
# # 5. K-Means customer segmentation
# Unsupervised grouping on behavioural features: age, spend, review rating, purchase history
# and cadence. Features are standardised first — K-Means uses Euclidean distance, so raw
# scales would let `previous_purchases` dominate `review_rating`.

# %%
CLUSTER_FEATURES = [
    c for c in [
        "age",
        "purchase_amount",
        "review_rating",
        "previous_purchases",
        "purchase_frequency_days",
        "subscription_flag",
    ] if c in data.columns
]

cluster_X = data[CLUSTER_FEATURES].dropna()
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_X)
print(f"Clustering on {cluster_scaled.shape[0]} rows x {cluster_scaled.shape[1]} features: "
      f"{CLUSTER_FEATURES}")


# %% [markdown]
# ## 5a. Elbow plot + silhouette scores

# %%
k_range = range(2, 11)
inertias, silhouettes = [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(cluster_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(cluster_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(list(k_range), inertias, marker="o", color="#264653")
axes[0].set_title("Elbow method — within-cluster sum of squares")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), silhouettes, marker="o", color="#e9c46a")
axes[1].set_title("Silhouette score by k")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

for k, inertia, sil in zip(k_range, inertias, silhouettes):
    print(f"k={k:2d}   inertia={inertia:12.1f}   silhouette={sil:.4f}")


# %% [markdown]
# ## 5b. Fit the chosen k and profile the segments
# Set `CHOSEN_K` from the elbow/silhouette output above — don't accept the default blindly.

# %%
CHOSEN_K = 4

kmeans_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=CHOSEN_K, random_state=RANDOM_STATE, n_init=10)),
])
cluster_labels = kmeans_pipeline.fit_predict(cluster_X)

segmented = cluster_X.copy()
segmented["cluster"] = cluster_labels

profile = (
    segmented.groupby("cluster")
    .agg(customers=("age", "size"), **{c: (c, "mean") for c in CLUSTER_FEATURES})
    .round(2)
)
profile["share_pct"] = (100 * profile["customers"] / profile["customers"].sum()).round(1)
print(f"Segment profile (k={CHOSEN_K}):\n")
print(profile.to_string())

print(f"\nOverall silhouette at k={CHOSEN_K}: "
      f"{silhouette_score(kmeans_pipeline.named_steps['scaler'].transform(cluster_X), cluster_labels):.4f}")


# %% [markdown]
# ## 5c. Visualise the segments

# %%
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    segmented["previous_purchases"],
    segmented["purchase_amount"],
    c=segmented["cluster"],
    cmap="viridis",
    alpha=0.6,
    s=18,
)
plt.colorbar(scatter, label="Cluster")
plt.xlabel("Previous purchases")
plt.ylabel("Purchase amount (USD)")
plt.title(f"Customer segments (k={CHOSEN_K})")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# Name the segments yourself from the profile table — e.g. high-frequency low-basket vs
# occasional high-basket — rather than assuming cluster 0 is always "best customers".


# %% [markdown]
# # 6. Persist the artefacts for deployment
# The whole `Pipeline` is saved, not just the estimator, so the serving code can pass a raw
# feature DataFrame and the preprocessing runs identically to training. Also saved: the
# K-Means pipeline, the exact feature list, and a small metadata dict for versioning.

# %%
OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(final_model, os.path.join(OUTPUT_DIR, "model.pkl"))
joblib.dump(kmeans_pipeline, os.path.join(OUTPUT_DIR, "kmeans_model.pkl"))
joblib.dump(
    {
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": FEATURES,
        "cluster_features": CLUSTER_FEATURES,
        "ridge_alpha": float(best_alpha),
        "test_metrics": results[-1],
        "chosen_k": CHOSEN_K,
        "source": source_used,
        "sklearn_version": __import__("sklearn").__version__,
    },
    os.path.join(OUTPUT_DIR, "model_metadata.pkl"),
)

for fname in sorted(os.listdir(OUTPUT_DIR)):
    size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, fname)) / 1024
    print(f"  {fname:<24} {size_kb:7.1f} KB")

# In Colab, download them locally:
# from google.colab import files
# files.download("artifacts/model.pkl")


# %% [markdown]
# ## 6a. Reload-and-predict smoke test
# Always verify the artefact round-trips before shipping it.

# %%
loaded_model = joblib.load(os.path.join(OUTPUT_DIR, "model.pkl"))
loaded_meta = joblib.load(os.path.join(OUTPUT_DIR, "model_metadata.pkl"))

sample = X_test.head(5)
reloaded_preds = loaded_model.predict(sample)

check = sample.copy()
check["actual"] = y_test.head(5).values
check["predicted"] = reloaded_preds.round(2)
print(check[["age", "review_rating", "previous_purchases",
             "subscription_flag", "actual", "predicted"]].to_string(index=False))

assert np.allclose(reloaded_preds, final_model.predict(sample)), "Reloaded model diverged!"
print("\nRound-trip check passed — reloaded model matches the in-memory pipeline.")


# %% [markdown]
# # 7. Minimal serving snippet (for reference)
# Drop this into `app.py` next to `artifacts/` and run `uvicorn app:app --reload`.
#
# ```python
# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib, pandas as pd
#
# model = joblib.load("artifacts/model.pkl")
# meta = joblib.load("artifacts/model_metadata.pkl")
# app = FastAPI(title="Purchase Amount Predictor")
#
# class Customer(BaseModel):
#     age: int
#     review_rating: float
#     previous_purchases: int
#     subscription_flag: int
#     discount_flag: int
#     purchase_frequency_days: int
#     annual_purchase_rate: float
#     rating_vs_category_avg: float
#     gender: str
#     category: str
#     season: str
#     shipping_type: str
#     payment_method: str
#     loyalty_segment: str
#
# @app.post("/predict")
# def predict(customer: Customer):
#     row = pd.DataFrame([customer.dict()])[meta["all_features"]]
#     return {"predicted_purchase_amount_usd": round(float(model.predict(row)[0]), 2)}
# ```

# %%
conn.close()
print("SQL connection closed. Pipeline complete.")
