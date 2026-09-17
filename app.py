import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration & UI Title
st.set_page_config(
    page_title="Customer Spend Predictor",
    page_icon="🛒",
    layout="wide",
)
st.title("🛒 Customer Spend & Analytics Predictor")
st.write(
    "Predict expected purchase amounts and analyze customer segments using Machine Learning."
)


# 2. Load ML Artifacts
@st.cache_resource
def load_models():
    model = joblib.load("artifacts/model.pkl")
    kmeans = joblib.load("artifacts/kmeans_model.pkl")
    metadata = joblib.load("artifacts/model_metadata.pkl")
    return model, kmeans, metadata


model, kmeans, metadata = load_models()

# 3. Build UI Input Forms
st.sidebar.header("Customer Profile")

age = st.sidebar.slider("Age", 18, 70, 35)
review_rating = st.sidebar.slider("Review Rating", 1.0, 5.0, 4.0, 0.1)
previous_purchases = st.sidebar.number_input(
    "Previous Purchases", 0, 100, 15
)
purchase_frequency_days = st.sidebar.selectbox(
    "Purchase Frequency (Days)", [7, 14, 30, 90, 365]
)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
category = st.sidebar.selectbox(
    "Category", ["Clothing", "Footwear", "Outerwear", "Accessories"]
)
season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
shipping_type = st.sidebar.selectbox(
    "Shipping Type",
    [
        "Express",
        "Free Shipping",
        "Standard",
        "Next Day Air",
        "2-Day Shipping",
        "Store Pickup",
    ],
)
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Credit Card", "Bank Transfer", "PayPal", "Venmo", "Cash", "Debit Card"],
)

subscription_status = st.sidebar.radio("Subscription Status", ["Yes", "No"])
discount_applied = st.sidebar.radio("Discount Applied", ["Yes", "No"])

# 4. Feature Transformations
subscription_flag = 1 if subscription_status == "Yes" else 0
discount_flag = 1 if discount_applied == "Yes" else 0
annual_purchase_rate = round(365.0 / purchase_frequency_days, 3)
rating_vs_category_avg = 0.0  # Default centered baseline

if previous_purchases <= 10:
    loyalty_segment = "New"
elif previous_purchases <= 30:
    loyalty_segment = "Returning"
else:
    loyalty_segment = "Loyal"

# 5. Model Inference on Button Click
if st.button("Predict Spend"):
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "review_rating": review_rating,
                "previous_purchases": previous_purchases,
                "subscription_flag": subscription_flag,
                "discount_flag": discount_flag,
                "purchase_frequency_days": purchase_frequency_days,
                "annual_purchase_rate": annual_purchase_rate,
                "rating_vs_category_avg": rating_vs_category_avg,
                "gender": gender,
                "category": category,
                "season": season,
                "shipping_type": shipping_type,
                "payment_method": payment_method,
                "loyalty_segment": loyalty_segment,
            }
        ]
    )

    # Make Predictions
    predicted_spend = model.predict(input_data)[0]

    # Render Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Predicted Purchase Amount",
            value=f"${round(predicted_spend, 2)}",
        )
    with col2:
        st.info(
            f"Customer Loyalty Tier: **{loyalty_segment}** | Purchase Rate: **{annual_purchase_rate} orders/year**"
        )
