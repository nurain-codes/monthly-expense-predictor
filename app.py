import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Expense Predictor",
    page_icon="💰",
    layout="centered"
)

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------
MODEL_FILE = "expense_prediction_model.pkl"

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error(
        "Model file not found. Please keep "
        "'expense_prediction_model.pkl' in the same folder as this app."
    )
    st.stop()

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------
st.title("💰 Monthly Expense Predictor")
st.write("Enter your financial details and predict your monthly expenses.")

st.divider()

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
with st.form("expense_form"):

    st.subheader("📋 Financial Details")

    col1, col2 = st.columns(2)

    with col1:
        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0.0,
            value=4000.0,
            step=100.0
        )

        savings_rate = st.number_input(
            "Savings Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
            help="Enter 0.20 for 20%."
        )

        budget_goal = st.number_input(
            "Budget Goal",
            min_value=0.0,
            value=3000.0,
            step=100.0
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=0.0,
            max_value=900.0,
            value=700.0,
            step=10.0
        )

        debt_to_income_ratio = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01
        )

        loan_payment = st.number_input(
            "Loan Payment",
            min_value=0.0,
            value=500.0,
            step=50.0
        )

        investment_amount = st.number_input(
            "Investment Amount",
            min_value=0.0,
            value=300.0,
            step=50.0
        )

    with col2:
        subscription_services = st.number_input(
            "Subscription Services",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

        emergency_fund = st.number_input(
            "Emergency Fund",
            min_value=0.0,
            value=2000.0,
            step=100.0
        )

        transaction_count = st.number_input(
            "Transaction Count",
            min_value=0.0,
            value=45.0,
            step=1.0
        )

        discretionary_spending = st.number_input(
            "Discretionary Spending",
            min_value=0.0,
            value=500.0,
            step=50.0
        )

        essential_spending = st.number_input(
            "Essential Spending",
            min_value=0.0,
            value=1800.0,
            step=100.0
        )

        rent_or_mortgage = st.number_input(
            "Rent / Mortgage",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        financial_advice_score = st.number_input(
            "Financial Advice Score",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=1.0
        )

    predict_button = st.form_submit_button(
        "🔮 Predict Monthly Expense",
        use_container_width=True
    )

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if predict_button:

    input_data = pd.DataFrame([{
        "monthly_income": monthly_income,
        "savings_rate": savings_rate,
        "budget_goal": budget_goal,
        "credit_score": credit_score,
        "debt_to_income_ratio": debt_to_income_ratio,
        "loan_payment": loan_payment,
        "investment_amount": investment_amount,
        "subscription_services": subscription_services,
        "emergency_fund": emergency_fund,
        "transaction_count": transaction_count,
        "discretionary_spending": discretionary_spending,
        "essential_spending": essential_spending,
        "rent_or_mortgage": rent_or_mortgage,
        "financial_advice_score": financial_advice_score
    }])

    prediction = model.predict(input_data)[0]
    annual_prediction = prediction * 12

    st.divider()
    st.subheader("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted Monthly Expense",
            f"₹{prediction:,.2f}"
        )

    with result_col2:
        st.metric(
            "Estimated Annual Expense",
            f"₹{annual_prediction:,.2f}"
        )

    st.success("Prediction completed successfully! 🎉")

st.divider()
st.caption("Machine Learning Mini Project • Random Forest Regression")
