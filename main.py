import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import joblib


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("dataset/personal_finance_tracker_dataset.csv")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "monthly_income",
    "savings_rate",
    "budget_goal",
    "credit_score",
    "debt_to_income_ratio",
    "loan_payment",
    "investment_amount",
    "subscription_services",
    "emergency_fund",
    "transaction_count",
    "discretionary_spending",
    "essential_spending",
    "rent_or_mortgage",
    "financial_advice_score"
]

target = "monthly_expense_total"

X = data[features]
y = data[target]


# ============================================================
# 3. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 4. LINEAR REGRESSION MODEL
# ============================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)

print("\n--- Linear Regression ---")
print("MAE:", linear_mae)
print("R2 Score:", linear_r2)


# ============================================================
# 5. RANDOM FOREST MODEL
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\n--- Random Forest ---")
print("MAE:", rf_mae)
print("R2 Score:", rf_r2)


# ============================================================
# 6. SAVE RANDOM FOREST MODEL
# ============================================================

joblib.dump(rf_model, "expense_prediction_model.pkl")

print("\nModel saved successfully!")
print("Saved as: expense_prediction_model.pkl")


# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)


# ============================================================
# 8. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(y_test, rf_predictions)

plt.xlabel("Actual Monthly Expense")
plt.ylabel("Predicted Monthly Expense")
plt.title("Actual vs Predicted Monthly Expenses")

plt.tight_layout()
plt.show()


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

models = ["Linear Regression", "Random Forest"]
mae_values = [linear_mae, rf_mae]

plt.figure(figsize=(8, 5))

plt.bar(models, mae_values)

plt.xlabel("Models")
plt.ylabel("Mean Absolute Error")
plt.title("Model Comparison using MAE")

plt.tight_layout()
plt.show()


# ============================================================
# 10. CORRELATION HEATMAP
# ============================================================

correlation = data[features + [target]].corr()

plt.figure(figsize=(12, 9))

plt.imshow(correlation, aspect="auto")
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 11. USER INPUT FOR PREDICTION
# ============================================================

print("\n==========================================")
print("       MONTHLY EXPENSE PREDICTOR")
print("==========================================")
print("Enter the following values:")
print()

user_values = []

for feature in features:
    value = float(input(f"{feature}: "))
    user_values.append(value)


# ============================================================
# 12. CREATE INPUT DATAFRAME
# ============================================================

user_data = pd.DataFrame(
    [user_values],
    columns=features
)


# ============================================================
# 13. MAKE FINAL PREDICTION
# ============================================================

final_prediction = rf_model.predict(user_data)[0]

print("\n==========================================")
print(f"Predicted Monthly Expense: ₹{final_prediction:.2f}")
print(f"Estimated Annual Expense: ₹{final_prediction * 12:.2f}")
print("==========================================")
