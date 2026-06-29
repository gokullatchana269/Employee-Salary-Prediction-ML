# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("salary_data.csv")

print(df.head())

# -----------------------------
# Handle Missing Values
# -----------------------------
df.dropna(inplace=True)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Education Level"] = le.fit_transform(df["Education Level"])
df["Job Title"] = le.fit_transform(df["Job Title"])

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Salary", axis=1)
y = df["Salary"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# Linear Regression
# ==========================================================

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n========== Linear Regression ==========")

print("MAE :", mean_absolute_error(y_test, lr_pred))

print("RMSE :", np.sqrt(mean_squared_error(y_test, lr_pred)))

print("R2 Score :", r2_score(y_test, lr_pred))

# ==========================================================
# Decision Tree
# ==========================================================

dt = DecisionTreeRegressor(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\n========== Decision Tree ==========")

print("MAE :", mean_absolute_error(y_test, dt_pred))

print("RMSE :", np.sqrt(mean_squared_error(y_test, dt_pred)))

print("R2 Score :", r2_score(y_test, dt_pred))

# ==========================================================
# Random Forest
# ==========================================================

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n========== Random Forest ==========")

print("MAE :", mean_absolute_error(y_test, rf_pred))

print("RMSE :", np.sqrt(mean_squared_error(y_test, rf_pred)))

print("R2 Score :", r2_score(y_test, rf_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(rf, "salary_model.pkl")

print("\nModel Saved Successfully!")

# -----------------------------
# Feature Importance
# -----------------------------
importance = rf.feature_importances_

feature_names = X.columns

plt.figure(figsize=(8,5))

plt.bar(feature_names, importance)

plt.title("Feature Importance")

plt.xticks(rotation=45)

plt.show()

# -----------------------------
# Actual vs Predicted
# -----------------------------
plt.figure(figsize=(7,6))

plt.scatter(y_test, rf_pred)

plt.xlabel("Actual Salary")

plt.ylabel("Predicted Salary")

plt.title("Actual vs Predicted Salary")

plt.show()
