import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load your cleaned dataset
df = pd.read_csv(r"C:\Users\kathy\311fixnow-backend\data_311_2023_analysis_ver1.csv")

# Drop rows with missing values in key fields
df = df.dropna(subset=["zip_code", "issue_type_reduced", "actual_completed_days"])

# Convert zip code to string
df["zip_code"] = df["zip_code"].astype(int).astype(str)

# Encode zip_code and issue_type_reduced
df["zip_code_code"] = df["zip_code"].astype("category").cat.codes
df["request_type_code"] = df["issue_type_reduced"].astype("category").cat.codes

# Features and target
X = df[["zip_code_code", "request_type_code"]]
y = df["actual_completed_days"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

# Save the encoded mappings for future use
df[["zip_code", "zip_code_code"]].drop_duplicates().to_csv("zip_code_map.csv", index=False)
df[["issue_type_reduced", "request_type_code"]].drop_duplicates().to_csv("request_type_map.csv", index=False)

print("✅ Model and mappings saved.")
