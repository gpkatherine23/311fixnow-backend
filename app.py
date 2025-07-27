from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load cleaned data
df = pd.read_csv("data_311_2023_analysis_ver1.csv")

# Load model and mappings
model = joblib.load("model.pkl")
zip_map = pd.read_csv("zip_code_map.csv")
type_map = pd.read_csv("request_type_map.csv")

# Convert mappings to dictionaries (strip spaces just in case)
zip_dict = dict(zip(zip_map["zip_code"].astype(str).str.strip(), zip_map["zip_code_code"]))
type_dict = dict(zip(type_map["issue_type_reduced"].astype(str).str.strip(), type_map["request_type_code"]))

@app.route("/")
def home():
    return "✅ 311FixNow Backend is running"

# ✅ Route 1: Request Types
@app.route("/api/top-issues")
def top_issues():
    top = df["issue_type_reduced"].value_counts().head(10).to_dict()
    return jsonify(top)

# ✅ Route 2: Issues by ZIP code
@app.route("/api/issues/<zip_code>")
def issues_by_zip(zip_code):
    zip_code = zip_code.strip()
    df["zip_code"] = df["zip_code"].fillna("").astype(str).str.strip()
    
    print("🔍 ZIP searched:", zip_code)

    filtered = df[df["zip_code"].str.startswith(zip_code)]

    if filtered.empty:
        print("⚠️ No records found for ZIP:", zip_code)
        return jsonify([]), 404

    print(f"✅ Found {len(filtered)} records for ZIP:", zip_code)
    return jsonify(filtered.head(50).to_dict(orient="records"))

# ✅ Route 3: Predict days to resolution
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    zip_code = str(data.get("zip_code", "")).strip()
    issue_type = str(data.get("issue_type_reduced", "")).strip()

    # Clean dictionary keys
    zip_lookup = {str(k).strip(): v for k, v in zip_dict.items()}
    type_lookup = {str(k).strip(): v for k, v in type_dict.items()}

    print("🧠 Predicting...")
    print("Received zip_code:", zip_code)
    print("Received issue_type:", issue_type)
    print("Valid zip? ", zip_code in zip_lookup)
    print("Valid issue_type? ", issue_type in type_lookup)

    if zip_code not in zip_lookup or issue_type not in type_lookup:
        return jsonify({
            "error": "Invalid ZIP code or request type",
            "valid_zip_codes": list(zip_lookup.keys())[:5],
            "valid_issue_types": list(type_lookup.keys())[:5]
        }), 400

    zip_encoded = zip_lookup[zip_code]
    type_encoded = type_lookup[issue_type]
    prediction = model.predict([[zip_encoded, type_encoded]])

    return jsonify({"predicted_days": round(prediction[0], 2)})

# ✅ Route 4: Help - show valid zip and issue values
@app.route("/api/help")
def help():
    print("✅ /api/help accessed")
    print("Sample ZIPs:", list(zip_dict.keys())[:3])
    print("Sample Issues:", list(type_dict.keys())[:3])

    return jsonify({
        "valid_zip_codes": list(zip_dict.keys())[:45],
        "valid_issue_types": list(type_dict.keys())[:45]
    })

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000) 
# app.run(debug=True)
