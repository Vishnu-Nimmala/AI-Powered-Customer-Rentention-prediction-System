from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ── Load Model & Scaler ──────────────────────────────────────────────────────
model  = joblib.load("CHURN_APP.pkl")
scaler = joblib.load("standard_scaler.pkl")

# ── Exact 20 feature names as trained (must match scaler.feature_names_in_) ──
FEATURES = [
    'SeniorCitizen',
    'tenure_yeo_tri',
    'MonthlyCharges_box_CapMS',
    'TotalChargesAAA_yeo_CapMS',   # ← correct name (not TotalCharges_rep_yeo_CapMS)
    'gender_Male',
    'Partner_Yes',
    'Dependents_Yes',
    'PhoneService_ordinal',
    'MultipleLines_ordinal',
    'InternetService_ordinal',
    'OnlineSecurity_ordinal',
    'OnlineBackup_ordinal',
    'DeviceProtection_ordinal',
    'TechSupport_ordinal',
    'StreamingTV_ordinal',
    'StreamingMovies_ordinal',
    'Contract_ordinal',
    'PaperlessBilling_ordinal',
    'PaymentMethod_ordinal',
    'Sim_ordinal'
]

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Build ordered list of values matching FEATURES
        values = [float(data.get(feat, 0)) for feat in FEATURES]

        # Create DataFrame with correct column names
        df = pd.DataFrame([values], columns=FEATURES)

        # Scale → Predict
        df_scaled  = scaler.transform(df)
        pred       = model.predict(df_scaled)[0]
        proba      = model.predict_proba(df_scaled)[0]

        churn_prob  = round(float(proba[1]) * 100, 2)
        retain_prob = round(float(proba[0]) * 100, 2)

        if churn_prob >= 70:
            risk = "High"
        elif churn_prob >= 40:
            risk = "Medium"
        else:
            risk = "Low"

        prediction_label = "Will Churn" if bool(pred) else "Will Stay"

        return jsonify({
            "success":              True,
            "churn":                bool(pred),
            "prediction":           prediction_label,
            "risk_level":           risk,
            "churn_probability":    churn_prob,
            "retention_probability": retain_prob,   # ← matches index.html
            "message":              "Prediction generated successfully"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error":   str(e)
        }), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
