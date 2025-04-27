from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  # 👈 Add this line

app = Flask(__name__)
CORS(app)  # 👈 Add this line to enable CORS for all routes
# Load models and encoders
model_disease    = joblib.load("model_disease.pkl")
le_disease       = joblib.load("le_disease.pkl")
symptom_features = joblib.load("symptom_features.pkl")

model_medicine   = joblib.load("model_medicine.pkl")
le_med_disease   = joblib.load("le_med_disease.pkl")
le_gender        = joblib.load("le_gender.pkl")
le_severity      = joblib.load("le_severity.pkl")
le_drug          = joblib.load("le_drug.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    try:
        # Extract values from frontend
        symptoms = data.get("symptoms", {})         # dict of symptom_name: 0/1
        age = int(data.get("age", 0))
        gender = data.get("gender", "").lower()
        severity = data.get("severity", "")
        
        print("Received symptoms:", symptoms)
        print(" Age:", age, "| Gender:", gender, "| Severity:", severity)

        # Ensure symptom order matches training
        symptom_vector = [symptoms.get(symptom, 0) for symptom in symptom_features]
        df_symptoms = pd.DataFrame([symptom_vector], columns=symptom_features)

        # Predict disease
        pred_disease_idx = model_disease.predict(df_symptoms)[0]
        pred_disease = le_disease.inverse_transform([pred_disease_idx])[0]

        print(" Predicted Disease:", pred_disease)

        # Encode inputs for medicine model
        encoded_input = pd.DataFrame([{
            "disease": le_med_disease.transform([pred_disease])[0],
            "age": age,
            "gender": le_gender.transform([gender])[0],
            "severity": le_severity.transform([severity])[0]
        }])

        # Predict medicine
        pred_drug_idx = model_medicine.predict(encoded_input)[0]
        pred_drug = le_drug.inverse_transform([pred_drug_idx])[0]

        print(" Recommended Drug:", pred_drug)

        return jsonify({"disease": pred_disease, "medicine": pred_drug})

    except Exception as e:
        print(" Error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
