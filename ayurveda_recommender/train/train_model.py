import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# STEP 1: Load & Normalize Data
# -----------------------------
df_symptoms = pd.read_csv("C:\\Users\\shiva\\Desktop\\ayurveda_recommender\\data\\data.csv")
df_meds     = pd.read_csv("C:\\Users\\shiva\\Desktop\\ayurveda_recommender\\data\\Drug prescription Dataset.csv")

# Strip & lowercase prognosis
df_symptoms.columns = df_symptoms.columns.str.strip()
df_symptoms['prognosis'] = df_symptoms['prognosis'].str.strip().str.lower()

# Strip, lowercase/uppercase med‑mapping cols
df_meds.columns = df_meds.columns.str.strip()
df_meds['disease']  = df_meds['disease'].str.strip().str.lower()
df_meds['gender']   = df_meds['gender'].str.strip().str.lower()
df_meds['severity'] = df_meds['severity'].str.strip().str.upper()
df_meds['drug']     = df_meds['drug'].str.strip()

# -----------------------------
# STEP 2: Train Disease Model
# -----------------------------
df_symptoms = df_symptoms.dropna()
X = df_symptoms.drop("prognosis", axis=1)
y = df_symptoms["prognosis"]

le_disease = LabelEncoder()
y_enc = le_disease.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)
model_disease = RandomForestClassifier(n_estimators=100, random_state=42)
model_disease.fit(X_train, y_train)
print(f"Disease Model Accuracy: {model_disease.score(X_test, y_test):.2f}")

# -----------------------------
# STEP 3: Train Medicine Model
# -----------------------------
df_meds = df_meds.dropna()

# Features: disease, age, gender, severity
X2 = df_meds[["disease", "age", "gender", "severity"]]
y2 = df_meds["drug"]

le_med_disease = LabelEncoder()
le_gender      = LabelEncoder()
le_severity    = LabelEncoder()
le_drug        = LabelEncoder()

X2_enc = pd.DataFrame({
    "disease":  le_med_disease.fit_transform(X2["disease"]),
    "age":      X2["age"].astype(int),
    "gender":   le_gender.fit_transform(X2["gender"]),
    "severity": le_severity.fit_transform(X2["severity"]),
})
y2_enc = le_drug.fit_transform(y2)

model_medicine = RandomForestClassifier(n_estimators=100, random_state=42)
model_medicine.fit(X2_enc, y2_enc)
print(f"Medicine Model Accuracy (train): {model_medicine.score(X2_enc, y2_enc):.2f}")

# -----------------------------
# STEP 4: Prediction Pipeline
# -----------------------------
# 1) Predict disease from symptoms
sample_symptoms = X.iloc[0].values.reshape(1, -1)
pred_d_idx      = model_disease.predict(sample_symptoms)
pred_disease    = le_disease.inverse_transform(pred_d_idx)[0]
print(f"\nPredicted Disease: {pred_disease}")

# 2) Recommend a drug
#    Provide real patient info here:
patient_age      = 4
patient_gender   = "male"
patient_severity = "LOW"

# Lower/upper to match encoder
pd_g = patient_gender.strip().lower()
pd_s = patient_severity.strip().upper()
pd_d = pred_disease.strip().lower()

if pd_d not in le_med_disease.classes_:
    print(f"No drugs found for '{pred_disease}'.")
else:
    enc_features = [
        le_med_disease.transform([pd_d])[0],
        patient_age,
        le_gender.transform([pd_g])[0],
        le_severity.transform([pd_s])[0]
    ]
    pred_drug_idx = model_medicine.predict([enc_features])
    pred_drug     = le_drug.inverse_transform(pred_drug_idx)[0]
    print(f"Recommended Drug: {pred_drug}")
import joblib

# … after fitting model_disease, le_disease, model_medicine, le_med_disease, le_gender, le_severity, le_drug …

# symptom feature order (must match the columns you trained on)
symptom_features = [
    "acidity","indigestion","headache","blurred_and_distorted_vision",
    "excessive_hunger","muscle_weakness","stiff_neck","swelling_joints",
    "movement_stiffness","depression","irritability","visual_disturbances",
    "painful_walking","abdominal_pain","nausea","vomiting",
    "blood_in_mucus","Fatigue","Fever","Dehydration","loss_of_appetite",
    "cramping","blood_in_stool","gnawing","upper_abdomain_pain",
    "fullness_feeling","hiccups","abdominal_bloating","heartburn",
    "belching","burning_ache"
]

joblib.dump(model_disease,      "model_disease.pkl")
joblib.dump(le_disease,         "le_disease.pkl")
joblib.dump(model_medicine,     "model_medicine.pkl")
joblib.dump(le_med_disease,     "le_med_disease.pkl")
joblib.dump(le_gender,          "le_gender.pkl")
joblib.dump(le_severity,        "le_severity.pkl")
joblib.dump(le_drug,            "le_drug.pkl")
joblib.dump(symptom_features,   "symptom_features.pkl")

print("Models, encoders, and feature list saved.")
