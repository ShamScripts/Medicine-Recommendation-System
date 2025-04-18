from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from joblib import load
from fuzzywuzzy import process
import ast

app = Flask(__name__)

# ==========================================
# Load Model and Datasets
# ==========================================
try:
    Rf = load("model/RandomForest.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

try:
    sym_des = pd.read_csv("kaggle_dataset/symptoms_df.csv")
    precautions = pd.read_csv("kaggle_dataset/precautions_df.csv")
    workout = pd.read_csv("kaggle_dataset/workout_df.csv")
    description = pd.read_csv("kaggle_dataset/description.csv")
    medications = pd.read_csv("kaggle_dataset/medications.csv")
    diets = pd.read_csv("kaggle_dataset/diets.csv")
    training_df = pd.read_csv("kaggle_dataset/Training.csv")
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Failed to load one or more datasets: {e}")
    raise

# ==========================================
# Preprocess: Create Symptom-Disease Mapping
# ==========================================
symptoms_list_processed = {
    col.replace("_", " ").lower(): idx for idx, col in enumerate(training_df.columns[:-1])
}

# ==========================================
# Helper Functions
# ==========================================
def correct_spelling(symptom):
    match, score = process.extractOne(symptom.lower(), list(symptoms_list_processed.keys()))
    return match if score >= 80 else None

def predicted_value(symptoms):
    vector = np.zeros(len(symptoms_list_processed))
    for s in symptoms:
        if s in symptoms_list_processed:
            vector[symptoms_list_processed[s]] = 1
    input_df = pd.DataFrame([vector], columns=symptoms_list_processed.values())
    return Rf.predict(input_df)[0]

def information(predicted_dis):
    desc = description[description['Disease'] == predicted_dis]['Description'].values
    precaution = precautions[precautions['Disease'] == predicted_dis][[
        'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'
    ]].values.tolist()
    meds = medications[medications['Disease'] == predicted_dis]['Medication'].values.tolist()
    diet = diets[diets['Disease'] == predicted_dis]['Diet'].values.tolist()
    workout_plan = workout[workout['disease'] == predicted_dis]['workout'].values.tolist()
    return (
        desc[0] if desc else "No description available.",
        precaution[0] if precaution else [],
        ast.literal_eval(meds[0]) if meds else [],
        ast.literal_eval(diet[0]) if diet else [],
        workout_plan if workout_plan else []
    )

# ==========================================
# Routes
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
    raw_input = request.form.get('symptoms')
    if not raw_input:
        return render_template('index.html', message="Please enter symptoms.")

    patient_symptoms = [s.strip("[]' ") for s in raw_input.split(',')]
    corrected = []

    for sym in patient_symptoms:
        cor = correct_spelling(sym)
        if cor:
            corrected.append(cor)
        else:
            return render_template('index.html', message=f"Symptom '{sym}' not recognized.")

    pred_disease = predicted_value(corrected)
    desc, pre, meds, diet, work = information(pred_disease)

    return render_template('index.html',
                           symptoms=corrected,
                           predicted_disease=pred_disease,
                           dis_des=desc,
                           my_precautions=pre,
                           medications=meds,
                           my_diet=diet,
                           workout=work)

# ==========================================
# For Local Debugging Only
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)
