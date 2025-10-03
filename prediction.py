import pandas as pd
import numpy as np
import pickle
import os


# Define the prediction function (copied from the training script)
def predict_new_patient(model, patient_data_dict, feature_names, label_encoder):
    """
    Predicts the outcome (e.g., 0 or 1) for a single new patient.
    """

    # 1. Convert patient data dictionary into a DataFrame
    # IMPORTANT: Ensure the column order matches the feature_names list
    new_patient_df = pd.DataFrame([patient_data_dict], columns=feature_names)

    # 2. Make the prediction (returns an encoded label, typically 0 or 1)
    encoded_prediction = model.predict(new_patient_df)[0]

    # 3. Decode the prediction back into the human-readable outcome (e.g., 0 or 1)
    predicted_outcome = str(label_encoder.inverse_transform([encoded_prediction])[0])

    # 4. Get the probability vector
    probability_vector = model.predict_proba(new_patient_df)[0]

    return predicted_outcome, probability_vector


# --- Deployment Logic ---
print("--- Loading Deployed Model and Encoder ---")

# Define file paths
MODEL_PATH = 'best_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'
FEATURES_PATH = 'feature_names.pkl'

# Load the deployed assets
try:
    with open(MODEL_PATH, 'rb') as f:
        loaded_model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        loaded_encoder = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        feature_names = pickle.load(f)

    print(f"✅ Model ({MODEL_PATH}), Encoder, and Features successfully loaded.")

except FileNotFoundError:
    print(f"\n!! CRITICAL ERROR: Deployment files not found !!")
    print(
        f"Please run 'python disease_prediction.py' first to create {MODEL_PATH}, {ENCODER_PATH}, and {FEATURES_PATH}.")
    exit()
except Exception as e:
    print(f"\n!! FATAL ERROR during model loading: {e} !!")
    exit()

# --- Define New Patient Data ---

# NOTE: This dictionary MUST contain all the feature names found in feature_names list.
# For simplicity, we initialize all features to 0 and set the test values.
test_patient_data = {col: 0 for col in feature_names}

# Example Patient Data (High Risk Profile)
test_patient_data['age'] = 65
test_patient_data['sex'] = 1
test_patient_data['cp'] = 3
test_patient_data['trestbps'] = 160
test_patient_data['chol'] = 300
test_patient_data['thalach'] = 120
test_patient_data['exang'] = 1
test_patient_data['oldpeak'] = 3.0
# The remaining features will remain 0 unless explicitly set above.

# --- Run Prediction ---
print("\n--- Running Prediction on New Patient Data ---")

try:
    final_outcome, final_probabilities = predict_new_patient(
        model=loaded_model,
        patient_data_dict=test_patient_data,
        feature_names=feature_names,
        label_encoder=loaded_encoder
    )

    # Interpretation
    interpretation = "Heart Disease Not Predicted (0)" if final_outcome == '0' else "Heart Disease Predicted (1)"

    # Get the confidence for the predicted class
    predicted_index = loaded_encoder.transform([int(final_outcome)])[0]
    confidence = final_probabilities[predicted_index]

    print(f"\n--- Prediction Results ---")
    print(
        f"Input Profile (Key features): Age={test_patient_data['age']}, ChestPain={test_patient_data['cp']}, Cholesterol={test_patient_data['chol']}")
    print(f"Final Prediction (Encoded): {final_outcome}")
    print(f"Interpretation: {interpretation}")
    print(f"Confidence in Prediction: {confidence:.2%}")

except Exception as e:
    print(f"\nError during prediction: {e}")
