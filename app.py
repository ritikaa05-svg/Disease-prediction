import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for cross-origin requests

# --- Configuration ---
MODEL_PATH = 'best_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'
FEATURES_PATH = 'feature_names.pkl'

app = Flask(__name__)
# Enable CORS so the web page (index.html) can talk to this API
CORS(app)

# Global variables for the loaded assets
loaded_model = None
loaded_encoder = None
feature_names = None


# --- Prediction Core Logic ---

def predict_new_patient(model, patient_data_dict, feature_names, label_encoder):
    """
    Predicts the outcome (e.g., 0 or 1) for a single new patient.

    This function handles input conversion and prediction logic.
    """

    # 1. Convert patient data dictionary into a DataFrame
    # Ensures the input features are in the correct order as expected by the model
    try:
        # Create DataFrame from input dictionary, ensuring column order matches training data
        new_patient_df = pd.DataFrame([patient_data_dict], columns=feature_names)
    except ValueError as e:
        # This handles cases where the required keys are missing from the input dict
        raise ValueError(f"Input features are missing or mismatched. Expected features: {feature_names}")

    # 2. Make the prediction (returns an encoded label, typically 0 or 1)
    encoded_prediction = model.predict(new_patient_df)[0]

    # 3. Decode the prediction back into the human-readable outcome (e.g., '0' or '1')
    predicted_outcome = str(label_encoder.inverse_transform([encoded_prediction])[0])

    # 4. Get the probability vector
    probability_vector = model.predict_proba(new_patient_df)[0]

    return predicted_outcome, probability_vector


# --- Model Loading (Executed once when the server starts) ---

def load_assets():
    """Loads the model, encoder, and feature names into global variables."""
    global loaded_model, loaded_encoder, feature_names
    print("--- Loading Deployed Model and Encoder ---")
    try:
        # Load the saved model
        with open(MODEL_PATH, 'rb') as f:
            loaded_model = pickle.load(f)
        # Load the saved label encoder
        with open(ENCODER_PATH, 'rb') as f:
            loaded_encoder = pickle.load(f)
        # Load the feature names list (required for correct DataFrame structure)
        with open(FEATURES_PATH, 'rb') as f:
            feature_names = pickle.load(f)

        print(f"✅ Model ({MODEL_PATH}), Encoder, and Features successfully loaded.")
        print(f"Model is ready to serve predictions.")

    except FileNotFoundError:
        print(f"\n!! CRITICAL ERROR: Deployment files not found !!")
        print(
            f"Please run 'python disease_prediction.py' first to create {MODEL_PATH}, {ENCODER_PATH}, and {FEATURES_PATH}.")
        exit()
    except Exception as e:
        print(f"\n!! FATAL ERROR during model loading: {e} !!")
        exit()


# --- Root Endpoint (For status checks) ---
@app.route('/', methods=['GET'])
def home():
    """Provides status and instructions for the API."""
    return jsonify({
        "status": "API Running",
        "message": "Welcome to the Heart Disease Prediction API.",
        "instructions": "Send a POST request to the /predict endpoint with a JSON payload of patient features.",
        "required_features": feature_names if feature_names else "Assets not loaded."
    })


# --- Prediction API Endpoint ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives patient data via POST request and returns the prediction.
    Expected JSON format: {"age": 65, "sex": 1, "cp": 3, "trestbps": 160, ...}
    """
    if loaded_model is None:
        return jsonify({"error": "Model not loaded. Server startup failed."}), 503

    # Check for valid JSON input
    if not request.json:
        return jsonify({"error": "Invalid input: Please provide patient data as JSON."}), 400

    patient_data_dict = request.json

    # Simple validation to ensure all features are present
    if not all(col in patient_data_dict for col in feature_names):
        missing_features = [col for col in feature_names if col not in patient_data_dict]
        return jsonify({
            "error": "Missing required features.",
            "missing": missing_features,
            "required": feature_names
        }), 400

    try:
        final_outcome, final_probabilities = predict_new_patient(
            model=loaded_model,
            patient_data_dict=patient_data_dict,
            feature_names=feature_names,
            label_encoder=loaded_encoder
        )

        # Calculate confidence
        predicted_index = loaded_encoder.transform([int(final_outcome)])[0]
        confidence = final_probabilities[predicted_index]

        # Interpretation
        interpretation = "Predicted Heart Disease" if final_outcome == '1' else "Predicted No Heart Disease"

        # Return results as JSON
        response = {
            "prediction": int(final_outcome),
            "interpretation": interpretation,
            "confidence_percent": round(confidence * 100, 2),
            "probabilities": {str(c): round(prob, 4) for c, prob in zip(loaded_encoder.classes_, final_probabilities)}
        }

        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Catch unexpected errors during prediction
        print(f"Unexpected prediction error: {e}")
        return jsonify({"error": "Internal server error during prediction."}), 500


# --- Start Server ---

if __name__ == '__main__':
    # Load assets before starting the API
    load_assets()
    # The 'host'='0.0.0.0' allows external connections
    app.run(host='0.0.0.0', port=5000, debug=True)
