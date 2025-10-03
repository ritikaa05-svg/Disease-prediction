import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Importing os to help with path checks
import pickle  # Added for saving/loading the model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Project Setup and Data Acquisition ---
print("--- 1. Data Acquisition and Initial Inspection ---")

# --- ACTION REQUIRED: Load your actual full, large dataset here. ---
# FIX: Using the relative path. Ensure 'disease_data.csv' is in the SAME folder.
data_filename = r'C:\Users\Ritika Kunwar\PycharmProjects\Disease prediction\heart.csv'  # <-- EDIT THIS LINE if your file is named differently (e.g., 'Training.csv')

try:
    # Check if the file exists in the current working directory for clarity
    current_dir = os.getcwd()
    if not os.path.exists(data_filename):
        print(f"\n!! CRITICAL HINT: File '{data_filename}' not found in current directory: {current_dir} !!")
        print("Please ensure the CSV file is located here or update the 'data_filename' variable above.")

    df = pd.read_csv(data_filename)

except FileNotFoundError:
    print("\n!! FATAL ERROR: Data file not found !!")
    print(f"Please ensure your dataset is named '{data_filename}' and is in the same directory as this script.")
    print("If it is not, replace the filename on line 24 with the correct absolute path to your CSV file.")
    exit()
except Exception as e:
    # Catches the PermissionError and other reading issues
    print(f"\n!! FATAL ERROR during file reading: {e} !!")
    print(
        "If this is a Permission Error, ensure the file is not open in another program (like Excel) and the path is correct.")
    exit()
# --------------------------------------------------------------------------

# --- Standard Data Cleanup for this type of symptom dataset ---

# 1. Drop extraneous columns, especially the common 'Unnamed: X' artifact
if 'Unnamed: 133' in df.columns:
    df.drop('Unnamed: 133', axis=1, inplace=True)

# 2. Identify symptom columns (all columns except 'target' - the disease outcome)
# FIX: Changed exclusion column from 'prognosis' to 'target'
symptom_cols = [col for col in df.columns if col != 'target']

# 3. FIX: Explicitly convert symptom columns to numeric.
# 'errors='coerce' will turn any problematic string values into NaN.
for col in symptom_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Fill any NaN values with 0.
df.fillna(0, inplace=True)

# 5. Remove rows with all zero symptoms
# This step is often less relevant for heart disease data, but kept for consistency
df['symptom_sum'] = df[symptom_cols].sum(axis=1)
df = df[df['symptom_sum'] > 0].drop('symptom_sum', axis=1)

# 6. Remove duplicate rows
df.drop_duplicates(inplace=True)
# -------------------------------------------------------------

print(f"Dataset shape after cleanup: {df.shape}")
print("\nFirst 5 rows of the data:")
print(df.head())
print("\nData Information (Dtypes and Nulls):")
df.info()

# --- CRITICAL DATA VALIDATION ---
# FIX: Checking 'target' column instead of 'prognosis'
if 'target' not in df.columns:
    print("\n\n!! CRITICAL ERROR: TARGET COLUMN MISSING !!")
    print("The required target column 'target' was not found after cleanup. Check your data file structure.")
    exit()

unique_classes = df['target'].nunique()
if unique_classes <= 1:
    print("\n\n!! CRITICAL ERROR: TARGET VARIABLE FAILURE !!")
    print(f"The 'target' column contains only {unique_classes} unique class(es).")
    print("This means the dataset is corrupted or only contains one outcome type.")
    print(
        "Please check your 'disease_data.csv' to ensure the 'target' column has different class labels (like 0 and 1).")
    # Exit cleanly as no model can be trained
    exit()
# ----------------------------------


# --- 2. Exploratory Data Analysis (EDA) and Preprocessing ---
print("\n--- 2. Preprocessing and EDA ---")

# 2.1 Feature and Target Separation
# FIX: Dropping 'target' for features X, and setting 'target' as target variable y
X = df.drop('target', axis=1)
y = df['target']  # Target column is now 'target'

# 2.2 Label Encoding for Multi-Class Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Note: For binary target (0/1), LabelEncoder is fine but doesn't change the values.
print(f"Original unique diseases/outcomes: {le.classes_}")
print(f"Encoded classes: 0 to {len(le.classes_) - 1}")

# 2.3 Data Splitting (80% Train, 20% Test)
# Stratify is used to ensure proportional representation of each disease in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

print("\nData encoded and split successfully.")

# --- 3. Model Training and Evaluation ---
print("\n--- 3. Model Training and Evaluation ---")

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000),
    "SVM (C=1, Kernel=RBF)": SVC(random_state=42, probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    # Changed eval_metric for binary classification consistency
}

results = {}

# Get all encoded labels (0, 1, ...)
all_labels = le.transform(le.classes_)

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], labels=all_labels,
                                   zero_division=0)

    results[name] = {
        'accuracy': acc,
        'cm': cm,
        'report': report
    }

    print(f"[{name}] Accuracy: {acc:.4f}")

# --- 4. Summary and Visualization ---
print("\n--- 4. Summary of Model Performance ---")

best_accuracy = 0
best_model_name = ""
best_model = None
feature_names_list = X.columns.tolist()  # Store feature names list here

for name, res in results.items():
    current_model = models[name]
    print(f"\n=======================================================")
    print(f"MODEL: {name}")
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"Classification Report (Precision, Recall, F1-Score):\n{res['report']}")
    print(f"=======================================================")

    # Determine the best model
    if res['accuracy'] >= best_accuracy:
        best_accuracy = res['accuracy']
        best_model_name = name
        best_model = current_model

print(
    f"\n🏆 The best performing model based on simple accuracy is: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# --- Deployment Step: Save the model and encoder ---
try:
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names_list, f)
    print("\n✅ Deployment files saved: 'best_model.pkl', 'label_encoder.pkl', and 'feature_names.pkl'.")
    print("You can now use 'prediction_service.py' to run new predictions.")
except Exception as e:
    print(f"\n!! WARNING: Could not save deployment files: {e}")
# ----------------------------------------------------


# Optional: Visualization of the best model's confusion matrix
plt.figure(figsize=(10, 8))
# Setting xticklabels and yticklabels to strings of the classes (e.g., '0', '1')
sns.heatmap(results[best_model_name]['cm'], annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(c) for c in le.classes_],
            yticklabels=[str(c) for c in le.classes_])
plt.title(f'Confusion Matrix for {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Optional: Feature Importance (Example using Random Forest)
rf_model = models["Random Forest"]
if hasattr(rf_model, 'feature_importances_'):
    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(
        'importance', ascending=False)

    print("\nTop 5 Feature Importances (from Random Forest):")
    print(feature_importance_df.head())

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(5))
    plt.title('Top 5 Feature Importances')
    plt.show()

# --- 5. Prediction Function (How to Use the Model) ---
print("\n--- 5. Using the Model for Prediction ---")


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
    # The output is now a number (0 or 1), so we convert it to a string for display if needed.
    predicted_outcome = str(label_encoder.inverse_transform([encoded_prediction])[0])

    # 4. Get the probability vector
    probability_vector = model.predict_proba(new_patient_df)[0]

    return predicted_outcome, probability_vector


# --- Example of New Patient Data ---
if best_model:
    # Generate a sample patient dictionary where all features are 0 (This is highly customized)
    # Since this is Heart Disease data, we need realistic inputs.
    # Let's use a pattern for a high-risk patient (older age, high cholesterol, low maximum heart rate)

    sample_input = {col: 0 for col in X.columns.tolist()}

    # Common features for Heart Disease prediction (assuming this is the Cleveland dataset structure)
    sample_input['age'] = 65
    sample_input['sex'] = 1  # Male
    sample_input['cp'] = 3  # Atypical angina (high risk chest pain type)
    sample_input['trestbps'] = 160  # High blood pressure
    sample_input['chol'] = 300  # High cholesterol
    sample_input['thalach'] = 120  # Low maximum heart rate achieved
    sample_input['exang'] = 1  # Exercise induced angina (yes)
    sample_input['oldpeak'] = 3.0  # ST depression induced by exercise (high value)

    try:
        final_outcome, final_probabilities = predict_new_patient(
            model=best_model,
            patient_data_dict=sample_input,
            feature_names=X.columns.tolist(),
            label_encoder=le
        )

        # Interpret binary outcome (0 or 1)
        interpretation = "Heart Disease Not Predicted (0)" if final_outcome == '0' else "Heart Disease Predicted (1)"

        # Find the confidence for the predicted class
        predicted_index = le.transform([int(final_outcome)])[0]
        confidence = final_probabilities[predicted_index]

        print(f"\n--- Prediction Results for High-Risk Patient (Using {best_model_name}) ---")
        print(
            f"Input Profile: Age={sample_input['age']}, ChestPain={sample_input['cp']}, Cholesterol={sample_input['chol']}, MaxHR={sample_input['thalach']}")
        print(f"Final Prediction (Encoded): {final_outcome}")
        print(f"Interpretation: {interpretation}")
        print(f"Confidence in Prediction: {confidence:.2%}")

    except Exception as e:
        print(f"\nError during prediction: {e}")
        print("This usually happens if the model failed to train or the input features are mismatched.")

else:
    print("\nError: The best model could not be determined or loaded.")
