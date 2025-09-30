import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # New: Required for multi-class string targets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Project Setup and Data Acquisition ---
print("--- 1. Data Acquisition and Initial Inspection ---")

# --- IMPORTANT: The original simulated data (Heart Disease) has been removed. ---
# --- Insert the code you used to load your symptom dataset here: ---
# Example: df = pd.read_csv('your_symptom_dataset.csv')
# The following CSV is ONLY for demonstration purposes now that we know your data structure.
data_csv = """itching,skin_rash,nodal_skin_eruptions,continuous_sneezing,prognosis
1,1,1,0,Fungal infection
0,0,0,1,Allergy
0,0,0,0,GERD
1,0,0,0,Chronic cholestasis
1,1,0,0,Drug Reaction
1,0,0,0,Chronic cholestasis
0,0,0,1,Allergy
"""
df = pd.read_csv(r"C:\Users\Ritika Kunwar\PycharmProjects\Disease prediction\Testing.csv")
# --------------------------------------------------------------------------

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows of the data:")
print(df.head())
print("\nData Information (Dtypes and Nulls):")
df.info()

# --- 2. Exploratory Data Analysis (EDA) and Preprocessing ---
print("\n--- 2. Preprocessing and EDA ---")

# 2.1 Feature and Target Separation (Changed 'target' to 'prognosis')
X = df.drop('prognosis', axis=1)  # FIX: Use 'prognosis' as the target column name
y = df['prognosis']

# 2.2 Label Encoding for Multi-Class Target
# Since 'prognosis' contains string disease names, we must convert them to numerical labels.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Original unique diseases: {le.classes_}")
print(f"Encoded classes: 0 to {len(le.classes_) - 1}")

# 2.3 Data Splitting (80% Train, 20% Test)
# FIX: Removed 'stratify=y_encoded' because many classes have only 1 sample, which prevents splitting.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 2.4 Data Scaling (REMOVED)
# The previous scaling section is removed because your symptom features are binary (0/1)
# and do not require Standardization.

print("\nData encoded and split successfully.")

# --- 3. Model Training and Evaluation ---
print("\n--- 3. Model Training and Evaluation ---")

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVM (C=1, Kernel=RBF)": SVC(random_state=42, probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    # Changed eval_metric for multi-class
}

results = {}

# Get all encoded labels (0, 1, 2, 3, 4, ...)
all_labels = le.transform(le.classes_)

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    acc = accuracy_score(y_test, y_pred)
    # The classification report now includes metrics for all encoded classes
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)  # Use all_labels for confusion matrix

    # FIX: Use 'labels=all_labels' to ensure the target names match the reported labels
    report = classification_report(y_test, y_pred, target_names=le.classes_, labels=all_labels, zero_division=0)

    results[name] = {
        'accuracy': acc,
        'cm': cm,
        'report': report
    }

    print(f"[{name}] Accuracy: {acc:.4f}")
    print(f"[{name}] Confusion Matrix:\n{cm}")

# --- 4. Summary and Visualization ---
print("\n--- 4. Summary of Model Performance ---")

best_accuracy = 0
best_model_name = ""
best_model = None

for name, res in results.items():
    current_model = models[name]
    print(f"\n=======================================================")
    print(f"MODEL: {name}")
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"Classification Report (Precision, Recall, F1-Score):\n{res['report']}")
    print(f"Confusion Matrix:\n{res['cm']}")
    print(f"=======================================================")

    # Determine the best model
    if res['accuracy'] >= best_accuracy:  # Use >= for stability in small samples
        best_accuracy = res['accuracy']
        best_model_name = name
        best_model = current_model  # Store the actual model object

print(
    f"\n🏆 The best performing model based on simple accuracy is: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Optional: Visualization of the best model's confusion matrix
plt.figure(figsize=(8, 6))
# Ensure the heatmap also uses the full set of class names for axes
sns.heatmap(results[best_model_name]['cm'], annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title(f'Confusion Matrix for {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
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
    Predicts the disease for a single new patient.

    Args:
        model: The trained scikit-learn model (the best_model).
        patient_data_dict (dict): A dictionary of the patient's features (0 or 1).
        feature_names (list): The list of features used in training (X.columns).
        label_encoder: The fitted LabelEncoder object.

    Returns:
        tuple: (Predicted Disease Name, Predicted Probability Vector)
    """

    # 1. Convert patient data dictionary into a DataFrame
    new_patient_df = pd.DataFrame([patient_data_dict], columns=feature_names)

    # 2. Make the prediction (returns an encoded label, e.g., 0, 1, 2...)
    encoded_prediction = model.predict(new_patient_df)[0]

    # 3. Decode the prediction back into the human-readable disease name
    predicted_disease = label_encoder.inverse_transform([encoded_prediction])[0]

    # 4. Get the probability vector (useful for multi-class tasks)
    probability_vector = model.predict_proba(new_patient_df)[0]

    return predicted_disease, probability_vector


# --- Example of New Patient Data ---
# Patient Profile:
# Assuming symptoms matching the dataset: itching, skin_rash, continuous_sneezing
new_patient_data = {
    'itching': 1,
    'skin_rash': 0,
    'nodal_skin_eruptions': 0,
    'continuous_sneezing': 1,
    # IMPORTANT: All 132 symptoms must be present here, even if set to 0.
    # We will use the columns from the training set for this example:
}

# Generate a sample patient dictionary where all features are 0, except for a couple of symptoms.
sample_input = {col: 0 for col in X.columns.tolist()}
# Set a few symptoms to 1 (e.g., trying to simulate Allergy or Fungal Infection)
sample_input['itching'] = 1
sample_input['skin_rash'] = 1
sample_input['continuous_sneezing'] = 0  # Example value

if best_model:
    final_disease, final_probabilities = predict_new_patient(
        model=best_model,
        patient_data_dict=sample_input,
        feature_names=X.columns.tolist(),
        label_encoder=le
    )

    # Find the confidence for the predicted class
    predicted_index = le.transform([final_disease])[0]
    confidence = final_probabilities[predicted_index]

    print(f"\n--- Prediction Results for New Patient (Using {best_model_name}) ---")
    print(f"Input Symptoms: {', '.join([k for k, v in sample_input.items() if v == 1])}")
    print(f"Final Prediction: {final_disease}")
    print(f"Confidence in Prediction: {confidence:.2%}")

else:
    print("\nError: The best model could not be determined or loaded.")
