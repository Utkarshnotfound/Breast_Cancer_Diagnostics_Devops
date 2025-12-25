import gradio as gr
import joblib
import numpy as np

# Load the model and scaler using Joblib
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names (matching your CSV columns)
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def predict_cancer(*args):
    # 1. Convert inputs to numpy array
    input_data = np.array(args).reshape(1, -1)
    
    # 2. Scale the data using the loaded scaler
    input_scaled = scaler.transform(input_data)
    
    # 3. Make prediction
    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    
    # 4. Return user-friendly results
    label = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"
    confidence = {"Benign": float(probs[0]), "Malignant": float(probs[1])}
    
    return label, confidence

# Build the Gradio Interface
ui = gr.Interface(
    fn=predict_cancer,
    inputs=[gr.Number(label=name) for name in feature_names],
    outputs=[gr.Textbox(label="Result"), gr.Label(label="Probabilities")],
    title="Breast Cancer Prediction Interface",
    description="Enter the clinical measurements to get a real-time diagnosis prediction.",
    theme="huggingface"
)

if __name__ == "__main__":
    ui.launch(share = "True")
