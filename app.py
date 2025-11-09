from flask import Flask, request, render_template, jsonify  # ✅ add jsonify
import pickle
import numpy as np
import os

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Paths to model and scaler
# -----------------------------
MODEL_PATH = 'model/model.pkl'
SCALER_PATH = 'model/scaler.pkl'

# Check files
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found. Please train and save them first.")

# Load model and scaler
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

print("✅ Model and scaler loaded successfully.")

# -----------------------------
# Feature names (order must match training)
# -----------------------------
feature_names = [
    'Clump_Thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
    'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei',
    'Bland_Chromatin', 'Normal_Nucleoli'
]

# -----------------------------
# Home route
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

# -----------------------------
# Prediction route (HTML form + JSON API)
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # -----------------------------
        # 1️⃣ HTML form submission
        # -----------------------------
        if request.form:
            # Read features from form dynamically
            features = [float(request.form[f]) for f in feature_names]

        # -----------------------------
        # 2️⃣ JSON API submission
        # -----------------------------
        else:
            data = request.get_json(force=True)
            features = data.get('features', None)
            if features is None:
                return jsonify({'error': "Missing 'features' key in JSON input."}), 400

        # -----------------------------
        # Convert to array and scale
        # -----------------------------
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        # -----------------------------
        # Return HTML for form submission
        # -----------------------------
        if request.form:
            return render_template('index.html', feature_names=feature_names, prediction_text=f'Predicted Class: {result}')

        # -----------------------------
        # Return JSON for API call
        # -----------------------------
        return jsonify({
            'prediction': int(prediction),
            'result': result
        })

    except Exception as e:
        # Handle errors for both HTML and API
        if request.form:
            return render_template('index.html', feature_names=feature_names, prediction_text=f'Error: {str(e)}')
        else:
            return jsonify({'error': str(e)}), 500

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
from flask import Flask, request, render_template, jsonify  # ✅ add jsonify
import pickle   