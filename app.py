from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# --------------------------------------------------
# Initialize Flask app
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Paths to model and scaler
# --------------------------------------------------
MODEL_PATH = 'model/model.pkl'
SCALER_PATH = 'model/scaler.pkl'

# Check if files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found. Please train and save them first.")

# Load model
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Load scaler
with open(SCALER_PATH, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

print("✅ Model and scaler loaded successfully.")

# --------------------------------------------------
# Home route (renders HTML form)
# --------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# --------------------------------------------------
# Predict route (supports HTML form and JSON API)
# --------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- HTML form input ---
        if request.form:
            # Make sure feature names match training data
            features = [
                float(request.form['Clump_Thickness']),
                float(request.form['Uniformity_Cell_Size']),
                float(request.form['Uniformity_Cell_Shape']),
                float(request.form['Marginal_Adhesion']),
                float(request.form['Single_Epithelial_Cell_Size']),
                float(request.form['Bare_Nuclei']),
                float(request.form['Bland_Chromatin']),
                float(request.form['Normal_Nucleoli'])
            ]

        # --- JSON API input ---
        else:
            data = request.get_json(force=True)
            features = data.get('features')
            if features is None:
                return jsonify({'error': "Missing 'features' key in JSON input."}), 400

        # Convert to array and scale
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        # Return prediction in HTML if form submission
        if request.form:
            return render_template('index.html', prediction_text=f'Predicted Class: {result}')

        # Return JSON if API
        return jsonify({'prediction': int(prediction), 'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------------------------------------
# Run the Flask app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

