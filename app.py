from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# --- LOAD YOUR SAVED MODEL ---
MODEL_PATH = os.path.join('Model', 'gradient_boosting_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    try:
        # Extract features from the HTML form
        acousticness = float(request.form['acousticness'])
        loudness = float(request.form['loudness'])
        tempo = float(request.form['tempo'])
        speechiness = float(request.form['speechiness'])
        valence = float(request.form['valence'])

        # Prepare input for prediction
        features = np.array([[acousticness, loudness, tempo, speechiness, valence]])
        prediction = model.predict(features)[0]
        result_text = f"Predicted Points: {round(prediction, 2)}"
    except Exception as e:
        result_text = f"An error occurred: {e}"

    return render_template('index.html', prediction_text=result_text)

if __name__ == "__main__":
    app.run(debug=True)
