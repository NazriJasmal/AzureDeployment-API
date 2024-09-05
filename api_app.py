from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd

# Import the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Create the Flask app for the API
app = Flask(__name__)

@app.route("/")
def home():
    # Root route for testing
    return "API is up and running!"

@app.route("/api/predict", methods=['POST'])
def api_predict():
    # Expecting a JSON payload
    data = request.get_json(force=True)

    # Extract features from JSON (assuming data['data'] is a list of values)
    features = pd.DataFrame([data['data']], columns=data['columns'])

    # Make predictions
    prediction = model.predict(features)

    # Return the prediction in JSON format
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
