from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd

# Import the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Create the Flask app for the API
app = Flask(__name__)


@app.route("/api/predict", methods=['POST'])
def api_predict():
    # Expecting a JSON payload
    data = request.get_json(force=True)

    # Extract features from JSON
    features = pd.DataFrame([data['data']], columns=data['columns'])

    # Make predictions
    prediction = model.predict(features)

    # Return the prediction in JSON format
    return jsonify({'prediction': prediction[0]})


# Main function to run the API Flask app
if __name__ == "__main__":
    app.run(debug=True)
