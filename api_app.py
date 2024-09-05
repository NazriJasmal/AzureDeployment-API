from flask import Flask, request, jsonify
import pandas as pd
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Create the Flask app for the API
app = Flask(__name__)

@app.route("/")
def home():
    return "API is up and running!"

@app.route("/api/predict", methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        logging.info(f"Received data: {data}")

        # Extract features from JSON
        if 'data' not in data or 'columns' not in data:
            raise ValueError("JSON must contain 'data' and 'columns' keys")

        # Create a DataFrame with the incoming data
        features = pd.DataFrame(data['data'], columns=data['columns'])
        logging.info(f"Features DataFrame: {features}")

        # Check if the columns match what the model was trained with
        expected_columns = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
        if list(features.columns) != expected_columns:
            raise ValueError(f"Expected columns: {expected_columns}, but got: {list(features.columns)}")

        # Make predictions
        prediction = model.predict(features)
        logging.info(f"Prediction: {prediction}")

        # Return the prediction in JSON format
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
