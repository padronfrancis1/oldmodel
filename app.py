import joblib
import numpy as np
import logging
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the trained model
model = joblib.load("old_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Configure logging for Azure
logging.basicConfig(level=logging.INFO, filename="app.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Store model performance metrics
model_metrics = {
    "total_predictions": 0,
    "correct_predictions": 0
}

@app.route('/')
def home():
    return "Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Convert input to numpy array
        features = np.array(data["features"]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Simulate ground truth for monitoring (In production, store actual labels)
        actual_label = data.get("actual_label", prediction[0])  # Default to prediction if unknown

        # Update Accuracy Metrics
        model_metrics["total_predictions"] += 1
        if prediction[0] == actual_label:
            model_metrics["correct_predictions"] += 1
        
        # Compute accuracy dynamically
        accuracy = model_metrics["correct_predictions"] / model_metrics["total_predictions"]

        # Log model performance
        logging.info(f"Request: {data}, Prediction: {prediction[0]}, Actual: {actual_label}, Accuracy: {accuracy:.2f}")

        return jsonify({"prediction": int(prediction[0]), "accuracy": accuracy})

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
