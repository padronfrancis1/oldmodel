import joblib
import numpy as np
import logging
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the trained model
model = joblib.load("old_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Configure logging to integrate with Azure Log Analytics
logging.basicConfig(level=logging.INFO, filename="app.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Track model performance over time
model_metrics = {
    "total_predictions": 0,
    "correct_predictions": 0,
    "true_labels": [],
    "predicted_labels": []
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

        # Simulate ground truth (In production, replace with real labels)
        actual_label = data.get("actual_label", prediction[0])  # Defaults to prediction if unknown

        # Update Metrics
        model_metrics["total_predictions"] += 1
        model_metrics["true_labels"].append(actual_label)
        model_metrics["predicted_labels"].append(prediction[0])

        # Compute Metrics
        correct_predictions = sum(1 for p, a in zip(model_metrics["predicted_labels"], model_metrics["true_labels"]) if p == a)
        accuracy = correct_predictions / model_metrics["total_predictions"]

        # Calculate precision, recall, F1-score (requires at least 2 samples)
        if len(model_metrics["true_labels"]) > 1:
            precision = precision_score(model_metrics["true_labels"], model_metrics["predicted_labels"], average="binary", zero_division=1)
            recall = recall_score(model_metrics["true_labels"], model_metrics["predicted_labels"], average="binary", zero_division=1)
            f1 = f1_score(model_metrics["true_labels"], model_metrics["predicted_labels"], average="binary", zero_division=1)
        else:
            precision, recall, f1 = 1.0, 1.0, 1.0  # Default for first prediction

        # Log the request & results for Azure Log Analytics
        log_message = f"Request: {data}, Prediction: {prediction[0]}, Actual: {actual_label}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}"
        logging.info(log_message)

        return jsonify({
            "prediction": int(prediction[0]),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
