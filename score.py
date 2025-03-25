import json
import joblib
import numpy as np
import os

def init():
    global model
    # Load model from file
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.joblib')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_data = np.array(data['data'])
        result = model.predict(input_data)
        return result.tolist()
    except Exception as e:
        return str(e)
