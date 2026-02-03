import joblib
import os
import pandas as pd
import numpy as np

class PredictorService:
    def __init__(self):
        self.model_path = os.path.join('ml', 'models', 'rul_model.joblib')
        self.scaler_path = os.path.join('ml', 'models', 'scaler.joblib')
        self.model = None
        self.scaler = None
        self._load_models()

    def _load_models(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("Models loaded successfully.")
        else:
            print(f"Models not found at {self.model_path} or {self.scaler_path}")

    def predict(self, data: list):
        if self.model is None or self.scaler is None:
            return {"error": "Model or scaler not loaded. Please train the model first."}
        
        try:
            # Convert input list/dict to DataFrame
            df = pd.DataFrame(data)
            
            # Ensure we have the correct columns
            feature_cols = ['setting1', 'setting2', 'setting3'] + ['s' + str(i) for i in range(1, 22)]
            X = df[feature_cols]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            predictions = self.model.predict(X_scaled)
            
            return predictions.tolist()
        except Exception as e:
            return {"error": str(e)}

predictor_service = PredictorService()
