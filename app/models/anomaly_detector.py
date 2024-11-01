import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
import os
from pathlib import Path

class SmartLockAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        # Create a models directory in the application root
        self.model_dir = Path(__file__).parent.parent / 'models'
        self.model_path = self.model_dir / 'model.pkl'
        
        # Ensure the models directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self, data):
        """Train the anomaly detection model"""
        try:
            # Validate input data
            if data is None or len(data) == 0:
                raise ValueError("Training data is empty or None")
            
            print(f"Training with data shape: {data.shape}")  # Debug log
            
            # Convert data to numpy array if it's not already
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Fit the model
            self.model.fit(data)
            self.is_trained = True
            
            # Save the model
            self._save_model()
            
            return {
                "status": "success", 
                "message": "Model trained successfully",
                "data_shape": data.shape
            }
            
        except Exception as e:
            print(f"Training error: {str(e)}")  # Debug log
            return {"status": "error", "message": f"Training failed: {str(e)}"}

    def predict(self, data):
        """Make predictions on input data"""
        try:
            if not self.is_trained:
                if not self._load_model():
                    raise ValueError("No trained model available")
            
            # Validate input data
            if data is None or len(data) == 0:
                raise ValueError("Prediction data is empty or None")
            
            # Convert data to numpy array if it's not already
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            print(f"Predicting with data shape: {data.shape}")  # Debug log
            
            predictions = self.model.predict(data)
            # Convert predictions to binary (1 for normal, 0 for anomaly)
            predictions = np.where(predictions == 1, 0, 1)
            
            # Return numpy array instead of list
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")  # Debug log
            raise ValueError(f"Prediction failed: {str(e)}")

    def _save_model(self):
        """Save the trained model"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {self.model_path}")  # Debug log
        except Exception as e:
            print(f"Error saving model: {str(e)}")  # Debug log
            raise

    def _load_model(self):
        """Load the trained model if it exists"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    self.is_trained = True
                print("Model loaded successfully")  # Debug log
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")  # Debug log
            return False