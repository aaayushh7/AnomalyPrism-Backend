import traceback
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
import os
from pathlib import Path
import pandas as pd
from ..utils.data_processor import DataProcessor


class SmartLockAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        self.data_processor = DataProcessor()  # Initialize DataProcessor
        self.label_encoders = {}  # Initialize label_encoders
        
        # Create a models directory in the application root
        self.model_dir = Path(__file__).parent.parent / 'models'
        self.model_path = self.model_dir / 'model.pkl'
        
        # Ensure the models directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def prepare_prediction_visualization2(self, prediction_data):
        """
        Prepare visualization data from existing prediction response format
        
        Args:
            prediction_data: Dictionary containing prediction response data
        Returns:
            dict with visualization data
        """
        try:
            # Convert the devices list to a DataFrame for processing
            devices_data = prediction_data['data']['devices']
            df = pd.DataFrame(devices_data)
            
            # 1. Time-based analysis - using existing timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_analysis = df.groupby([df['timestamp'].dt.strftime('%Y-%m-%d %H:00')])['prediction'].agg({
                'total': 'count',
                'anomalies': lambda x: (x == 1).sum()
            }).reset_index()
            
            time_series = time_analysis.apply(lambda x: {
                'timestamp': x['timestamp'],
                'total': int(x['total']),
                'anomalies': int(x['anomalies']),
                'anomaly_rate': round((x['anomalies'] / x['total'] * 100), 2) if x['total'] > 0 else 0
            }, axis=1).tolist()
            
            # 2. Location analysis - using existing location_stats
            location_stats = prediction_data['data'].get('location_stats', [])
            
            # 3. Device summary - using existing devices data
            device_summary = (df.groupby('device_id')
                .agg({
                    'prediction': ['count', lambda x: (x == 1).sum()],
                    'timestamp': ['min', 'max']
                })
                .reset_index()
                .head(100)  # Limit to top 100 devices
            )
            
            device_stats = device_summary.apply(lambda x: {
                'device_id': x['device_id'],
                'total_events': int(x[('prediction', 'count')]),
                'anomaly_count': int(x[('prediction', '<lambda_0>')]),
                'first_seen': x[('timestamp', 'min')].strftime('%Y-%m-%d %H:%M'),
                'last_seen': x[('timestamp', 'max')].strftime('%Y-%m-%d %H:%M'),
                'anomaly_rate': round(
                    x[('prediction', '<lambda_0>')] / x[('prediction', 'count')] * 100, 2
                ) if x[('prediction', 'count')] > 0 else 0
            }, axis=1).tolist()
            
            return {
                'summary': {
                    'total_devices': prediction_data['data']['total_devices'],
                    'total_anomalies': prediction_data['data']['total_anomalies'],
                    'anomaly_rate': round(
                        (prediction_data['data']['total_anomalies'] / len(devices_data) * 100), 2
                    ) if devices_data else 0
                },
                'time_series': time_series,
                'location_stats': location_stats,
                'device_stats': device_stats
            }
            
        except Exception as e:
            print(f"Error preparing visualization data: {str(e)}")
            return None

    def prepare_visualization_data(self, df, predictions=None):
        """
        Prepare data for frontend visualization with support for encoded data
        
        Args:
            df (pd.DataFrame): Input DataFrame with encoded features
            predictions (array-like, optional): Anomaly predictions (1 for anomaly, 0 for normal)
            
        Returns:
            dict: Visualization data structure
        """
        df_copy = df.copy()
        
        # Save original data columns before processing
        original_columns = list(df_copy.columns)
        print(f"Original columns: {original_columns}")

        # Handle temporal features
        temporal_features = {
            'hour': {'range': (0, 23), 'default': 0},
            'day_of_week': {'range': (0, 6), 'default': 0}
        }
        
        for feature, params in temporal_features.items():
            if feature not in df_copy.columns:
                if any(col.startswith(feature) for col in original_columns):
                    df_copy[feature] = df_copy[next(col for col in original_columns if col.startswith(feature))]
                else:
                    print(f"Warning: Creating default {feature} values from timestamp")
                    if 'timestamp' in df_copy.columns:
                        timestamps = pd.to_datetime(df_copy['timestamp'])
                        if feature == 'hour':
                            df_copy[feature] = timestamps.dt.hour
                        else:  # day_of_week
                            df_copy[feature] = timestamps.dt.dayofweek
                    else:
                        df_copy[feature] = params['default']
            
            # Ensure values are within valid range
            df_copy[feature] = df_copy[feature].clip(*params['range'])

        # Create time_period
        df_copy['time_period'] = pd.cut(
            df_copy['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )

        # Handle DeviceStatus
        device_status_cols = ['DeviceStatus_encoded', 'DeviceStatus']
        device_status_col = next((col for col in device_status_cols if col in df_copy.columns), None)

        # For training data, we'll consider all points as normal if predictions aren't provided
        if predictions is None:
            df_copy['is_anomaly'] = 0
        else:
            # Convert IsolationForest predictions (-1 for anomaly, 1 for normal) to binary (1 for anomaly, 0 for normal)
            df_copy['is_anomaly'] = np.where(predictions == -1, 1, 0)

        # Initialize visualization data structure
        viz_data = {
            'hourly_activity': {
                'labels': list(range(24)),
                'normal': [0] * 24,
                'anomaly': [0] * 24
            },
            'device_status': {
                'labels': [],
                'normal': [],
                'anomaly': []
            },
            'weekly_pattern': {
                'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'normal': [0] * 7,
                'anomaly': [0] * 7
            },
            'time_periods': {
                'labels': ['Night', 'Morning', 'Afternoon', 'Evening'],
                'data': [0] * 4
            }
        }

        try:
            # Populate hourly activity data
            for is_anomaly in [0, 1]:
                mask = df_copy['is_anomaly'] == is_anomaly
                hourly_counts = df_copy[mask]['hour'].value_counts()
                data_key = 'anomaly' if is_anomaly else 'normal'
                viz_data['hourly_activity'][data_key] = [
                    int(hourly_counts.get(hour, 0)) for hour in range(24)
                ]

            # Populate device status data
            if device_status_col:
                unique_statuses = sorted(df_copy[device_status_col].unique())
                viz_data['device_status']['labels'] = [str(status) for status in unique_statuses]
                
                for is_anomaly in [0, 1]:
                    mask = df_copy['is_anomaly'] == is_anomaly
                    status_counts = df_copy[mask][device_status_col].value_counts()
                    data_key = 'anomaly' if is_anomaly else 'normal'
                    viz_data['device_status'][data_key] = [
                        int(status_counts.get(status, 0)) for status in unique_statuses
                    ]

            # Populate weekly pattern data
            for is_anomaly in [0, 1]:
                mask = df_copy['is_anomaly'] == is_anomaly
                weekly_counts = df_copy[mask]['day_of_week'].value_counts()
                data_key = 'anomaly' if is_anomaly else 'normal'
                viz_data['weekly_pattern'][data_key] = [
                    int(weekly_counts.get(day, 0)) for day in range(7)
                ]

            # Populate time periods data
            time_period_counts = df_copy.groupby(['time_period', 'is_anomaly']).size().unstack(fill_value=0)
            viz_data['time_periods']['data'] = [
                int(time_period_counts.get((period, 0), 0) + time_period_counts.get((period, 1), 0))
                for period in viz_data['time_periods']['labels']
            ]

        except Exception as e:
            print(f"Error preparing visualization data: {str(e)}")
            return viz_data

        return viz_data

    def train(self, data):
        """
        Train the anomaly detection model
        
        Args:
            data (pd.DataFrame): Input training data
            
        Returns:
            dict: Training results including status and visualization data
        """
        try:
            print("Starting model training")
            
            if data is None or len(data) == 0:
                raise ValueError("Training data is empty or None")
                
            print(f"Training data shape: {data.shape}")
            print(f"Input columns: {data.columns.tolist()}")
            
            # Store original data for visualization
            self.processed_df = data.copy()
            
            # Extract features for training
            try:
                training_features = self.data_processor.preprocess_data(data)
                print(f"Training features shape: {training_features.shape}")
                print(f"Processed columns: {training_features.columns.tolist()}")
            except Exception as e:
                raise Exception(f"Feature processing error: {str(e)}")
            
            # Fit the model and get initial predictions
            try:
                print("Starting model fitting")
                # Initialize IsolationForest with better parameters
                self.model = IsolationForest(
                    contamination=0.1,  # Expected proportion of anomalies
                    n_estimators=100,   # More trees for better accuracy
                    max_samples='auto',
                    max_features=1.0,   # Use all features
                    bootstrap=False,    # Don't use bootstrapping for deterministic results
                    n_jobs=-1,         # Use all available cores
                    random_state=42,
                    verbose=0
                )
                
                # Fit the model
                self.model.fit(training_features)
                
                # Get initial predictions (-1 for anomaly, 1 for normal)
                initial_predictions = self.model.predict(training_features)
                
                self.is_trained = True
                print("Model fitting completed successfully")
                
                # Calculate anomaly scores for better understanding
                anomaly_scores = -self.model.score_samples(training_features)
                anomaly_threshold = np.percentile(anomaly_scores, 90)  # 90th percentile
                print(f"Anomaly score threshold: {anomaly_threshold:.3f}")
                
            except Exception as e:
                raise Exception(f"Model fitting failed: {str(e)}")
            
            # Prepare visualization data using the original data and initial predictions
            try:
                viz_data = self.prepare_visualization_data(
                    self.processed_df,
                    predictions=initial_predictions
                )
                print("Visualization data prepared successfully")
            except Exception as e:
                raise Exception(f"Visualization preparation failed: {str(e)}")
            
            # Calculate training statistics
            anomaly_count = np.sum(initial_predictions == -1)
            total_samples = len(initial_predictions)
            anomaly_rate = (anomaly_count / total_samples) * 100
            
            result = {
                "status": "success",
                "message": "Model trained successfully",
                "details": {
                    "data_shape": training_features.shape,
                    "model_status": "success",
                    "anomalies_detected": int(anomaly_count),
                    "total_samples": total_samples,
                    "anomaly_rate": f"{anomaly_rate:.2f}%",
                    
                },
                "visualization_data": viz_data
            }
            
            print("\nTraining Summary:")
            print(f"Total samples processed: {total_samples:,}")
            print(f"Anomalies detected: {anomaly_count:,} ({anomaly_rate:.2f}%)")
            # print(f"Number of features used: {len(self.feature_columns)}")
            print("Training completed successfully")
            
            return result
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Training failed: {str(e)}"
            }
                
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
            
            return predictions.tolist()
            
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