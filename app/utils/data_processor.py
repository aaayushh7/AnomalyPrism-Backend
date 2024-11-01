import traceback
from typing import Any, Dict, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}

    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string to datetime object"""
        formats = [
            "%d/%m/%Y %H:%M",
            "%d-%m-%y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M"
        ]
        for fmt in formats:
            try:
                return pd.to_datetime(timestamp_str, format=fmt)
            except:
                continue
        try:
            return pd.to_datetime(timestamp_str, format='mixed')
        except:
            # If all else fails, try pandas default parser
            return pd.to_datetime(timestamp_str)

    def process_data(self, df):
        """Process and clean smart lock data for anomaly detection"""
        try:
            print("Starting data processing...")
            print(f"Input DataFrame shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            
            df_copy = df.copy()
            
            # Convert timestamp using custom parser
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'].apply(self.parse_timestamp))
            df_copy['hour'] = df_copy['timestamp'].dt.hour
            df_copy['minute'] = df_copy['timestamp'].dt.minute
            df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
            df_copy['is_weekend'] = df_copy['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Calculate usage frequency by device and time period
            df_copy['time_period'] = pd.cut(df_copy['hour'],
                                          bins=[0,6,12,18,24],
                                          labels=['Night','Morning','Afternoon','Evening'])
            
            # Create frequency features
            df_copy['device_activity_count'] = df_copy.groupby('device_id')['device_id'].transform('count')
            df_copy['location_activity_count'] = df_copy.groupby('locationId')['locationId'].transform('count')

            categorical_columns = ['lock_status', 'name', 'DeviceStatus',
                                 'manufacturerName', 'locationId', 'ownerId',
                                 'roomId', 'device_id']

            for column in categorical_columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    df_copy[f'{column}_encoded'] = self.label_encoders[column].fit_transform(df_copy[column])
                else:
                    try:
                        df_copy[f'{column}_encoded'] = self.label_encoders[column].transform(df_copy[column])
                    except ValueError:
                        print(f"Warning: New categories found in {column}. Treating them as anomalies.")
                        df_copy[f'{column}_encoded'] = -1

            features = [f'{col}_encoded' for col in categorical_columns] + [
                'hour', 'minute', 'day_of_week', 'is_weekend',
                'device_activity_count', 'location_activity_count'
            ]

            # Store the processed DataFrame for later use in prediction response
            self.processed_df = df_copy

            # Verify all features exist
            missing_features = [f for f in features if f not in df_copy.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            # Check for NaN values
            if df_copy[features].isna().any().any():
                raise ValueError("NaN values found in processed features")

            result = df_copy[features]
            print(f"Processed data shape: {result.shape}")
            return result

        except Exception as e:
            print(f"Error in process_data: {str(e)}")
            print(traceback.format_exc())
            raise Exception(f"Error processing data: {str(e)}")

    def format_prediction_response(self, df: pd.DataFrame, predictions: Union[np.ndarray, List]) -> Dict[str, Any]:
  
        try:
            # Ensure timestamp is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'].apply(self.parse_timestamp))
            
            # Convert predictions to list if needed
            pred_list = predictions.tolist() if isinstance(predictions, np.ndarray) else list(predictions)
            
            # Create a list of device data with all information
            devices = []
            for i, (_, row) in enumerate(df.iterrows()):
                device_info = {
                    'device_id': str(row['device_id']),
                    'prediction': int(pred_list[i]),
                    'location_id': str(row['locationId']),
                    'lock_status': str(row['lock_status']),
                    'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M')
                }
                devices.append(device_info)
            
            # Calculate summary statistics
            total_devices = len(devices)
            total_anomalies = sum(1 for device in devices if device['prediction'] == 1)
            
            # Group devices by location
            location_stats = {}
            for device in devices:
                loc_id = device['location_id']
                if loc_id not in location_stats:
                    location_stats[loc_id] = {'total': 0, 'anomalies': 0}
                location_stats[loc_id]['total'] += 1
                if device['prediction'] == 1:
                    location_stats[loc_id]['anomalies'] += 1
            
            return {
                'success': True,
                'data': {
                    'devices': devices,
                    'total_devices': total_devices,
                    'total_anomalies': total_anomalies,
                    'location_stats': [
                        {
                            'location_id': loc_id,
                            'total_devices': stats['total'],
                            'anomalies': stats['anomalies'],
                            'anomaly_percentage': round((stats['anomalies'] / stats['total']) * 100, 2)
                        }
                        for loc_id, stats in location_stats.items()
                    ],
                    'timestamps': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
                }
            }
            
        except Exception as e:
            print(f"Error in format_prediction_response: {str(e)}")
            print(traceback.format_exc())
            # Return a simplified response if there's an error
            return {
                'success': False,
                'error': str(e),
                'data': {
                    'devices': [
                        {
                            'device_id': str(device_id),
                            'prediction': int(pred),
                            'location_id': str(loc),
                            'lock_status': str(status),
                            'timestamp': str(ts)
                        }
                        for device_id, pred, loc, status, ts in zip(
                            df['device_id'].tolist(),
                            pred_list,
                            df['locationId'].tolist(),
                            df['lock_status'].tolist(),
                            df['timestamp'].astype(str).tolist()
                        )
                    ],
                    'total_devices': len(pred_list),
                    'total_anomalies': sum(1 for p in pred_list if p == 1)
                }
            }