from flask import Blueprint, request, jsonify
from ..models.anomaly_detector import SmartLockAnomalyDetector
from ..utils.data_processor import DataProcessor
import pandas as pd
from werkzeug.utils import secure_filename
import os
import traceback

bp = Blueprint('api', __name__, url_prefix='/api')
detector = SmartLockAnomalyDetector()
data_processor = DataProcessor()  # Create an instance of DataProcessor

def validate_csv_file(file):
    """Validate the uploaded CSV file"""
    if not file or file.filename == '':
        raise ValueError('No file selected')
    
    if not file.filename.endswith('.csv'):
        raise ValueError('Only CSV files are allowed')
    
    return True

def safe_read_csv(file):
    """Safely read CSV file"""
    try:
        # Read the file contents for preview
        content_preview = file.read(200).decode('utf-8')  # Read first 200 bytes for preview
        print("File content preview:", content_preview)  # Debug log
        
        # Reset file pointer to the beginning
        file.seek(0)
        
        # Read the CSV file
        df = pd.read_csv(file)
        print(f"Created DataFrame with shape: {df.shape}")  # Debug log
        
        # Basic validation
        if df.empty:
            raise ValueError("Empty CSV file")
            
        required_columns = ['device_id', 'lock_status', 'timestamp', 'name', 
                          'DeviceStatus', 'manufacturerName', 'locationId', 
                          'ownerId', 'roomId']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
        print("DataFrame columns:", df.columns.tolist())  # Debug log
        return df
        
    except pd.errors.EmptyDataError:
        print("Error: Empty CSV file")
        raise ValueError('CSV file is empty')
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise ValueError(f'Error reading file: {str(e)}')

@bp.route('/train', methods=['POST'])
def train():
    try:
        print("Received training request")  # Debug log
        
        if 'file' not in request.files:
            print("No file in request")  # Debug log
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"Received file: {file.filename}")  # Debug log
        
        try:
            validate_csv_file(file)
            print("File validation passed")  # Debug log
            
            df = safe_read_csv(file)
            print(f"Data shape before processing: {df.shape}")  # Debug log
            
            processed_data = data_processor.process_data(df)  # Use the instance
            print(f"Data shape after processing: {processed_data.shape if processed_data is not None else 'None'}")  # Debug log
            
            if processed_data is None or len(processed_data) == 0:
                print("Data processing failed - empty result")  # Debug log
                return jsonify({'error': 'Data processing failed - no valid data after processing'}), 400
            
            result = detector.train(processed_data)
            print(f"Training result: {result}")  # Debug log
            
            if result.get('status') == 'error':
                return jsonify({'error': result['message']}), 500
            
            return jsonify({
                'success': True,
                'message': result['message'],
                'details': {
                    'data_shape': result.get('data_shape'),
                    'status': result['status']
                }
            })

        except ValueError as e:
            print(f"Validation error: {str(e)}")  # Debug log
            return jsonify({'error': str(e)}), 400
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return jsonify({'error': f'Internal server error during training: {str(e)}'}), 500

@bp.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        try:
            validate_csv_file(file)
            df = safe_read_csv(file)
            
            processed_data = data_processor.process_data(df)
            
            if processed_data is None or processed_data.empty:
                return jsonify({'error': 'Data processing failed'}), 400
            
            predictions = detector.predict(processed_data)
            
            response_data = data_processor.format_prediction_response(df, predictions)
            
            # Check if the response was successful
            if not response_data['success']:
                return jsonify({'error': 'Error formatting prediction response'}), 500

            # Access the nested 'data' key
            return jsonify({
                'success': True,
                'data': {
                    'devices': response_data['data']['devices'],
                    'total_devices': response_data['data']['total_devices'],
                    'total_anomalies': response_data['data']['total_anomalies'],
                    'location_stats': response_data['data'].get('location_stats', []),
                    'timestamps': response_data['data'].get('timestamps', [])
                }
            })

        except ValueError as e:
            return jsonify({'error': str(e)}), 400
            
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error during prediction'}), 500
    
@bp.route('/predict-single', methods=['POST'])
def predict_single():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        try:
            df = pd.DataFrame([data])
            processed_data = data_processor.process_data(df)  # Use the instance
            
            if processed_data is None or processed_data.empty:
                return jsonify({'error': 'Data processing failed'}), 400
            
            prediction = detector.predict(processed_data)
            
            return jsonify({
                'success': True,
                'prediction': prediction[0],
                'is_anomaly': bool(prediction[0])
            })

        except ValueError as e:
            return jsonify({'error': str(e)}), 400
            
    except Exception as e:
        print(f"Single prediction error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error during prediction'}), 500

@bp.route('/train', methods=['OPTIONS'])
@bp.route('/predict', methods=['OPTIONS'])
@bp.route('/predict-single', methods=['OPTIONS'])
def handle_options():
    return '', 204