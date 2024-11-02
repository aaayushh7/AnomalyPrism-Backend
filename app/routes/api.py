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
        print("Received training request")
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"Received file: {file.filename}")
        
        try:
            validate_csv_file(file)
            print("File validation passed")
            
            df = safe_read_csv(file)
            print(f"Data shape before processing: {df.shape}")
            
            processed_data = data_processor.process_data(df)
            print(f"Data shape after processing: {processed_data.shape if processed_data is not None else 'None'}")
            
            if processed_data is None or len(processed_data) == 0:
                print("Data processing failed - empty result")
                return jsonify({'error': 'Data processing failed - no valid data after processing'}), 400
            
            # Train the model and get results
            result = detector.train(processed_data)
            print(f"Training result: {result}")
            
            if result.get('status') == 'error':
                return jsonify({'error': result['message']}), 500
            
            # Prepare visualization data from the processed data
            viz_data = detector.prepare_visualization_data(processed_data)
            
            # Return success response with visualization data
            return jsonify({
                'status': 'success',
                'message': result['message'],
                'visualization_data': viz_data,
                'details': {
                    'data_shape': result.get('data_shape'),
                    'model_status': result['status']
                }
            })

        except ValueError as e:
            print(f"Validation error: {str(e)}")
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

            # Generate visualization data from response
            viz_data = detector.prepare_prediction_visualization2(response_data)
            
            # Add visualization data to response
            response_data['data']['visualization_data'] = viz_data

            return jsonify(response_data)

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
        print("Received data:", data)  # Log the received data
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        try:
            df = pd.DataFrame([data])
            print("DataFrame columns:", df.columns.tolist())  # Log DataFrame columns
            
            processed_data = data_processor.process_data(df)
            print("Processed data:", processed_data)  # Log processed data
            
            if processed_data is None or processed_data.empty:
                return jsonify({'error': 'Data processing failed'}), 400
            
            prediction = detector.predict(processed_data)
            
            # Convert NumPy types to Python native types
            prediction_value = int(prediction[0]) if hasattr(prediction[0], 'item') else prediction[0]
            
            return jsonify({
                'success': True,
                'prediction': prediction_value,
                'is_anomaly': bool(prediction_value)
            })

        except ValueError as e:
            print(f"Validation error: {str(e)}")
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