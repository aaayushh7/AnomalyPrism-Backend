from flask import Flask, request
from flask_cors import CORS
from flask import jsonify

def create_app():
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, 
         resources={r"/api/*": {
             "origins": ["https://anomaly-prism-frontend.vercel.app"],
             "methods": ["GET", "POST", "OPTIONS"],
             "allow_headers": ["Content-Type", "Accept"],
             "supports_credentials": True,
             "max_age": 3600
         }},
         expose_headers=["Content-Type", "X-CSRFToken"])
    
    # Additional CORS handling
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', 'https://anomaly-prism-frontend.vercel.app')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        # Handle pre-flight OPTIONS requests
        if request.method == 'OPTIONS':
            response.headers['Access-Control-Max-Age'] = '3600'
            response.status_code = 204
            return response
        return response

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def server_error(error):
        return jsonify({'error': 'Internal server error'}), 500

    # Register blueprint
    from .routes import api
    app.register_blueprint(api.bp)

    return app