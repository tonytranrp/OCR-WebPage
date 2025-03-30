import socket
import cv2
import numpy as np
import threading
import json
import time
import base64
from typing import Optional, Tuple, Dict, Any
from queue import Queue
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from src.network.camera_server import CameraServer
from src.processing.pipeline import ProcessingPipeline
from src.utils.performance import PerformanceMonitor

class WebServer:
    """Web server that serves a camera client interface and processes video streams."""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 8080, 
                 camera_server_port: int = 8081,
                 enable_gpu: bool = True):
        self.host = host
        self.port = port
        self.camera_server_port = camera_server_port
        self.enable_gpu = enable_gpu
        
        # Initialize Flask app and SocketIO
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'bruh-ocr-secret-key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize camera server
        self.camera_server = CameraServer(
            host=host,
            port=camera_server_port,
            max_clients=10,
            enable_gpu=enable_gpu
        )
        
        # Initialize processing pipeline
        self.pipeline = ProcessingPipeline(
            resolution=(640, 480),
            ocr_languages=['en'],
            use_gpu=enable_gpu,
            detection_interval=3,
            buffer_size=3
        )
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        
        # Processing settings
        self.replace_text = False
        self.visualize_detections = True
        self.replacement_strategy = 'text'
        
        # Setup routes and socket events
        self._setup_routes()
        self._setup_socket_events()
        
        # Client tracking
        self.web_clients = {}
        self.processing_threads = {}
        self.stop_event = threading.Event()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/settings', methods=['GET', 'POST'])
        def settings():
            if request.method == 'POST':
                data = request.json
                if 'replace_text' in data:
                    self.replace_text = data['replace_text']
                if 'visualize_detections' in data:
                    self.visualize_detections = data['visualize_detections']
                if 'replacement_strategy' in data:
                    self.replacement_strategy = data['replacement_strategy']
                return jsonify({'status': 'success'})
            else:
                return jsonify({
                    'replace_text': self.replace_text,
                    'visualize_detections': self.visualize_detections,
                    'replacement_strategy': self.replacement_strategy
                })
    
    def _setup_socket_events(self):
        """Setup SocketIO event handlers."""
        @self.socketio.on('connect')
        def handle_connect():
            client_id = request.sid
            print(f"Web client connected: {client_id}")
            self.web_clients[client_id] = {
                'connected': True,
                'last_frame_time': time.time()
            }
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = request.sid
            if client_id in self.web_clients:
                print(f"Web client disconnected: {client_id}")
                self.web_clients[client_id]['connected'] = False
                del self.web_clients[client_id]
        
        @self.socketio.on('frame')
        def handle_frame(data):
            client_id = request.sid
            if client_id not in self.web_clients:
                return
            
            try:
                # Decode the base64 image
                image_data = data['image'].split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if frame is None or frame.size == 0:
                    return
                
                # Process the frame
                self.perf_monitor.start_timer('process')
                processed_frame, detections = self.pipeline.process_frame(frame)
                process_time = self.perf_monitor.stop_timer('process')
                
                # Apply text replacement if enabled
                if self.replace_text and detections:
                    processed_frame = self.pipeline.text_replacer.replace_all(
                        processed_frame, 
                        detections,
                        replacement_strategy=self.replacement_strategy,
                        replacement_text="[REDACTED]"
                    )
                
                # Visualize detections if enabled
                if self.visualize_detections and detections:
                    processed_frame = self.pipeline.text_detector.visualize_detections(processed_frame, detections)
                
                # Encode the processed frame to base64
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                processed_image = base64.b64encode(buffer).decode('utf-8')
                
                # Calculate performance metrics
                fps = 1.0 / (time.time() - self.web_clients[client_id]['last_frame_time'])
                self.web_clients[client_id]['last_frame_time'] = time.time()
                
                # Send the processed frame back to the client
                emit('processed_frame', {
                    'image': f"data:image/jpeg;base64,{processed_image}",
                    'fps': round(fps, 1),
                    'process_time_ms': round(process_time * 1000, 1),
                    'ocr_time_ms': round(self.pipeline.text_detector.get_processing_time() * 1000, 1),
                    'detection_count': len(detections) if detections else 0
                })
                
            except Exception as e:
                print(f"Error processing frame from web client {client_id}: {e}")
    
    def start(self):
        """Start the web server and camera server."""
        # Start the camera server
        if not self.camera_server.start():
            print("Failed to start camera server")
            return False
        
        # Start the processing pipeline
        if not self.pipeline.start():
            print("Failed to start processing pipeline")
            self.camera_server.stop()
            return False
        
        # Add pre-processing hook to enhance contrast for better OCR
        def enhance_for_ocr(frame):
            from src.utils.image_utils import enhance_contrast, enhance_text_readability
            from src.utils.performance import resize_for_performance
            # Resize for better performance
            resized = resize_for_performance(frame, target_width=640)
            # Enhance contrast and optimize for text detection
            enhanced = enhance_contrast(resized, clip_limit=3.0)
            # Further enhance text readability for better OCR, especially for paper documents
            return enhance_text_readability(enhanced, sharpen=True, denoise=True)
        
        self.pipeline.add_pre_processing_hook(enhance_for_ocr)
        
        # Add post-processing hook to draw custom overlays
        def draw_custom_overlays(frame, detections):
            from src.drawing.drawing_utils import DrawingUtils
            drawing = DrawingUtils()
            
            # Add a timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            frame = drawing.draw_text(frame, timestamp, (10, frame.shape[0] - 20), 
                                    font_scale=0.5, color=(255, 255, 255))
            
            # Draw a semi-transparent header bar
            frame = drawing.draw_filled_rectangle(
                frame, 
                (0, 0), 
                (frame.shape[1], 50), 
                color=(0, 0, 100), 
                alpha=0.7
            )
            
            # Add title text
            frame = drawing.draw_text(
                frame,
                "Web OCR Processing",
                (frame.shape[1] // 2 - 100, 30),
                font_scale=0.7,
                color=(255, 255, 255)
            )
            
            # Draw detection count if any
            if detections:
                count_text = f"Detected {len(detections)} text regions"
                frame = drawing.draw_text(
                    frame,
                    count_text,
                    (10, 70),
                    font_scale=0.6,
                    color=(0, 255, 0)
                )
            
            return frame
        
        self.pipeline.add_post_processing_hook(draw_custom_overlays)
        
        print(f"Starting web server on http://{self.host}:{self.port}")
        print(f"Camera server running on port {self.camera_server_port}")
        
        # Start the Flask-SocketIO server
        try:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False, allow_unsafe_werkzeug=True)
            return True
        except Exception as e:
            print(f"Failed to start web server: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the web server and camera server."""
        # Stop the pipeline
        try:
            self.pipeline.stop()
        except:
            pass
        
        # Stop the camera server
        try:
            self.camera_server.stop()
        except:
            pass
        
        # Set stop event for all threads
        self.stop_event.set()
        
        print("Web server stopped")

def main():
    import argparse
    import socket
    parser = argparse.ArgumentParser(description='Web Camera Server')
    parser.add_argument('--host', default='127.0.0.1',
                      help='Host address to bind to')
    parser.add_argument('--port', type=int, default=5000,
                      help='Web server port number (default: 5000)')
    parser.add_argument('--camera-port', type=int, default=8000,
                      help='Camera server port number (default: 8000)')
    parser.add_argument('--no-gpu', action='store_true',
                      help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Try to find available ports if the defaults are in use
    def find_available_port(start_port, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                # Test if port is available
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.bind(('127.0.0.1', port))
                test_socket.close()
                return port
            except OSError:
                continue
        return None
    
    # Check if specified ports are available, find alternatives if not
    if args.port == args.camera_port:
        print("Warning: Web server and camera server cannot use the same port")
        args.camera_port = args.port + 1
    
    # Test and find available web server port
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.bind((args.host, args.port))
        test_socket.close()
    except OSError:
        print(f"Port {args.port} is in use, finding an available port...")
        available_port = find_available_port(args.port + 1)
        if available_port:
            print(f"Using port {available_port} for web server")
            args.port = available_port
        else:
            print("Could not find an available port for web server")
            return
    
    # Test and find available camera server port
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.bind((args.host, args.camera_port))
        test_socket.close()
    except OSError:
        print(f"Port {args.camera_port} is in use, finding an available port...")
        available_port = find_available_port(args.camera_port + 1)
        if available_port:
            print(f"Using port {available_port} for camera server")
            args.camera_port = available_port
        else:
            print("Could not find an available port for camera server")
            return
    
    server = WebServer(
        host=args.host,
        port=args.port,
        camera_server_port=args.camera_port,
        enable_gpu=not args.no_gpu
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.stop()

if __name__ == "__main__":
    main()