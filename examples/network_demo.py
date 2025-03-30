import cv2
import numpy as np
import time
import os
import sys
import threading

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.camera_server import CameraServer
from src.network.camera_client import CameraClient
from src.processing.pipeline import ProcessingPipeline
from src.utils.performance import PerformanceMonitor

def run_server(port=8000):
    """Run the camera processing server."""
    print("Starting Camera Processing Server")
    print(f"Listening on port {port}")
    print("Press 'q' to quit, 'r' to toggle text replacement, 'd' to toggle detection visualization")
    
    # Initialize the server with GPU acceleration
    try:
        server = CameraServer(
            host='0.0.0.0',
            port=port,
            max_clients=5,
            enable_gpu=True
        )
    except Exception as e:
        print(f"Failed to initialize server: {e}")
        print("Try using a different port with --port argument")
        return
    
    # Initialize the processing pipeline
    pipeline = ProcessingPipeline(
        resolution=(640, 480),
        ocr_languages=['en'],
        use_gpu=True,  # Enable GPU acceleration
        detection_interval=3,  # Detect text every 3 frames for better performance
        buffer_size=3  # Smaller buffer for reduced stuttering
    )
    
    # Add a pre-processing hook to enhance contrast for better OCR
    def enhance_for_ocr(frame):
        from src.utils.image_utils import enhance_contrast
        from src.utils.performance import resize_for_performance
        # Resize for better performance
        resized = resize_for_performance(frame, target_width=640)
        # Enhance contrast to improve text detection
        return enhance_contrast(resized, clip_limit=2.0)
    
    pipeline.add_pre_processing_hook(enhance_for_ocr)
    
    # Add a post-processing hook to draw custom overlays
    def draw_custom_overlays(frame, detections):
        from src.drawing.drawing_utils import DrawingUtils
        drawing = DrawingUtils()
        
        # Add a timestamp
        import time
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
            "Network OCR Processing",
            (frame.shape[1] // 2 - 120, 30),
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
    
    pipeline.add_post_processing_hook(draw_custom_overlays)
    
    # Start the server
    if not server.start():
        print("Failed to start server")
        return
    
    # Start the pipeline
    if not pipeline.start():
        print("Failed to start pipeline")
        server.stop()
        return
    
    # Initialize visualization flags
    replace_text = False
    visualize_detections = True
    replacement_strategy = 'text'
    perf_monitor = PerformanceMonitor()
    
    try:
        while True:
            perf_monitor.start_timer('frame')
            
            # Get client information
            clients = server.get_client_info()
            
            # Process frames from each client
            for client_id in clients:
                success, frame = server.get_frame(client_id)
                if success:
                    # Process the frame through our pipeline
                    perf_monitor.start_timer('process')
                    processed_frame, detections = pipeline.process_frame(frame)
                    process_time = perf_monitor.stop_timer('process')
                    
                    # Apply text replacement if enabled
                    if replace_text and detections:
                        processed_frame = pipeline.text_replacer.replace_all(
                            processed_frame, 
                            detections,
                            replacement_strategy=replacement_strategy,
                            replacement_text="[REDACTED]"
                        )
                    
                    # Visualize detections if enabled
                    if visualize_detections and detections:
                        processed_frame = pipeline.text_detector.visualize_detections(processed_frame, detections)
                    
                    # Display performance metrics
                    fps = 1.0 / perf_monitor.stop_timer('frame') if perf_monitor.get_average_time('frame') > 0 else 0
                    metrics_text = f"FPS: {fps:.1f} | Process: {process_time*1000:.1f}ms | OCR: {pipeline.text_detector.get_processing_time()*1000:.1f}ms"
                    cv2.putText(processed_frame, metrics_text, (10, frame.shape[0] - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Display the processed frame
                    window_name = f"Client: {client_id}"
                    cv2.imshow(window_name, processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                replace_text = not replace_text
                print(f"Text replacement: {'ON' if replace_text else 'OFF'}")
            elif key == ord('d'):
                visualize_detections = not visualize_detections
                print(f"Detection visualization: {'ON' if visualize_detections else 'OFF'}")
            elif key == ord('1'):
                replacement_strategy = 'text'
                print("Replacement strategy: TEXT")
            elif key == ord('2'):
                replacement_strategy = 'blur'
                print("Replacement strategy: BLUR")
            elif key == ord('c'):
                # Capture the current frame at high resolution
                high_res_frame = cv2.resize(frame, (1280, 720))
                cv2.imshow('Captured Frame', high_res_frame)
                # Perform advanced OCR analysis
                advanced_detections = pipeline.text_detector.detect(high_res_frame, tight_bbox=True)
                # Display OCR results in a UI next to the image
                # (Assuming a function display_ocr_results exists)
                display_ocr_results(high_res_frame, advanced_detections)
    
    finally:
        pipeline.stop()
        server.stop()
        cv2.destroyAllWindows()
        print("Server stopped")

def run_client(server_host: str, server_port=8000):
    """Run a camera client that streams to the server."""
    print(f"Starting Camera Client - Connecting to {server_host}:{server_port}")
    
    # Initialize the client with GPU acceleration
    try:
        client = CameraClient(
            server_host=server_host,
            server_port=server_port,
            camera_source=0,  # Use default camera
            resolution=(640, 480),
            enable_gpu=True
        )
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        print("Make sure the server is running and the port is correct")
        return
    
    # Connect to server
    if not client.connect():
        print("Failed to connect to server")
        return
    
    try:
        # Stream camera feed
        client.stream()
    except KeyboardInterrupt:
        print("\nStopping client...")
    finally:
        client.disconnect()
        print("Client stopped")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Network Camera Demo')
    parser.add_argument('--mode', choices=['server', 'client'], required=True,
                      help='Run as server or client')
    parser.add_argument('--host', default='localhost',
                      help='Server host address (for client mode)')
    parser.add_argument('--port', type=int, default=8000,
                      help='Server port number (default: 8000)')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        run_server(args.port)
    else:
        run_client(args.host, args.port)

if __name__ == "__main__":
    main()