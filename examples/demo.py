import cv2
import numpy as np
import time
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capture.video_capture import VideoCapture
from src.ocr.text_detector import TextDetector
from src.ocr.text_replacer import TextReplacer
from src.drawing.drawing_utils import DrawingUtils
from src.processing.pipeline import ProcessingPipeline
from src.utils.performance import PerformanceMonitor, resize_for_performance
from src.utils.image_utils import enhance_contrast

def main():
    print("Starting Video Processing Demo with OCR")
    print("Press 'q' to quit, 'r' to toggle text replacement, 'd' to toggle detection visualization")
    
    # Initialize the processing pipeline
    pipeline = ProcessingPipeline(
        video_source=0,  # Use default camera
        resolution=(640, 480),
        ocr_languages=['en'],
        use_gpu=True,  # Set to True if you have GPU support
        detection_interval=1  # Detect text every 10 frames for better performance
    )
    
    # Create a performance monitor
    perf_monitor = PerformanceMonitor()
    
    # Initialize drawing utils for custom overlays
    drawing = DrawingUtils()
    
    # Add a pre-processing hook to enhance contrast for better OCR
    def enhance_for_ocr(frame):
        # Resize for better performance
        resized = resize_for_performance(frame, target_width=640)
        # Enhance contrast to improve text detection
        return enhance_contrast(resized, clip_limit=2.0)
    
    pipeline.add_pre_processing_hook(enhance_for_ocr)
    
    # Add a post-processing hook to draw custom overlays
    def draw_custom_overlays(frame, detections):
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
            "Real-time Video Processing with OCR",
            (frame.shape[1] // 2 - 150, 30),
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
    
    # Start the pipeline
    if not pipeline.start():
        print("Failed to start the pipeline")
        return
    
    # Run the main loop
    replace_text = True
    visualize_detections = True
    replacement_strategy = 'text'
    
    try:
        while True:
            perf_monitor.start_timer('frame')
            
            # Read a frame
            success, frame = pipeline.capture.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Process the frame
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
            
            # Display the frame
            cv2.imshow('Video Processing Demo', processed_frame)
            
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
    
    finally:
        # Clean up
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Demo finished")

if __name__ == "__main__":
    main()