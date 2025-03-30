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
from src.utils.performance import PerformanceMonitor
from src.utils.image_utils import resize_image

def main():
    print("Starting Text Replacement Demo")
    print("This demo shows how to detect text and replace it with custom content")
    print("Press 'q' to quit, 's' to cycle through replacement strategies")
    
    # Initialize the processing pipeline
    pipeline = ProcessingPipeline(
        video_source=0,  # Use default camera
        resolution=(640, 480),
        ocr_languages=['en'],
        use_gpu=False,
        detection_interval=5  # Detect text every 5 frames
    )
    
    # Create a performance monitor
    perf_monitor = PerformanceMonitor()
    
    # Initialize drawing utils
    drawing = DrawingUtils()
    
    # Create a simple replacement image (a colored rectangle with text)
    def create_replacement_image(width=100, height=50, text="CENSORED"):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (0, 0, 200)  # Red background
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 0.5, (255, 255, 255), 1)
        
        return img
    
    # Create a custom replacement function
    def custom_replacement(frame, detection):
        # Get the bounding box
        x, y, w, h = detection['rect']
        
        # Create a pulsating effect based on time
        intensity = int(127 + 127 * np.sin(time.time() * 5))
        color = (0, intensity, intensity)  # Cyan with pulsating intensity
        
        # Draw a filled rectangle with pulsating color
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        
        # Add a warning symbol
        cv2.putText(frame, "!", (x + w//2 - 5, y + h//2 + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    # Create the replacement image
    replacement_img = create_replacement_image(200, 50, "CENSORED")
    
    # Start the pipeline
    if not pipeline.start():
        print("Failed to start the pipeline")
        return
    
    # Available replacement strategies
    strategies = ['text', 'blur', 'image', 'function']
    current_strategy = 0
    
    try:
        while True:
            perf_monitor.start_timer('frame')
            
            # Read a frame
            success, frame = pipeline.capture.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Process the frame
            processed_frame, detections = pipeline.process_frame(frame)
            
            # Apply the current replacement strategy
            strategy = strategies[current_strategy]
            if detections:
                if strategy == 'text':
                    processed_frame = pipeline.text_replacer.replace_all(
                        processed_frame, detections, 
                        replacement_strategy='text',
                        replacement_text="[REDACTED]",
                        font_scale=0.6, bg_color=(0, 0, 200)
                    )
                elif strategy == 'blur':
                    processed_frame = pipeline.text_replacer.replace_all(
                        processed_frame, detections, 
                        replacement_strategy='blur'
                    )
                elif strategy == 'image':
                    for detection in detections:
                        # Resize the replacement image to fit the detection
                        x, y, w, h = detection['rect']
                        resized_img = resize_image(replacement_img, width=w, height=h)
                        processed_frame = pipeline.text_replacer.replace_with_image(
                            processed_frame, detection, resized_img, alpha=0.9
                        )
                elif strategy == 'function':
                    for detection in detections:
                        processed_frame = pipeline.text_replacer.replace_with_function(
                            processed_frame, detection, custom_replacement
                        )
            
            # Add a header showing the current strategy
            processed_frame = drawing.draw_filled_rectangle(
                processed_frame, (0, 0), (processed_frame.shape[1], 40), 
                color=(50, 50, 50), alpha=0.7
            )
            strategy_text = f"Replacement Strategy: {strategy.upper()}"
            processed_frame = drawing.draw_text(
                processed_frame, strategy_text, 
                (20, 25), font_scale=0.6, color=(255, 255, 255)
            )
            
            # Add instructions
            instructions = "Press 's' to change strategy, 'q' to quit"
            processed_frame = drawing.draw_text(
                processed_frame, instructions, 
                (processed_frame.shape[1] - 320, 25), font_scale=0.5, color=(200, 200, 200)
            )
            
            # Display detection count
            if detections:
                count_text = f"Detected {len(detections)} text regions"
                processed_frame = drawing.draw_text(
                    processed_frame, count_text, 
                    (20, 60), font_scale=0.5, color=(0, 255, 0)
                )
            
            # Display FPS
            fps = 1.0 / perf_monitor.stop_timer('frame') if perf_monitor.get_average_time('frame') > 0 else 0
            fps_text = f"FPS: {fps:.1f}"
            processed_frame = drawing.draw_text(
                processed_frame, fps_text, 
                (processed_frame.shape[1] - 80, processed_frame.shape[0] - 20), 
                font_scale=0.5, color=(0, 255, 0)
            )
            
            # Display the frame
            cv2.imshow('Text Replacement Demo', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                current_strategy = (current_strategy + 1) % len(strategies)
                print(f"Switched to {strategies[current_strategy].upper()} replacement strategy")
    
    finally:
        # Clean up
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Demo finished")

if __name__ == "__main__":
    main()