import cv2
import numpy as np
import time
import os
import sys
import math

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capture.video_capture import VideoCapture
from src.drawing.drawing_utils import DrawingUtils
from src.utils.performance import PerformanceMonitor

def main():
    print("Starting Custom Drawing Demo")
    print("This demo shows how to draw custom shapes and overlays on video frames")
    print("Press 'q' to quit, 'd' to cycle through drawing modes")
    
    # Initialize video capture
    cap = VideoCapture(source=0, resolution=(640, 480))
    if not cap.start():
        print("Failed to start video capture")
        return
    
    # Initialize drawing utilities
    drawing = DrawingUtils()
    
    # Create a performance monitor
    perf_monitor = PerformanceMonitor()
    
    # Drawing modes
    modes = ['rectangles', 'circles', 'lines', 'polygons', 'text', 'combined']
    current_mode = 0
    
    # Animation variables
    animation_time = 0
    
    try:
        while True:
            perf_monitor.start_timer('frame')
            
            # Read a frame
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Update animation time
            animation_time += 0.05
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Apply the current drawing mode
            mode = modes[current_mode]
            processed_frame = frame.copy()
            
            if mode == 'rectangles':
                # Draw animated rectangles
                for i in range(5):
                    # Calculate position based on time
                    offset = i * 72 + animation_time * 20
                    x = int(100 + 50 * math.sin(animation_time + i * 0.7))
                    y = int(100 + 30 * math.cos(animation_time + i * 0.5))
                    w = 80 + int(20 * math.sin(animation_time * 2))
                    h = 60 + int(15 * math.cos(animation_time * 2))
                    
                    # Generate color based on position
                    color = (int(127 + 127 * math.sin(animation_time + i)),
                            int(127 + 127 * math.sin(animation_time + i + 2)),
                            int(127 + 127 * math.sin(animation_time + i + 4)))
                    
                    # Draw filled rectangle with transparency
                    processed_frame = drawing.draw_filled_rectangle(
                        processed_frame, 
                        (x, y), 
                        (x + w, y + h), 
                        color=color, 
                        alpha=0.6
                    )
                    
                    # Draw rectangle border
                    processed_frame = drawing.draw_rectangle(
                        processed_frame, 
                        (x, y), 
                        (x + w, y + h), 
                        color=(255, 255, 255), 
                        thickness=1
                    )
            
            elif mode == 'circles':
                # Draw animated circles
                for i in range(8):
                    # Calculate position based on time
                    center_x = int(width/2 + 100 * math.cos(animation_time + i * math.pi/4))
                    center_y = int(height/2 + 100 * math.sin(animation_time + i * math.pi/4))
                    radius = 20 + int(10 * math.sin(animation_time * 3 + i))
                    
                    # Generate color based on position
                    color = (int(127 + 127 * math.sin(animation_time * 2 + i)),
                            int(127 + 127 * math.sin(animation_time * 2 + i + 2)),
                            int(127 + 127 * math.sin(animation_time * 2 + i + 4)))
                    
                    # Draw circle
                    processed_frame = drawing.draw_circle(
                        processed_frame, 
                        (center_x, center_y), 
                        radius, 
                        color=color, 
                        thickness=2
                    )
            
            elif mode == 'lines':
                # Draw animated lines
                for i in range(10):
                    # Calculate positions based on time
                    start_x = int(width/2 + 200 * math.cos(animation_time + i * math.pi/5))
                    start_y = int(height/2 + 200 * math.sin(animation_time + i * math.pi/5))
                    end_x = int(width/2 + 200 * math.cos(animation_time + i * math.pi/5 + math.pi))
                    end_y = int(height/2 + 200 * math.sin(animation_time + i * math.pi/5 + math.pi))
                    
                    # Generate color based on position
                    color = (int(127 + 127 * math.sin(animation_time + i)),
                            int(127 + 127 * math.sin(animation_time + i + 2)),
                            int(127 + 127 * math.sin(animation_time + i + 4)))
                    
                    # Draw line
                    processed_frame = drawing.draw_line(
                        processed_frame, 
                        (start_x, start_y), 
                        (end_x, end_y), 
                        color=color, 
                        thickness=2
                    )
            
            elif mode == 'polygons':
                # Draw animated polygons
                for i in range(3):
                    # Create polygon points
                    sides = i + 3  # Triangle, square, pentagon
                    points = []
                    radius = 80 + i * 20
                    center_x = width // 4 * (i + 1)
                    center_y = height // 2
                    
                    for j in range(sides):
                        angle = animation_time + j * (2 * math.pi / sides)
                        x = int(center_x + radius * math.cos(angle))
                        y = int(center_y + radius * math.sin(angle))
                        points.append((x, y))
                    
                    # Generate color based on position
                    color = (int(127 + 127 * math.sin(animation_time + i)),
                            int(127 + 127 * math.sin(animation_time + i + 2)),
                            int(127 + 127 * math.sin(animation_time + i + 4)))
                    
                    # Draw filled polygon with transparency
                    processed_frame = drawing.draw_filled_polygon(
                        processed_frame, 
                        points, 
                        color=color, 
                        alpha=0.5
                    )
                    
                    # Draw polygon outline
                    processed_frame = drawing.draw_polygon(
                        processed_frame, 
                        points, 
                        color=(255, 255, 255), 
                        thickness=2
                    )
            
            elif mode == 'text':
                # Draw animated text
                for i in range(5):
                    # Calculate position based on time
                    x = int(50 + i * 100 + 20 * math.sin(animation_time + i))
                    y = int(height/2 + 50 * math.cos(animation_time + i))
                    
                    # Generate color based on position
                    color = (int(127 + 127 * math.sin(animation_time + i)),
                            int(127 + 127 * math.sin(animation_time + i + 2)),
                            int(127 + 127 * math.sin(animation_time + i + 4)))
                    
                    # Generate text size based on time
                    font_scale = 0.5 + 0.3 * math.sin(animation_time * 2 + i)
                    
                    # Draw text
                    text = f"Text {i+1}"
                    processed_frame = drawing.draw_text(
                        processed_frame, 
                        text, 
                        (x, y), 
                        font_scale=font_scale, 
                        color=color, 
                        thickness=2
                    )
            
            elif mode == 'combined':
                # Draw a combination of shapes to create a complex animation
                
                # Draw a header bar
                processed_frame = drawing.draw_filled_rectangle(
                    processed_frame, 
                    (0, 0), 
                    (width, 50), 
                    color=(50, 50, 100), 
                    alpha=0.7
                )
                
                # Draw title text
                title = "Real-time Video Processing"
                processed_frame = drawing.draw_text(
                    processed_frame, 
                    title, 
                    (width//2 - 120, 30), 
                    font_scale=0.8, 
                    color=(255, 255, 255), 
                    thickness=2
                )
                
                # Draw animated circles in the background
                for i in range(5):
                    center_x = int(width/2 + 150 * math.cos(animation_time * 0.5 + i * math.pi/2.5))
                    center_y = int(height/2 + 100 * math.sin(animation_time * 0.5 + i * math.pi/2.5))
                    radius = 30 + int(15 * math.sin(animation_time + i))
                    
                    color = (int(50 + 50 * math.sin(animation_time + i)),
                            int(50 + 50 * math.sin(animation_time + i + 2)),
                            int(50 + 50 * math.sin(animation_time + i + 4)))
                    
                    processed_frame = drawing.draw_filled_rectangle(
                        processed_frame, 
                        (center_x - radius, center_y - radius), 
                        (center_x + radius, center_y + radius), 
                        color=color, 
                        alpha=0.3
                    )
                
                # Draw a border around the frame
                border_width = 10
                processed_frame = drawing.draw_rectangle(
                    processed_frame, 
                    (border_width, border_width), 
                    (width - border_width, height - border_width), 
                    color=(0, 255, 255), 
                    thickness=border_width//2
                )
                
                # Draw animated text at the bottom
                bottom_text = "Custom Drawing Demo"
                text_x = int(width/2 - 100 + 50 * math.sin(animation_time))
                processed_frame = drawing.draw_text(
                    processed_frame, 
                    bottom_text, 
                    (text_x, height - 30), 
                    font_scale=0.7, 
                    color=(255, 255, 0), 
                    thickness=2
                )
            
            # Add a header showing the current mode
            processed_frame = drawing.draw_filled_rectangle(
                processed_frame, (0, 0), (width, 40), 
                color=(50, 50, 50), alpha=0.7
            )
            mode_text = f"Drawing Mode: {mode.upper()}"
            processed_frame = drawing.draw_text(
                processed_frame, mode_text, 
                (20, 25), font_scale=0.6, color=(255, 255, 255)
            )
            
            # Add instructions
            instructions = "Press 'd' to change mode, 'q' to quit"
            processed_frame = drawing.draw_text(
                processed_frame, instructions, 
                (width - 300, 25), font_scale=0.5, color=(200, 200, 200)
            )
            
            # Display FPS
            fps = 1.0 / perf_monitor.stop_timer('frame') if perf_monitor.get_average_time('frame') > 0 else 0
            fps_text = f"FPS: {fps:.1f}"
            processed_frame = drawing.draw_text(
                processed_frame, fps_text, 
                (width - 80, height - 20), 
                font_scale=0.5, color=(0, 255, 0)
            )
            
            # Display the frame
            cv2.imshow('Custom Drawing Demo', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                current_mode = (current_mode + 1) % len(modes)
                print(f"Switched to {modes[current_mode].upper()} drawing mode")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Demo finished")

if __name__ == "__main__":
    main()