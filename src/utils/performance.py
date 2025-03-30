import time
import cv2
import numpy as np
from typing import Callable, Any, Dict, List, Tuple, Optional

class PerformanceMonitor:
    """
    Utility class for monitoring and optimizing performance of video processing operations.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize the performance monitor.
        
        Args:
            window_size: Number of measurements to keep for rolling average
        """
        self.window_size = window_size
        self.timings = {}
        self.timing_history = {}
    
    def start_timer(self, name: str) -> None:
        """
        Start a named timer.
        
        Args:
            name: Name of the timer
        """
        self.timings[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and return the elapsed time.
        
        Args:
            name: Name of the timer
            
        Returns:
            float: Elapsed time in seconds
        """
        if name not in self.timings:
            return 0.0
            
        elapsed = time.time() - self.timings[name]
        
        # Store in history for rolling average
        if name not in self.timing_history:
            self.timing_history[name] = []
            
        history = self.timing_history[name]
        history.append(elapsed)
        
        # Keep only the most recent measurements
        if len(history) > self.window_size:
            history.pop(0)
            
        return elapsed
    
    def get_average_time(self, name: str) -> float:
        """
        Get the average time for a named timer.
        
        Args:
            name: Name of the timer
            
        Returns:
            float: Average time in seconds
        """
        if name not in self.timing_history or not self.timing_history[name]:
            return 0.0
            
        return sum(self.timing_history[name]) / len(self.timing_history[name])
    
    def get_fps(self, name: str) -> float:
        """
        Get the frames per second for a named timer.
        
        Args:
            name: Name of the timer
            
        Returns:
            float: Frames per second
        """
        avg_time = self.get_average_time(name)
        if avg_time <= 0:
            return 0.0
            
        return 1.0 / avg_time
    
    def reset(self) -> None:
        """
        Reset all timers and history.
        """
        self.timings = {}
        self.timing_history = {}


def resize_for_performance(frame: np.ndarray, target_width: int = 640) -> np.ndarray:
    """
    Resize a frame to improve processing performance while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_width: Target width for resizing
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    # Only resize if the frame is larger than the target width
    if width > target_width:
        scale = target_width / width
        new_height = int(height * scale)
        return cv2.resize(frame, (target_width, new_height))
    
    return frame


def limit_processing_rate(func: Callable) -> Callable:
    """
    Decorator to limit the processing rate of a function to maintain performance.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    last_call_time = [0.0]  # Use a list to maintain state between calls
    min_interval = [0.033]  # Default to ~30 FPS
    
    def wrapper(*args, **kwargs):
        current_time = time.time()
        elapsed = current_time - last_call_time[0]
        
        # If called too soon, return None or previous result
        if elapsed < min_interval[0]:
            return kwargs.get('default_result', None)
            
        result = func(*args, **kwargs)
        last_call_time[0] = time.time()
        
        # Adjust interval based on execution time to maintain target FPS
        execution_time = time.time() - current_time
        if execution_time > min_interval[0]:
            # If execution takes longer than interval, adjust to actual time
            min_interval[0] = execution_time
        
        return result
    
    return wrapper


def create_processing_regions(frame: np.ndarray, region_size: Tuple[int, int] = (320, 240)) -> List[Dict[str, Any]]:
    """
    Divide a frame into processing regions for parallel or selective processing.
    
    Args:
        frame: Input frame
        region_size: Size of each region as (width, height)
        
    Returns:
        List of regions with coordinates and slices
    """
    height, width = frame.shape[:2]
    region_width, region_height = region_size
    
    regions = []
    
    for y in range(0, height, region_height):
        for x in range(0, width, region_width):
            # Calculate actual region size (may be smaller at edges)
            actual_width = min(region_width, width - x)
            actual_height = min(region_height, height - y)
            
            # Skip very small regions
            if actual_width < 32 or actual_height < 32:
                continue
                
            regions.append({
                'x': x,
                'y': y,
                'width': actual_width,
                'height': actual_height,
                'slice': (slice(y, y + actual_height), slice(x, x + actual_width)),
                'roi': frame[y:y + actual_height, x:x + actual_width]
            })
    
    return regions