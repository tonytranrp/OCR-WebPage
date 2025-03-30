import cv2
import time
import numpy as np
from typing import Tuple, Optional, Union, List, Dict
from collections import deque

class VideoCapture:
    """
    A lightweight wrapper around OpenCV's VideoCapture for efficient video capture.
    Optimized for high frame rates (30-60 FPS) and real-time processing.
    Memory-optimized with frame buffer pooling to reduce allocations.
    """
    
    def __init__(self, source: Union[int, str] = 0, resolution: Tuple[int, int] = (640, 480), 
                 buffer_pool_size: int = 5, enable_memory_optimization: bool = True):
        """
        Initialize the video capture from a camera or video file.
        
        Args:
            source: Camera index (0 for default camera) or path to video file
            resolution: Desired resolution as (width, height)
            buffer_pool_size: Number of frame buffers to pre-allocate for reuse
            enable_memory_optimization: Whether to enable memory optimization features
        """
        self.source = source
        self.resolution = resolution
        self.buffer_pool_size = max(2, buffer_pool_size)  # Minimum of 2 buffers
        self.enable_memory_optimization = enable_memory_optimization
        self.cap = None
        self.fps = 0
        self.frame_time = 0
        self.is_running = False
        
        # Frame buffer pool for memory reuse
        self.frame_pool = []
        self.available_buffers = []
        self.current_frame_index = 0
    
    def _initialize_frame_pool(self) -> None:
        """
        Initialize the frame buffer pool with pre-allocated buffers.
        """
        if not self.enable_memory_optimization:
            return
            
        # Clear any existing buffers
        self.frame_pool.clear()
        self.available_buffers.clear()
        
        # Pre-allocate frame buffers with the correct resolution
        for i in range(self.buffer_pool_size):
            # Create a pre-allocated buffer with the right dimensions
            buffer = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            self.frame_pool.append(buffer)
            self.available_buffers.append(i)
            
        print(f"Initialized frame pool with {self.buffer_pool_size} buffers")
    
    def _get_buffer(self) -> np.ndarray:
        """
        Get an available buffer from the pool or create a new one if needed.
        
        Returns:
            A numpy array buffer to use for frame capture
        """
        if not self.enable_memory_optimization or not self.frame_pool:
            # If optimization is disabled or pool isn't initialized, return None
            # to let OpenCV allocate memory
            return None
            
        if not self.available_buffers:
            # If no buffers are available, create a new one
            buffer = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            self.frame_pool.append(buffer)
            buffer_index = len(self.frame_pool) - 1
        else:
            # Get an available buffer from the pool
            buffer_index = self.available_buffers.pop(0)
            
        self.current_frame_index = buffer_index
        return self.frame_pool[buffer_index]
    
    def _release_buffer(self, buffer_index: int) -> None:
        """
        Release a buffer back to the pool.
        
        Args:
            buffer_index: Index of the buffer to release
        """
        if not self.enable_memory_optimization or buffer_index >= len(self.frame_pool):
            return
            
        if buffer_index not in self.available_buffers:
            self.available_buffers.append(buffer_index)
    
    def start(self) -> bool:
        """
        Start the video capture.
        
        Returns:
            bool: True if capture started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                print(f"Error: Could not open video source {self.source}")
                return False
                
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Try to set higher FPS if possible
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            
            # Get actual FPS
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera initialized at {self.fps} FPS")
            
            # Initialize the frame buffer pool
            self._initialize_frame_pool()
            
            self.is_running = True
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video source using buffer pooling to reduce memory allocations.
        
        Returns:
            Tuple containing:
                - success (bool): True if frame was successfully captured
                - frame: The captured frame or None if unsuccessful
        """
        if not self.is_running or self.cap is None:
            return False, None
            
        # Measure frame capture time for FPS calculation
        start_time = time.time()
        
        # Get a buffer from the pool if memory optimization is enabled
        if self.enable_memory_optimization and self.frame_pool:
            buffer = self._get_buffer()
            if buffer is not None:
                success = self.cap.read(buffer)
                self.frame_time = time.time() - start_time
                
                if not success:
                    # Release the buffer back to the pool
                    self._release_buffer(self.current_frame_index)
                    return False, None
                    
                return success, buffer
        
        # Fall back to standard OpenCV read if optimization is disabled or failed
        success, frame = self.cap.read()
        self.frame_time = time.time() - start_time
        
        return success, frame
    
    def get_fps(self) -> float:
        """
        Get the current FPS based on actual frame processing time.
        
        Returns:
            float: Current frames per second
        """
        if self.frame_time > 0:
            return 1.0 / self.frame_time
        return self.fps
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics for the video capture.
        
        Returns:
            Dict containing memory usage information
        """
        memory_info = {
            "frame_pool_size": len(self.frame_pool),
            "available_buffers": len(self.available_buffers),
            "buffer_memory_bytes": 0
        }
        
        if self.frame_pool and len(self.frame_pool) > 0:
            # Calculate memory used by frame buffers
            buffer = self.frame_pool[0]
            bytes_per_buffer = buffer.nbytes
            memory_info["buffer_memory_bytes"] = bytes_per_buffer * len(self.frame_pool)
            memory_info["bytes_per_buffer"] = bytes_per_buffer
            
        return memory_info
    
    def release(self) -> None:
        """
        Release the video capture resources.
        """
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            self.cap = None
            
        # Clear frame pool to free memory
        self.frame_pool.clear()
        self.available_buffers.clear()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()