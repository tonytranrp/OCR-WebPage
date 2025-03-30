import cv2
import numpy as np
import time
import threading
import queue
import gc
from typing import List, Dict, Any, Tuple, Optional, Callable, Union, Deque
from collections import deque

from ..capture.video_capture import VideoCapture
from ..ocr.text_detector import TextDetector
from ..ocr.text_replacer import TextReplacer
from ..drawing.drawing_utils import DrawingUtils
from ..utils.performance import PerformanceMonitor

class ProcessingPipeline:
    """
    Main video processing pipeline that integrates video capture, OCR detection,
    text replacement, and drawing utilities into a cohesive system.
    Memory-optimized with buffer pooling and efficient resource management.
    """
    
    def __init__(self, video_source: Union[int, str] = 0, 
                 resolution: Tuple[int, int] = (640, 480),
                 ocr_languages: List[str] = ['en'],
                 use_gpu: bool = True,
                 detection_interval: int = 5,
                 buffer_size: int = 5,
                 use_threading: bool = True,
                 tight_bbox: bool = True,
                 enable_memory_optimization: bool = True,
                 frame_pool_size: int = 8,
                 max_cache_size: int = 5,
                 low_memory_mode: bool = True):
        """
        Initialize the video processing pipeline.
        
        Args:
            video_source: Camera index or video file path
            resolution: Desired resolution as (width, height)
            ocr_languages: List of language codes for OCR
            use_gpu: Whether to use GPU acceleration for OCR
            detection_interval: Number of frames between OCR detections
            buffer_size: Size of the frame buffer for smoother playback
            use_threading: Whether to use multi-threading for OCR processing
            tight_bbox: Whether to create tight bounding boxes around text
            enable_memory_optimization: Whether to enable memory optimization features
            frame_pool_size: Number of frame buffers to pre-allocate for reuse
            max_cache_size: Maximum number of detection results to cache
            low_memory_mode: Whether to use aggressive memory optimization for low-RAM devices
        """
        self.video_source = video_source
        self.resolution = resolution
        self.ocr_languages = ocr_languages
        self.use_gpu = use_gpu
        self.detection_interval = detection_interval
        self.buffer_size = max(2, buffer_size)  # Minimum buffer size of 2
        self.use_threading = use_threading
        self.tight_bbox = tight_bbox
        self.enable_memory_optimization = enable_memory_optimization
        self.frame_pool_size = max(4, frame_pool_size)  # Minimum pool size of 4
        self.max_cache_size = max_cache_size
        self.low_memory_mode = low_memory_mode
        
        # Adjust parameters for low memory mode
        if self.low_memory_mode:
            # Reduce buffer sizes and increase detection interval in low memory mode
            self.buffer_size = min(self.buffer_size, 3)
            self.frame_pool_size = min(self.frame_pool_size, 4)
            self.detection_interval = max(self.detection_interval, 10)
        
        # Initialize components with memory optimization settings
        self.capture = VideoCapture(
            source=video_source, 
            resolution=resolution,
            buffer_pool_size=self.frame_pool_size,
            enable_memory_optimization=self.enable_memory_optimization
        )
        
        self.text_detector = TextDetector(
            languages=ocr_languages, 
            gpu=use_gpu,
            enable_memory_optimization=self.enable_memory_optimization,
            max_cache_size=self.max_cache_size,
            scale_factor=0.4 if self.low_memory_mode else 0.5  # Use smaller images in low memory mode
        )
        
        self.text_replacer = TextReplacer()
        self.drawing = DrawingUtils()
        self.performance = PerformanceMonitor(window_size=30)
        
        # State variables
        self.frame_count = 0
        self.last_detections = []
        self.processing_fps = 0
        self.is_running = False
        self.pre_processing_hooks = []
        self.post_processing_hooks = []
        
        # Threading and buffering variables
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.detection_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.detection_thread = None
        self.detection_active = False
        self.last_frame_time = 0
    
    def add_pre_processing_hook(self, hook: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Add a pre-processing hook function that will be applied to frames before OCR.
        
        Args:
            hook: Function that takes a frame and returns a processed frame
        """
        self.pre_processing_hooks.append(hook)
    
    def add_post_processing_hook(self, hook: Callable[[np.ndarray, List[Dict[str, Any]]], np.ndarray]) -> None:
        """
        Add a post-processing hook function that will be applied to frames after OCR.
        
        Args:
            hook: Function that takes a frame and detections and returns a processed frame
        """
        self.post_processing_hooks.append(hook)
    
    def start(self) -> bool:
        """
        Start the video capture and processing threads.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            return True
            
        success = self.capture.start()
        if success:
            self.is_running = True
            
            # Start the detection thread if threading is enabled
            if self.use_threading:
                self.detection_active = True
                self.detection_thread = threading.Thread(
                    target=self._detection_worker,
                    daemon=True
                )
                self.detection_thread.start()
                print("Started detection thread")
                
        return success
    
    def _detection_worker(self):
        """
        Worker thread for text detection processing.
        Runs in a separate thread to avoid blocking the main rendering loop.
        """
        while self.detection_active:
            try:
                # Get a frame from the queue with a timeout
                frame = self.detection_queue.get(timeout=0.5)
                
                # Start timing the detection process
                self.performance.start_timer('detection')
                
                # Perform OCR detection with tight bounding boxes
                detections = self.text_detector.detect(frame, tight_bbox=self.tight_bbox)
                
                # Stop timing and put the result in the queue
                self.performance.stop_timer('detection')
                self.result_queue.put(detections)
                
                # Mark the task as done
                self.detection_queue.task_done()
                
            except queue.Empty:
                # Queue is empty, just continue waiting
                continue
            except Exception as e:
                print(f"Error in detection worker: {e}")
                # Put an empty list in the result queue to avoid blocking
                self.result_queue.put([])
                self.detection_queue.task_done()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple containing:
                - Processed frame
                - List of text detections
        """
        # Start timing the frame processing
        self.performance.start_timer('frame_processing')
        
        # Apply pre-processing hooks
        processed_frame = frame.copy()
        for hook in self.pre_processing_hooks:
            processed_frame = hook(processed_frame)
        
        # Add the frame to the buffer for smoother playback
        self.frame_buffer.append(processed_frame.copy())
        
        # Determine if we need to perform OCR detection
        current_time = time.time()
        time_since_last_detection = current_time - self.last_frame_time
        should_detect = (self.frame_count % self.detection_interval == 0) and \
                       (time_since_last_detection > 0.1)  # Limit detection rate
        
        detections = self.last_detections
        
        # Handle OCR detection based on threading mode
        if self.use_threading:
            # Check if we have results from the detection thread
            try:
                if not self.result_queue.empty():
                    # Get the latest detection results
                    detections = self.result_queue.get_nowait()
                    self.last_detections = detections
                    self.result_queue.task_done()
            except queue.Empty:
                pass
            
            # Queue a new frame for detection if needed
            if should_detect and self.detection_queue.qsize() < 1:
                try:
                    self.detection_queue.put_nowait(processed_frame)
                    self.last_frame_time = current_time
                except queue.Full:
                    pass  # Queue is full, skip this frame
        else:
            # Perform OCR in the main thread at specified intervals
            if should_detect:
                self.performance.start_timer('detection')
                detections = self.text_detector.detect(processed_frame, tight_bbox=self.tight_bbox)
                self.performance.stop_timer('detection')
                self.last_detections = detections
                self.last_frame_time = current_time
        
        # Apply post-processing hooks
        for hook in self.post_processing_hooks:
            processed_frame = hook(processed_frame, detections)
        
        self.frame_count += 1
        
        # Stop timing the frame processing
        processing_time = self.performance.stop_timer('frame_processing')
        if processing_time > 0:
            self.processing_fps = 1.0 / processing_time
        
        return processed_frame, detections
    
    def run(self, display: bool = True, 
            replacement_strategy: Optional[str] = None,
            replacement_text: str = '[REDACTED]',
            replacement_image: Optional[np.ndarray] = None,
            replacement_func: Optional[Callable] = None,
            visualize_detections: bool = False,
            max_fps: Optional[int] = None,
            show_performance: bool = True) -> None:
        """
        Run the video processing pipeline continuously.
        
        Args:
            display: Whether to display the processed frames
            replacement_strategy: Text replacement strategy ('text', 'blur', 'image', 'function', or None)
            replacement_text: Text to use when strategy is 'text'
            replacement_image: Image to use when strategy is 'image'
            replacement_func: Function to use when strategy is 'function'
            visualize_detections: Whether to visualize text detections
            max_fps: Maximum FPS to limit processing to (None for unlimited)
            show_performance: Whether to show performance metrics on the frame
        """
        if not self.start():
            print("Failed to start video capture")
            return
        
        # Initialize the frame buffer with empty frames
        for _ in range(self.buffer_size):
            self.frame_buffer.append(np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8))
        
        try:
            while True:
                self.performance.start_timer('loop')
                
                # Read frame
                success, frame = self.capture.read()
                if not success:
                    print("Failed to read frame")
                    break
                
                # Process frame (this will add it to the buffer)
                processed_frame, detections = self.process_frame(frame)
                
                # Get a frame from the buffer for display (smoother playback)
                # Use the most recent fully processed frame
                if len(self.frame_buffer) > 0:
                    display_frame = self.frame_buffer[-1].copy()
                else:
                    display_frame = processed_frame.copy()
                
                # Apply text replacement if requested
                if replacement_strategy is not None:
                    display_frame = self.text_replacer.replace_all(
                        display_frame, detections,
                        replacement_strategy=replacement_strategy,
                        replacement_text=replacement_text,
                        replacement_image=replacement_image,
                        replacement_func=replacement_func
                    )
                
                # Visualize detections if requested
                if visualize_detections and detections:
                    display_frame = self.text_detector.visualize_detections(display_frame, detections)
                
                # Display performance metrics on frame
                if show_performance:
                    # Get performance metrics
                    processing_fps = self.processing_fps
                    camera_fps = self.capture.get_fps()
                    detection_time = self.performance.get_average_time('detection')
                    
                    # Create performance text
                    fps_text = f"Processing: {processing_fps:.1f} FPS | Camera: {camera_fps:.1f} FPS"
                    if detection_time > 0:
                        detection_text = f"OCR: {detection_time*1000:.1f} ms"
                    else:
                        detection_text = "OCR: N/A"
                    
                    # Draw performance metrics
                    cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                
                # Display the frame
                if display:
                    cv2.imshow('Video Processing', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Limit FPS if requested
                loop_time = self.performance.stop_timer('loop')
                if max_fps is not None:
                    target_time = 1.0 / max_fps
                    if loop_time < target_time:
                        time.sleep(target_time - loop_time)
        
        finally:
            self.stop()
    
    def stop(self) -> None:
        """
        Stop the video processing pipeline and release resources.
        """
        if self.is_running:
            # Stop the detection thread if it's running
            if self.use_threading and self.detection_thread is not None:
                self.detection_active = False
                # Wait for the thread to finish with a timeout
                if self.detection_thread.is_alive():
                    self.detection_thread.join(timeout=1.0)
                self.detection_thread = None
            
            # Clear the queues
            while not self.detection_queue.empty():
                try:
                    self.detection_queue.get_nowait()
                    self.detection_queue.task_done()
                except queue.Empty:
                    break
                    
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.task_done()
                except queue.Empty:
                    break
            
            # Clear the frame buffer
            self.frame_buffer.clear()
            
            # Release capture resources
            self.capture.release()
            cv2.destroyAllWindows()
            self.is_running = False
    
    def get_fps(self) -> float:
        """
        Get the current processing FPS.
        
        Returns:
            float: Current frames per second for processing
        """
        return self.processing_fps
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()