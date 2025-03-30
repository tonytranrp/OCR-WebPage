import cv2
import numpy as np
import easyocr
import time
import threading
import gc
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from functools import lru_cache

class TextDetector:
    """
    A lightweight text detector using EasyOCR for efficient text detection and recognition.
    Optimized for real-time processing with caching and performance enhancements.
    Memory-optimized with buffer reuse and lazy loading to reduce RAM consumption.
    """
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True, 
                 detection_threshold: float = 0.4, recognition_threshold: float = 0.5,
                 scale_factor: float = 1, enable_memory_optimization: bool = True,
                 max_cache_size: int = 5, gpu_batch_size: int = 4):
        """
        Initialize the text detector.
        
        Args:
            languages: List of language codes to detect
            gpu: Whether to use GPU acceleration
            detection_threshold: Confidence threshold for text detection
            recognition_threshold: Confidence threshold for text recognition
            scale_factor: Scale factor for resizing images before OCR (smaller = faster but less accurate)
            enable_memory_optimization: Whether to enable memory optimization features
            max_cache_size: Maximum number of results to cache
        """
        self.languages = languages
        self.gpu = gpu
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.scale_factor = max(0.1, min(0.8, scale_factor))  # Adjust scale factor for faster processing
        self.enable_memory_optimization = enable_memory_optimization
        self.max_cache_size = max_cache_size
        
        # Lazy-loaded resources
        self.reader = None
        self.reader_lock = threading.Lock()
        self.reader_initialized = False
        self.gpu_batch_size = gpu_batch_size
        
        # GPU memory management
        if self.gpu:
            try:
                import torch
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                if self.device.type == 'cuda':
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                    # Set optimal GPU memory usage
                    torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory
                    torch.cuda.empty_cache()
            except ImportError:
                print("Warning: GPU acceleration requested but PyTorch not available")
                self.gpu = False
        
        # Caching variables
        self.last_frame_hash = None
        self.cached_results = None
        self.cache = {}
        self.cache_keys = []
        
        # Performance tracking
        self.processing_time = 0
        self.original_frame_size = None
        
        # Reusable buffers for intermediate processing
        self.processing_buffers = {}
        
        # Configure hash computation without using lru_cache directly on numpy arrays
        # We'll implement our own simple caching mechanism instead
        self._frame_hash_cache = {}
        self._frame_hash_cache_keys = []
    
    def _initialize_reader(self):
        """
        Initialize the EasyOCR reader (lazy initialization to save resources).
        Thread-safe implementation with locking to prevent multiple initializations.
        """
        # Use a lock to prevent multiple threads from initializing the reader simultaneously
        with self.reader_lock:
            if not self.reader_initialized:
                print(f"Initializing EasyOCR with languages: {self.languages}")
                try:
                    # Force garbage collection before loading the model to free memory
                    if self.enable_memory_optimization:
                        gc.collect()
                    
                    self.reader = easyocr.Reader(
                        self.languages, 
                        gpu=self.gpu,
                        detector=True,
                        recognizer=True,
                        # Use quantized models if memory optimization is enabled
                        quantize=self.enable_memory_optimization
                    )
                    self.reader_initialized = True
                except Exception as e:
                    print(f"Error initializing EasyOCR reader: {e}")
                    raise
                finally:
                    # Force garbage collection after loading to clean up temporary objects
                    if self.enable_memory_optimization:
                        gc.collect()
    
    def _compute_frame_hash(self, frame: np.ndarray) -> int:
        """
        Compute a hash value for a frame that can be used as a dictionary key.
        This implementation avoids using numpy arrays directly as keys.
        
        Args:
            frame: Input frame
            
        Returns:
            int: A hash value representing the frame content
        """
        # Check if we have a cached hash for this frame
        # We can't use the frame directly as a key, so we'll use its id
        frame_id = id(frame)
        
        # Check if we've already computed a hash for this exact frame object
        if frame_id in self._frame_hash_cache:
            return self._frame_hash_cache[frame_id]
        
        try:
            # Get or create a reusable buffer for the small frame
            if self.enable_memory_optimization and 'hash_buffer' in self.processing_buffers:
                # Reuse existing buffer
                small_frame = self.processing_buffers['hash_buffer']
                # Resize in-place to the small frame buffer
                cv2.resize(frame, (32, 32), dst=small_frame)
            else:
                # Create a new buffer or use standard resize
                small_frame = cv2.resize(frame, (32, 32))
                if self.enable_memory_optimization:
                    # Store for future reuse
                    self.processing_buffers['hash_buffer'] = small_frame
            
            # Convert to grayscale to reduce dimensionality if it's not already
            if len(small_frame.shape) > 2 and small_frame.shape[2] > 1:
                if self.enable_memory_optimization and 'gray_hash_buffer' in self.processing_buffers:
                    gray_small = self.processing_buffers['gray_hash_buffer']
                    cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY, dst=gray_small)
                else:
                    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    if self.enable_memory_optimization:
                        self.processing_buffers['gray_hash_buffer'] = gray_small
            else:
                gray_small = small_frame
            
            # Compute a hash from the bytes of the grayscale image
            frame_bytes = gray_small.tobytes()
            hash_value = hash(frame_bytes)
            
            # Store in our simple cache
            self._frame_hash_cache[frame_id] = hash_value
            self._frame_hash_cache_keys.append(frame_id)
            
            # Keep cache size limited
            if len(self._frame_hash_cache_keys) > 16:  # Similar to lru_cache maxsize
                oldest_key = self._frame_hash_cache_keys.pop(0)
                if oldest_key in self._frame_hash_cache:
                    del self._frame_hash_cache[oldest_key]
            
            return hash_value
            
        except Exception as e:
            # If anything fails, use a simpler hash based on frame dimensions and a sample of pixels
            print(f"Warning: Advanced hash computation failed: {e}, using fallback method")
            try:
                # Use shape and mean values as a simple hash
                shape_str = str(frame.shape)
                # Sample a few pixels from different parts of the image
                h, w = frame.shape[:2]
                samples = []
                if h > 0 and w > 0:
                    # Sample from corners and center if possible
                    samples.append(str(frame[0, 0].sum()))
                    if h > 10 and w > 10:
                        samples.append(str(frame[h//2, w//2].sum()))
                    if h > 1 and w > 1:
                        samples.append(str(frame[h-1, w-1].sum()))
                
                # Combine shape and samples into a hash
                hash_str = shape_str + ''.join(samples)
                return hash(hash_str)
            except:
                # Ultimate fallback - just use the current time
                return hash(str(time.time()))
    
    def _create_tight_bbox(self, frame: np.ndarray, rect: Tuple[int, int, int, int]) -> List[List[int]]:
        """
        Create a tighter bounding box around text using contour analysis.
        Memory-optimized to reuse buffers and reduce allocations.
        
        Args:
            frame: Input frame
            rect: Rectangle coordinates (x, y, w, h)
            
        Returns:
            List of points forming a tight polygon around the text
        """
        x, y, w, h = rect
        
        # Early return for tiny regions to avoid processing overhead
        if w < 5 or h < 5:
            return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        
        # Extract the region of interest (ROI) - this is a view, not a copy
        roi = frame[y:y+h, x:x+w]
        
        # Reuse buffers for intermediate processing steps if memory optimization is enabled
        if self.enable_memory_optimization:
            # Get or create reusable buffers with the right dimensions
            roi_key = f"roi_{w}x{h}"
            gray_key = f"gray_{w}x{h}"
            binary_key = f"binary_{w}x{h}"
            morph_key = f"morph_{w}x{h}"
            
            # Create or resize gray buffer
            if gray_key not in self.processing_buffers or self.processing_buffers[gray_key].shape[:2] != (h, w):
                self.processing_buffers[gray_key] = np.zeros((h, w), dtype=np.uint8)
            
            # Create or resize binary and morph buffers
            if binary_key not in self.processing_buffers or self.processing_buffers[binary_key].shape[:2] != (h, w):
                self.processing_buffers[binary_key] = np.zeros((h, w), dtype=np.uint8)
                self.processing_buffers[morph_key] = np.zeros((h, w), dtype=np.uint8)
            
            # Convert to grayscale using the reusable buffer
            if len(roi.shape) > 2 and roi.shape[2] > 1:
                cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY, dst=self.processing_buffers[gray_key])
                gray = self.processing_buffers[gray_key]
            else:
                # If already grayscale, just copy to our buffer
                np.copyto(self.processing_buffers[gray_key], roi)
                gray = self.processing_buffers[gray_key]
            
            # Apply adaptive thresholding to get a binary image
            binary = self.processing_buffers[binary_key]
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2, dst=binary)
            
            # Apply morphological operations to connect nearby text components
            morph = self.processing_buffers[morph_key]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, dst=morph, iterations=1)
        else:
            # Standard processing without buffer reuse
            # Convert to grayscale if it's not already
            if len(roi.shape) > 2 and roi.shape[2] > 1:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()
            
            # Apply adaptive thresholding to get a binary image
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
            
            # Apply morphological operations to connect nearby text components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours - this creates new arrays but is unavoidable
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, return the original rectangle as a polygon
            return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        
        # Combine all contours to get a tight boundary around all text
        all_contours = np.vstack([contour for contour in contours])
        hull = cv2.convexHull(all_contours)
        
        # Simplify the hull to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # Offset the points by the ROI position
        tight_bbox = [[p[0][0] + x, p[0][1] + y] for p in approx]
        
        # Ensure we have at least 4 points for a polygon
        if len(tight_bbox) < 4:
            return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        
        return tight_bbox
    
    def detect(self, frame: np.ndarray, use_cache: bool = True, tight_bbox: bool = True) -> List[Dict[str, Any]]:
        """
        Detect and recognize text in a frame.
        Memory-optimized implementation with buffer reuse and efficient caching.
        
        Args:
            frame: Input frame
            use_cache: Whether to use cached results for identical frames
            tight_bbox: Whether to create tight bounding boxes around text
            
        Returns:
            List of detected text regions with text, bounding boxes, and confidence
        """
        # Ensure the reader is initialized
        self._initialize_reader()
        
        # Check if we can use cached results
        frame_hash = None
        if use_cache:
            try:
                frame_hash = self._compute_frame_hash(frame)
                
                # Check if result is in our LRU cache
                if frame_hash in self.cache:
                    return self.cache[frame_hash]
                    
                # Check if it matches the last processed frame
                if frame_hash == self.last_frame_hash and self.cached_results is not None:
                    return self.cached_results
            except Exception as e:
                # If hashing fails, log the error and continue without caching
                print(f"Warning: Frame hashing failed: {e}. Continuing without cache.")
                use_cache = False
                
            if frame_hash is not None:
                self.last_frame_hash = frame_hash
        
        # Measure processing time
        start_time = time.time()
        
        # Store original frame size for later scaling
        height, width = frame.shape[:2]
        self.original_frame_size = (width, height)
        
        # Scale down frame for faster processing using buffer reuse if possible
        if self.scale_factor < 1.0:
            scaled_width = int(width * self.scale_factor)
            scaled_height = int(height * self.scale_factor)
            
            # Try to reuse an existing buffer for the scaled frame
            scaled_key = f"scaled_{scaled_width}x{scaled_height}"
            if self.enable_memory_optimization and scaled_key in self.processing_buffers:
                scaled_frame = self.processing_buffers[scaled_key]
                cv2.resize(frame, (scaled_width, scaled_height), dst=scaled_frame)
            else:
                scaled_frame = cv2.resize(frame, (scaled_width, scaled_height))
                if self.enable_memory_optimization:
                    self.processing_buffers[scaled_key] = scaled_frame
        else:
            scaled_frame = frame
        
        # Run OCR detection on scaled frame
        try:
            raw_results = self.reader.readtext(scaled_frame)
        except Exception as e:
            print(f"Error in OCR detection: {e}")
            return []
        
        # Process and filter results
        results = []
        for bbox, text, confidence in raw_results:
            if confidence >= self.recognition_threshold:
                # Convert bbox to a more usable format
                # EasyOCR returns bbox as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Convert to (x, y, w, h) format for easier processing
                x_min = min(point[0] for point in bbox)
                y_min = min(point[1] for point in bbox)
                x_max = max(point[0] for point in bbox)
                y_max = max(point[1] for point in bbox)
                
                # Scale bounding box coordinates back to original size if needed
                if self.scale_factor < 1.0:
                    scale = 1.0 / self.scale_factor
                    bbox = [[p[0] * scale, p[1] * scale] for p in bbox]
                    x_min *= scale
                    y_min *= scale
                    x_max *= scale
                    y_max *= scale
                
                # Create rectangle coordinates
                rect = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                
                # Create tight bounding box if requested
                if tight_bbox:
                    tight_points = self._create_tight_bbox(frame, rect)
                else:
                    tight_points = bbox
                
                # Create a lightweight result dictionary
                # Use tuples instead of lists where possible to reduce memory
                results.append({
                    'text': text,
                    'bbox': tight_points,  # Tight bbox points
                    'rect': rect,  # x, y, w, h
                    'confidence': confidence
                })
            else:
                results.clear()  # Clear results if confidence is below threshold
        
        self.processing_time = time.time() - start_time
        self.cached_results = results
        
        # Update the LRU cache
        if use_cache and frame_hash is not None:
            try:
                # Add to cache
                self.cache[frame_hash] = results
                self.cache_keys.append(frame_hash)
                
                # Maintain cache size limit
                if len(self.cache_keys) > self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = self.cache_keys.pop(0)
                    if oldest_key in self.cache:
                        del self.cache[oldest_key]
            except Exception as e:
                # If caching fails, log the error and continue without caching
                print(f"Warning: Failed to update cache: {e}")
                # Clear problematic cache entries
                self.cache = {}
                self.cache_keys = []
        
        # Trigger garbage collection if memory optimization is enabled
        if self.enable_memory_optimization and len(results) > 10:
            # Only trigger GC for large result sets to avoid overhead
            gc.collect()
        
        return results
    
    def get_processing_time(self) -> float:
        """
        Get the processing time for the last detection.
        
        Returns:
            float: Processing time in seconds
        """
        return self.processing_time
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]],
                            draw_bbox: bool = True, draw_text: bool = True,
                            color: Tuple[int, int, int] = (0, 255, 0),
                            thickness: int = 2) -> np.ndarray:
        """
        Visualize text detections on a frame.
        
        Args:
            frame: Input frame
            detections: List of text detections from detect()
            draw_bbox: Whether to draw bounding boxes
            draw_text: Whether to draw detected text
            color: Color for bounding boxes and text
            thickness: Line thickness
            
        Returns:
            Frame with visualized detections
        """
        result = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            text = detection['text']
            confidence = detection['confidence']
            
            # Draw bounding box
            if draw_bbox:
                pts = np.array(bbox, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(result, [pts], True, color, thickness)
            
            # Draw text with confidence
            if draw_text:
                x, y = bbox[0]
                text_to_draw = f"{text} ({confidence:.2f})"
                cv2.putText(result, text_to_draw, (int(x), int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        return result