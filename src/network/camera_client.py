import socket
import cv2
import numpy as np
import json
import time
from typing import Optional, Tuple, Dict, Any

class CameraClient:
    """Client for streaming camera feed to a remote processing server."""
    
    def __init__(self, server_host: str, server_port: int = 8000,
                 camera_source: int = 0, resolution: Tuple[int, int] = (640, 480),
                 compression_quality: int = 90, enable_gpu: bool = True):
        self.server_host = server_host
        self.server_port = server_port
        self.camera_source = camera_source
        self.resolution = resolution
        self.compression_quality = compression_quality
        self.enable_gpu = enable_gpu
        
        self.socket = None
        self.camera = None
        self.is_connected = False
        
        # GPU acceleration setup
        if self.enable_gpu:
            try:
                import cupy as cp
                self.cp = cp
                self.gpu_context = cp.cuda.Device(0)
                self.gpu_stream = cp.cuda.Stream()
                print("GPU acceleration enabled for frame processing")
            except ImportError:
                print("Warning: GPU acceleration requested but cupy not available")
                self.enable_gpu = False
    
    def connect(self) -> bool:
        """Connect to the processing server."""
        try:
            # Initialize socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            
            # Initialize camera
            self.camera = cv2.VideoCapture(self.camera_source)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Send client configuration
            config = {
                'camera_id': str(self.camera_source),
                'resolution': self.resolution,
                'fps': self.camera.get(cv2.CAP_PROP_FPS),
                'gpu_enabled': self.enable_gpu
            }
            self._send_json(config)
            
            self.is_connected = True
            print(f"Connected to server at {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            self.disconnect()
            return False
    
    def _send_json(self, data: Dict) -> bool:
        """Send JSON data to server."""
        try:
            json_data = json.dumps(data).encode('utf-8')
            size = len(json_data)
            
            # Send size first
            self.socket.sendall(size.to_bytes(4, byteorder='big'))
            # Send data
            self.socket.sendall(json_data)
            return True
        except:
            return False
    
    def stream(self) -> None:
        """Stream camera feed to the server."""
        if not self.is_connected:
            print("Not connected to server")
            return
        
        # Initialize frame rate control variables
        last_time = time.time()
        frame_interval = 1.0 / 30.0  # Target 30 FPS
        frame_count = 0
        fps_display_interval = 5.0  # Display FPS every 5 seconds
        last_fps_time = last_time
        
        try:
            while self.is_connected:
                # Frame rate control
                current_time = time.time()
                elapsed = current_time - last_time
                
                # Only process frame if enough time has elapsed (reduces CPU/network load)
                if elapsed >= frame_interval:
                    # Read frame
                    success, frame = self.camera.read()
                    if not success:
                        print("Failed to read frame")
                        break
                    
                    # Process frame using GPU if available
                    if self.enable_gpu:
                        with self.gpu_context:
                            with self.gpu_stream:
                                # Transfer frame to GPU
                                gpu_frame = self.cp.asarray(frame)
                                # Apply pre-processing to improve quality and reduce size
                                gpu_frame = self.cp.ascontiguousarray(gpu_frame)
                                # Transfer back to CPU
                                frame = self.cp.asnumpy(gpu_frame)
                    
                    # Resize frame to reduce bandwidth if needed
                    if frame.shape[1] > self.resolution[0] or frame.shape[0] > self.resolution[1]:
                        frame = cv2.resize(frame, self.resolution)
                    
                    # Encode frame with optimized compression
                    encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.compression_quality]
                    _, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                    frame_data = encoded_frame.tobytes()
                    
                    # Send frame size
                    size = len(frame_data)
                    self.socket.sendall(size.to_bytes(8, byteorder='big'))
                    
                    # Send frame data
                    self.socket.sendall(frame_data)
                    
                    # Update timing variables
                    last_time = current_time
                    frame_count += 1
                    
                    # Display FPS periodically
                    if current_time - last_fps_time >= fps_display_interval:
                        fps = frame_count / (current_time - last_fps_time)
                        print(f"Streaming at {fps:.1f} FPS, frame size: {size/1024:.1f} KB")
                        frame_count = 0
                        last_fps_time = current_time
                else:
                    # Small sleep to prevent CPU hogging when waiting for next frame interval
                    time.sleep(0.001)
        except Exception as e:
            print(f"Error streaming camera feed: {e}")
        finally:
            self.disconnect()
    
    def disconnect(self) -> None:
        """Disconnect from the server and release resources."""
        self.is_connected = False
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        # Release camera
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
        
        # Clean up GPU resources
        if self.enable_gpu:
            try:
                with self.gpu_context:
                    self.gpu_stream.synchronize()
            except:
                pass
        
        print("Disconnected from server")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()