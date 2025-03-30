import socket
import cv2
import numpy as np
import threading
import json
import time
from typing import Optional, Tuple, Dict, Any
from queue import Queue

class CameraServer:
    """Server that accepts connections from remote cameras and processes their video streams."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8081, max_clients: int = 5,
                 frame_buffer_size: int = 10, enable_gpu: bool = True):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.frame_buffer_size = frame_buffer_size
        self.enable_gpu = enable_gpu
        
        self.server_socket = None
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.is_running = False
        self.frame_buffers: Dict[str, Queue] = {}
        
        # GPU memory management
        if self.enable_gpu:
            try:
                import cupy as cp
                self.cp = cp
                self.gpu_context = cp.cuda.Device(0)
                self.gpu_streams: Dict[str, cp.cuda.Stream] = {}
                print("GPU acceleration enabled for network processing")
            except ImportError:
                print("Warning: GPU acceleration requested but cupy not available")
                self.enable_gpu = False
    
    def start(self) -> bool:
        """Start the camera server."""
        try:
            # Create socket with timeout to avoid blocking indefinitely
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.settimeout(30)  # 30 second timeout
            
            # Try to bind to the specified port, with fallback options
            try:
                self.server_socket.bind((self.host, self.port))
            except OSError as e:
                # If the port is already in use or permission denied, try alternative ports
                if e.errno == 10013 or e.errno == 10048:  # Permission denied or address already in use
                    print(f"Port {self.port} is unavailable, trying alternative ports...")
                    # Try a range of alternative ports
                    for alt_port in range(self.port + 1, self.port + 20):
                        try:
                            self.server_socket.bind((self.host, alt_port))
                            self.port = alt_port  # Update port if successful
                            print(f"Successfully bound to alternative port {alt_port}")
                            break
                        except OSError:
                            continue
                    else:  # No available ports found
                        raise Exception(f"Could not find an available port. Original error: {e}")
                else:
                    # Re-raise other socket errors
                    raise
            
            # Start listening for connections
            self.server_socket.listen(self.max_clients)
            
            self.is_running = True
            print(f"Camera server started on {self.host}:{self.port}")
            
            # Start accepting client connections
            self.accept_thread = threading.Thread(target=self._accept_connections)
            self.accept_thread.daemon = True
            self.accept_thread.start()
            
            return True
        except Exception as e:
            print(f"Failed to start camera server: {e}")
            if hasattr(self, 'server_socket') and self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass
            return False
    
    def _accept_connections(self) -> None:
        """Accept incoming client connections."""
        while self.is_running:
            try:
                client_socket, address = self.server_socket.accept()
                client_id = f"{address[0]}:{address[1]}"
                
                # Initialize client resources
                self.frame_buffers[client_id] = Queue(maxsize=self.frame_buffer_size)
                if self.enable_gpu:
                    with self.gpu_context:
                        self.gpu_streams[client_id] = self.cp.cuda.Stream()
                
                # Start client handling thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_id)
                )
                client_thread.daemon = True
                client_thread.start()
                
                self.clients[client_id] = {
                    'socket': client_socket,
                    'address': address,
                    'thread': client_thread,
                    'last_frame_time': time.time()
                }
                
                print(f"New client connected: {client_id}")
            except Exception as e:
                print(f"Error accepting client connection: {e}")
    
    def _handle_client(self, client_socket: socket.socket, client_id: str) -> None:
        """Handle communication with a connected client."""
        try:
            # Receive client configuration
            config_data = self._receive_json(client_socket)
            if not config_data:
                raise Exception("Failed to receive client configuration")
            
            # Store client configuration
            self.clients[client_id]['config'] = config_data
            print(f"Client {client_id} connected with config: {config_data}")
            
            # Initialize frame rate tracking
            frame_count = 0
            start_time = time.time()
            last_fps_update = start_time
            
            # Process frames from client
            while self.is_running and client_id in self.clients:
                # Receive frame size
                size_data = client_socket.recv(8)
                if not size_data:
                    break
                frame_size = int.from_bytes(size_data, byteorder='big')
                
                # Receive frame data
                frame_data = self._receive_exact(client_socket, frame_size)
                if not frame_data:
                    break
                
                # Decode frame
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                # Skip processing if frame is empty or invalid
                if frame is None or frame.size == 0:
                    continue
                
                # Process frame using GPU if available
                if self.enable_gpu:
                    with self.gpu_context:
                        stream = self.gpu_streams[client_id]
                        with stream:
                            # Transfer frame to GPU
                            gpu_frame = self.cp.asarray(frame)
                            # Apply minimal processing to reduce latency
                            # No horizontal flip to preserve original orientation
                            gpu_frame = self.cp.ascontiguousarray(gpu_frame)
                            # Transfer back to CPU
                            frame = self.cp.asnumpy(gpu_frame)
                
                # Add to frame buffer, dropping oldest frame if full
                if self.frame_buffers[client_id].full():
                    try:
                        # Remove oldest frame to make room
                        self.frame_buffers[client_id].get_nowait()
                    except:
                        pass
                
                # Add new frame to buffer
                self.frame_buffers[client_id].put(frame)
                
                # Update last frame time
                self.clients[client_id]['last_frame_time'] = time.time()
                
                # Update frame rate statistics
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_update >= 5.0:  # Update every 5 seconds
                    fps = frame_count / (current_time - last_fps_update)
                    self.clients[client_id]['fps'] = fps
                    print(f"Client {client_id} streaming at {fps:.1f} FPS")
                    frame_count = 0
                    last_fps_update = current_time
        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
        finally:
            self._remove_client(client_id)
    
    def _receive_json(self, client_socket: socket.socket) -> Optional[Dict]:
        """Receive JSON data from client."""
        try:
            size_data = client_socket.recv(4)
            if not size_data:
                return None
            size = int.from_bytes(size_data, byteorder='big')
            
            json_data = self._receive_exact(client_socket, size)
            if not json_data:
                return None
            
            return json.loads(json_data.decode('utf-8'))
        except:
            return None
    
    def _receive_exact(self, sock: socket.socket, size: int) -> Optional[bytes]:
        """Receive exact number of bytes from socket."""
        data = bytearray()
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
    
    def _remove_client(self, client_id: str) -> None:
        """Remove a client and clean up its resources."""
        if client_id in self.clients:
            try:
                self.clients[client_id]['socket'].close()
            except:
                pass
            
            # Clean up GPU resources
            if self.enable_gpu and client_id in self.gpu_streams:
                try:
                    with self.gpu_context:
                        del self.gpu_streams[client_id]
                except:
                    pass
            
            # Clean up other resources
            if client_id in self.frame_buffers:
                del self.frame_buffers[client_id]
            
            # Clean up other resources
            if client_id in self.frame_buffers:
                del self.frame_buffers[client_id]
            
            del self.clients[client_id]
            print(f"Client disconnected: {client_id}")
    
    def get_frame(self, client_id: str) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame from a client's buffer."""
        if client_id not in self.frame_buffers:
            return False, None
        
        try:
            frame = self.frame_buffers[client_id].get_nowait()
            return True, frame
        except:
            return False, None
    
    def get_client_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about connected clients."""
        return {cid: {
            'address': info['address'],
            'last_frame_time': info['last_frame_time']
        } for cid, info in self.clients.items()}
    
    def stop(self) -> None:
        """Stop the camera server and clean up resources."""
        self.is_running = False
        
        # Close all client connections
        for client_id in list(self.clients.keys()):
            self._remove_client(client_id)
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Clean up GPU resources
        if self.enable_gpu:
            try:
                with self.gpu_context:
                    for stream in self.gpu_streams.values():
                        stream.synchronize()
                    self.gpu_streams.clear()
            except:
                pass