import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union

class DrawingUtils:
    """
    Utility class for drawing on video frames.
    Provides methods for drawing text, shapes, and custom overlays.
    """
    
    @staticmethod
    def draw_text(frame: np.ndarray, text: str, position: Tuple[int, int], 
                 font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2, font_face: int = cv2.FONT_HERSHEY_SIMPLEX) -> np.ndarray:
        """
        Draw text on a frame.
        
        Args:
            frame: Input frame
            text: Text to draw
            position: (x, y) position of the text
            font_scale: Font scale factor
            color: Text color in BGR
            thickness: Line thickness
            font_face: Font type
            
        Returns:
            Frame with text drawn on it
        """
        # Create a copy to avoid modifying the original frame
        result = frame.copy()
        cv2.putText(result, text, position, font_face, font_scale, color, thickness)
        return result
    
    @staticmethod
    def draw_rectangle(frame: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int],
                      color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw a rectangle on a frame.
        
        Args:
            frame: Input frame
            top_left: (x, y) coordinates of the top-left corner
            bottom_right: (x, y) coordinates of the bottom-right corner
            color: Rectangle color in BGR
            thickness: Line thickness
            
        Returns:
            Frame with rectangle drawn on it
        """
        result = frame.copy()
        cv2.rectangle(result, top_left, bottom_right, color, thickness)
        return result
    
    @staticmethod
    def draw_filled_rectangle(frame: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int],
                             color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
        """
        Draw a semi-transparent filled rectangle on a frame.
        
        Args:
            frame: Input frame
            top_left: (x, y) coordinates of the top-left corner
            bottom_right: (x, y) coordinates of the bottom-right corner
            color: Rectangle color in BGR
            alpha: Transparency factor (0.0 to 1.0)
            
        Returns:
            Frame with filled rectangle drawn on it
        """
        result = frame.copy()
        overlay = result.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)  # -1 for filled rectangle
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        return result
    
    @staticmethod
    def draw_circle(frame: np.ndarray, center: Tuple[int, int], radius: int,
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw a circle on a frame.
        
        Args:
            frame: Input frame
            center: (x, y) coordinates of the circle center
            radius: Circle radius
            color: Circle color in BGR
            thickness: Line thickness
            
        Returns:
            Frame with circle drawn on it
        """
        result = frame.copy()
        cv2.circle(result, center, radius, color, thickness)
        return result
    
    @staticmethod
    def draw_line(frame: np.ndarray, start_point: Tuple[int, int], end_point: Tuple[int, int],
                 color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw a line on a frame.
        
        Args:
            frame: Input frame
            start_point: (x, y) coordinates of the line start point
            end_point: (x, y) coordinates of the line end point
            color: Line color in BGR
            thickness: Line thickness
            
        Returns:
            Frame with line drawn on it
        """
        result = frame.copy()
        cv2.line(result, start_point, end_point, color, thickness)
        return result
    
    @staticmethod
    def draw_polygon(frame: np.ndarray, points: List[Tuple[int, int]],
                    color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw a polygon on a frame.
        
        Args:
            frame: Input frame
            points: List of (x, y) coordinates of polygon vertices
            color: Polygon color in BGR
            thickness: Line thickness
            
        Returns:
            Frame with polygon drawn on it
        """
        result = frame.copy()
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(result, [points], True, color, thickness)
        return result
    
    @staticmethod
    def draw_filled_polygon(frame: np.ndarray, points: List[Tuple[int, int]],
                          color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
        """
        Draw a semi-transparent filled polygon on a frame.
        
        Args:
            frame: Input frame
            points: List of (x, y) coordinates of polygon vertices
            color: Polygon color in BGR
            alpha: Transparency factor (0.0 to 1.0)
            
        Returns:
            Frame with filled polygon drawn on it
        """
        result = frame.copy()
        overlay = result.copy()
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        return result
    
    @staticmethod
    def overlay_image(frame: np.ndarray, overlay: np.ndarray, position: Tuple[int, int],
                     scale: float = 1.0, alpha: float = 1.0) -> np.ndarray:
        """
        Overlay an image on a frame at the specified position.
        
        Args:
            frame: Input frame
            overlay: Image to overlay
            position: (x, y) coordinates of the top-left corner of the overlay
            scale: Scale factor for the overlay image
            alpha: Transparency factor (0.0 to 1.0)
            
        Returns:
            Frame with image overlaid on it
        """
        result = frame.copy()
        
        # Resize overlay if needed
        if scale != 1.0:
            h, w = overlay.shape[:2]
            overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)))
        
        h, w = overlay.shape[:2]
        x, y = position
        
        # Check if overlay is partially outside the frame
        if x >= result.shape[1] or y >= result.shape[0] or x + w <= 0 or y + h <= 0:
            return result  # Overlay is completely outside the frame
        
        # Calculate the valid region for overlay
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(result.shape[1], x + w)
        y_end = min(result.shape[0], y + h)
        
        # Calculate the corresponding region in the overlay image
        overlay_x_start = max(0, -x)
        overlay_y_start = max(0, -y)
        overlay_x_end = overlay_x_start + (x_end - x_start)
        overlay_y_end = overlay_y_start + (y_end - y_start)
        
        # Get the region of interest in the result frame
        roi = result[y_start:y_end, x_start:x_end]
        
        # Get the region of interest in the overlay image
        overlay_roi = overlay[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]
        
        # Check if overlay has an alpha channel
        if overlay_roi.shape[2] == 4:
            # Extract RGB and alpha channels
            overlay_rgb = overlay_roi[:, :, :3]
            overlay_alpha = overlay_roi[:, :, 3] / 255.0 * alpha
            
            # Create alpha mask
            alpha_mask = np.dstack((overlay_alpha, overlay_alpha, overlay_alpha))
            
            # Blend the images
            roi[:] = (1 - alpha_mask) * roi + alpha_mask * overlay_rgb
        else:
            # Simple alpha blending for RGB images
            cv2.addWeighted(overlay_roi, alpha, roi, 1 - alpha, 0, roi)
        
        return result