import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable

class TextReplacer:
    """
    A utility for replacing detected text in frames with custom content.
    Supports various replacement strategies including text, images, and custom functions.
    """
    
    def __init__(self, blur_strength: int = 15):
        """
        Initialize the text replacer.
        
        Args:
            blur_strength: Strength of the blur effect when using blur replacement
        """
        self.blur_strength = blur_strength
    
    def replace_with_text(self, frame: np.ndarray, detection: Dict[str, Any], 
                         replacement_text: str, font_scale: float = 0.7,
                         color: Tuple[int, int, int] = (255, 255, 255),
                         bg_color: Optional[Tuple[int, int, int]] = (0, 0, 255),
                         thickness: int = 2) -> np.ndarray:
        """
        Replace detected text with new text.
        
        Args:
            frame: Input frame
            detection: Text detection dictionary from TextDetector
            replacement_text: Text to replace the detected text with
            font_scale: Font scale factor
            color: Text color in BGR
            bg_color: Background color in BGR (None for transparent)
            thickness: Line thickness
            
        Returns:
            Frame with replaced text
        """
        result = frame.copy()
        
        # Get the bounding box coordinates
        x, y, w, h = detection['rect']
        
        # Create a background for the text if specified
        if bg_color is not None:
            cv2.rectangle(result, (x, y), (x + w, y + h), bg_color, -1)
        
        # Calculate text size to center it in the bounding box
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(replacement_text, font_face, font_scale, thickness)
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        
        # Draw the replacement text
        cv2.putText(result, replacement_text, (text_x, text_y), font_face, 
                   font_scale, color, thickness)
        
        return result
    
    def replace_with_blur(self, frame: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
        """
        Replace detected text with a blurred region.
        
        Args:
            frame: Input frame
            detection: Text detection dictionary from TextDetector
            
        Returns:
            Frame with blurred text region
        """
        result = frame.copy()
        
        # Get the bounding box coordinates
        x, y, w, h = detection['rect']
        
        # Extract the region to blur
        region = result[y:y+h, x:x+w]
        
        # Apply blur
        blurred = cv2.GaussianBlur(region, (self.blur_strength, self.blur_strength), 0)
        
        # Replace the region with the blurred version
        result[y:y+h, x:x+w] = blurred
        
        return result
    
    def replace_with_image(self, frame: np.ndarray, detection: Dict[str, Any], 
                          image: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Replace detected text with an image.
        
        Args:
            frame: Input frame
            detection: Text detection dictionary from TextDetector
            image: Replacement image
            alpha: Transparency factor (0.0 to 1.0)
            
        Returns:
            Frame with text replaced by image
        """
        result = frame.copy()
        
        # Get the bounding box coordinates
        x, y, w, h = detection['rect']
        
        # Resize the image to fit the bounding box
        resized_image = cv2.resize(image, (w, h))
        
        # Check if the image has an alpha channel
        if resized_image.shape[2] == 4:
            # Extract RGB and alpha channels
            overlay_rgb = resized_image[:, :, :3]
            overlay_alpha = resized_image[:, :, 3] / 255.0 * alpha
            
            # Create alpha mask
            alpha_mask = np.dstack((overlay_alpha, overlay_alpha, overlay_alpha))
            
            # Extract the region to overlay
            region = result[y:y+h, x:x+w]
            
            # Blend the images
            result[y:y+h, x:x+w] = (1 - alpha_mask) * region + alpha_mask * overlay_rgb
        else:
            # Simple alpha blending for RGB images
            region = result[y:y+h, x:x+w]
            cv2.addWeighted(resized_image, alpha, region, 1 - alpha, 0, region)
            result[y:y+h, x:x+w] = region
        
        return result
    
    def replace_with_function(self, frame: np.ndarray, detection: Dict[str, Any], 
                             replacement_func: Callable[[np.ndarray, Dict[str, Any]], np.ndarray]) -> np.ndarray:
        """
        Replace detected text using a custom function.
        
        Args:
            frame: Input frame
            detection: Text detection dictionary from TextDetector
            replacement_func: Function that takes (frame, detection) and returns modified frame
            
        Returns:
            Frame with text replaced according to the custom function
        """
        return replacement_func(frame, detection)
    
    def replace_all(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                   replacement_strategy: str = 'blur',
                   replacement_text: str = '[REDACTED]',
                   replacement_image: Optional[np.ndarray] = None,
                   replacement_func: Optional[Callable] = None,
                   **kwargs) -> np.ndarray:
        """
        Replace all detected text regions in a frame using the specified strategy.
        
        Args:
            frame: Input frame
            detections: List of text detections from TextDetector
            replacement_strategy: Strategy to use ('text', 'blur', 'image', or 'function')
            replacement_text: Text to use when strategy is 'text'
            replacement_image: Image to use when strategy is 'image'
            replacement_func: Function to use when strategy is 'function'
            **kwargs: Additional arguments for the replacement function
            
        Returns:
            Frame with all text regions replaced
        """
        result = frame.copy()
        
        for detection in detections:
            if replacement_strategy == 'text':
                result = self.replace_with_text(result, detection, replacement_text, **kwargs)
            elif replacement_strategy == 'blur':
                result = self.replace_with_blur(result, detection)
            elif replacement_strategy == 'image' and replacement_image is not None:
                result = self.replace_with_image(result, detection, replacement_image, **kwargs)
            elif replacement_strategy == 'function' and replacement_func is not None:
                result = self.replace_with_function(result, detection, replacement_func)
        
        return result