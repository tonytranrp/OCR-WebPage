import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union

def convert_color(image: np.ndarray, conversion_code: int) -> np.ndarray:
    """
    Convert image color space.
    
    Args:
        image: Input image
        conversion_code: OpenCV color conversion code (e.g., cv2.COLOR_BGR2GRAY)
        
    Returns:
        Converted image
    """
    return cv2.cvtColor(image, conversion_code)

def apply_threshold(image: np.ndarray, threshold: int = 127, max_value: int = 255, 
                   threshold_type: int = cv2.THRESH_BINARY) -> Tuple[float, np.ndarray]:
    """
    Apply thresholding to an image.
    
    Args:
        image: Input image (should be grayscale)
        threshold: Threshold value
        max_value: Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV types
        threshold_type: OpenCV thresholding type
        
    Returns:
        Tuple containing the threshold value used and the thresholded image
    """
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = convert_color(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(image, threshold, max_value, threshold_type)

def apply_adaptive_threshold(image: np.ndarray, max_value: int = 255, 
                           adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           threshold_type: int = cv2.THRESH_BINARY,
                           block_size: int = 11, constant: int = 2) -> np.ndarray:
    """
    Apply adaptive thresholding to an image.
    
    Args:
        image: Input image (should be grayscale)
        max_value: Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV types
        adaptive_method: Adaptive thresholding method
        threshold_type: OpenCV thresholding type
        block_size: Size of a pixel neighborhood that is used to calculate a threshold value
        constant: Constant subtracted from the mean or weighted mean
        
    Returns:
        Thresholded image
    """
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = convert_color(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, constant)

def apply_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
              blur_type: str = 'gaussian', sigma: int = 0) -> np.ndarray:
    """
    Apply blur to an image.
    
    Args:
        image: Input image
        kernel_size: Size of the blurring kernel
        blur_type: Type of blur ('gaussian', 'median', 'box')
        sigma: Standard deviation of the Gaussian kernel
        
    Returns:
        Blurred image
    """
    if blur_type == 'gaussian':
        return cv2.GaussianBlur(image, kernel_size, sigma)
    elif blur_type == 'median':
        k = max(kernel_size[0], kernel_size[1])
        return cv2.medianBlur(image, k if k % 2 == 1 else k + 1)  # Must be odd
    elif blur_type == 'box':
        return cv2.blur(image, kernel_size)
    else:
        raise ValueError(f"Unsupported blur type: {blur_type}")

def apply_morphology(image: np.ndarray, operation: int, kernel_size: Tuple[int, int] = (5, 5), 
                    iterations: int = 1) -> np.ndarray:
    """
    Apply morphological operation to an image.
    
    Args:
        image: Input image
        operation: OpenCV morphological operation (e.g., cv2.MORPH_OPEN)
        kernel_size: Size of the structuring element
        iterations: Number of times to apply the operation
        
    Returns:
        Processed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, operation, kernel, iterations=iterations)

def detect_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150, 
                aperture_size: int = 3, l2gradient: bool = False) -> np.ndarray:
    """
    Detect edges in an image using the Canny edge detector.
    
    Args:
        image: Input image
        low_threshold: Lower threshold for the hysteresis procedure
        high_threshold: Upper threshold for the hysteresis procedure
        aperture_size: Aperture size for the Sobel operator
        l2gradient: Flag indicating whether to use the L2 norm
        
    Returns:
        Edge map
    """
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = convert_color(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size, L2gradient=l2gradient)

def enhance_contrast(image: np.ndarray, clip_limit: float = 3.0, 
                    tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Enhance the contrast of an image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    if len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        # For grayscale images
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
    
    return enhanced

def enhance_text_readability(image: np.ndarray, sharpen: bool = True, denoise: bool = True) -> np.ndarray:
    """
    Enhance image for better text recognition, especially for printed text on paper.
    
    Args:
        image: Input image
        sharpen: Whether to apply sharpening
        denoise: Whether to apply denoising
        
    Returns:
        Enhanced image optimized for OCR
    """
    # Make a copy to avoid modifying the original
    result = image.copy()
    
    # Convert to grayscale if it's a color image
    is_color = len(result.shape) == 3 and result.shape[2] == 3
    if is_color:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result.copy()
    
    # Apply bilateral filter to remove noise while preserving edges
    if denoise:
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding to handle different lighting conditions
    # This works especially well for printed text on paper
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to clean up the text
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Sharpen the image to make text more defined
    if sharpen:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        binary = cv2.filter2D(binary, -1, kernel)
    
    # If original was color, convert back to color for consistent output
    if is_color:
        # Create a 3-channel image from the binary image
        enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Blend with original to maintain some color information
        alpha = 0.7  # Blend factor
        enhanced = cv2.addWeighted(enhanced, alpha, result, 1-alpha, 0)
    else:
        enhanced = binary
    
    return enhanced

def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None, 
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio if only one dimension is specified.
    
    Args:
        image: Input image
        width: Target width (None to calculate from height)
        height: Target height (None to calculate from width)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        # Calculate width to maintain aspect ratio
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    elif height is None:
        # Calculate height to maintain aspect ratio
        aspect_ratio = h / w
        height = int(width * aspect_ratio)
    
    return cv2.resize(image, (width, height), interpolation=interpolation)

def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Crop a region from an image.
    
    Args:
        image: Input image
        x: X-coordinate of the top-left corner
        y: Y-coordinate of the top-left corner
        width: Width of the crop region
        height: Height of the crop region
        
    Returns:
        Cropped image
    """
    return image[y:y+height, x:x+width]

def rotate_image(image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None, 
                scale: float = 1.0) -> np.ndarray:
    """
    Rotate an image around its center.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive values mean counter-clockwise rotation)
        center: Center of rotation (None for the center of the image)
        scale: Isotropic scale factor
        
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, rotation_matrix, (w, h))

def draw_contours(image: np.ndarray, contours: List[np.ndarray], color: Tuple[int, int, int] = (0, 255, 0), 
                 thickness: int = 2) -> np.ndarray:
    """
    Draw contours on an image.
    
    Args:
        image: Input image
        contours: List of contours to draw
        color: Contour color in BGR
        thickness: Line thickness
        
    Returns:
        Image with contours
    """
    result = image.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    return result

def find_contours(image: np.ndarray, retrieval_mode: int = cv2.RETR_EXTERNAL, 
                 approximation_method: int = cv2.CHAIN_APPROX_SIMPLE) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Find contours in a binary image.
    
    Args:
        image: Input image (should be binary)
        retrieval_mode: Contour retrieval mode
        approximation_method: Contour approximation method
        
    Returns:
        Tuple containing the contours and hierarchy
    """
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = convert_color(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # OpenCV 4.x returns contours, hierarchy
    contours, hierarchy = cv2.findContours(binary, retrieval_mode, approximation_method)
    return contours, hierarchy