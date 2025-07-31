import numpy as np
from scipy.ndimage import gaussian_filter


def create_gaussian_defect(center, size, sigma, image_shape):
    """
    Create a single gaussian defect at specified location
    
    Args:
        center: (x, y) center position of defect
        size: (height, width) size of defect region
        sigma: gaussian sigma value (can be tuple for different x,y sigma)
        image_shape: (height, width) shape of full image
        
    Returns:
        defect_image: full image with gaussian defect (0-1)
    """
    h, w = image_shape
    defect_image = np.zeros((h, w), dtype=np.float32)
    
    # Create coordinate grids
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Calculate gaussian
    cx, cy = center
    if isinstance(sigma, (list, tuple)):
        sigma_x, sigma_y = sigma
    else:
        sigma_x = sigma_y = sigma
    
    # Gaussian formula
    gaussian = np.exp(-((X - cx)**2 / (2 * sigma_x**2) + (Y - cy)**2 / (2 * sigma_y**2)))
    
    # Apply size constraint (optional, for more control)
    height, width = size
    mask_x = np.abs(X - cx) <= width / 2
    mask_y = np.abs(Y - cy) <= height / 2
    size_mask = mask_x & mask_y
    
    defect_image = gaussian * size_mask
    
    # Normalize to 0-1
    if defect_image.max() > 0:
        defect_image = defect_image / defect_image.max()
    
    return defect_image


def create_binary_mask(defect_image, threshold=0.1):
    """
    Create binary mask from gaussian defect
    
    Args:
        defect_image: gaussian defect image (0-1)
        threshold: threshold value (default 0.1 = 10%)
        
    Returns:
        binary_mask: binary mask (0 or 1)
    """
    return (defect_image > threshold).astype(np.float32)


def apply_defect_to_background(background, defect_image, intensity):
    """
    Apply defect to background image
    
    Args:
        background: background image (0-255)
        defect_image: gaussian defect (0-1)
        intensity: defect intensity (positive=bright, negative=dark)
        
    Returns:
        output: image with defect applied (same dtype as background)
    """
    output = background + defect_image * intensity
    output = np.clip(output, 0, 255)
    # Keep the same dtype as the input background
    return output.astype(background.dtype)


def generate_multiple_defects(image_shape, defect_params_list):
    """
    Generate multiple defects on single image
    
    Args:
        image_shape: (height, width)
        defect_params_list: list of dict with keys:
            - center: (x, y)
            - size: (height, width)
            - sigma: gaussian sigma
            - intensity: defect intensity
            
    Returns:
        combined_defect: combined defect image
        defect_images: list of individual defect images
    """
    h, w = image_shape
    combined_defect = np.zeros((h, w), dtype=np.float32)
    defect_images = []
    
    for params in defect_params_list:
        defect = create_gaussian_defect(
            center=params['center'],
            size=params['size'],
            sigma=params['sigma'],
            image_shape=image_shape
        )
        defect_images.append(defect)
        
        # Combine defects (for now, use maximum for overlapping)
        # This maintains the strongest defect at each pixel
        combined_defect = np.maximum(combined_defect, defect)
    
    return combined_defect, defect_images


def generate_random_defect_params(image_shape, num_defects, 
                                size_range=(10, 30),
                                sigma_range=(3, 10),
                                intensity_range=(-50, 50)):
    """
    Generate random defect parameters
    
    Args:
        image_shape: (height, width)
        num_defects: number of defects to generate
        size_range: (min, max) size range
        sigma_range: (min, max) sigma range
        intensity_range: (min, max) intensity range
        
    Returns:
        defect_params_list: list of defect parameters
    """
    h, w = image_shape
    defect_params_list = []
    
    for _ in range(num_defects):
        # Random size
        size = np.random.uniform(*size_range)
        
        # Random sigma (proportional to size)
        sigma = size / 3  # Rule of thumb: sigma = size/3
        
        # Random center (ensure defect fits in image)
        margin = size / 2
        center_x = np.random.uniform(margin, w - margin)
        center_y = np.random.uniform(margin, h - margin)
        
        # Random intensity
        intensity = np.random.uniform(*intensity_range)
        
        params = {
            'center': (center_x, center_y),
            'size': (size, size),  # Square defect for now (height, width)
            'sigma': sigma,
            'intensity': intensity
        }
        defect_params_list.append(params)
    
    return defect_params_list