import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Try to import scipy for faster processing, fallback if not available
try:
    from scipy.ndimage import maximum_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy not available, using fallback methods")

def canny_edge_detection(image, low_threshold=50, high_threshold=150, scale_factor=1.0):
    """
    Apply Canny edge detection to the image with optional downscaling for speed
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Optionally downscale for faster processing
    if scale_factor != 1.0:
        new_width = int(gray.shape[1] * scale_factor)
        new_height = int(gray.shape[0] * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height))
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Scale back up if we downscaled
    if scale_factor != 1.0:
        original_height, original_width = image.shape[:2]
        edges = cv2.resize(edges, (original_width, original_height))
    
    return edges

def hough_circle_transform(edges, min_radius, max_radius, threshold=100):
    """
    Implement Hough Circle Transform from scratch (optimized version)
    
    Args:
        edges: Binary edge image
        min_radius: Minimum circle radius to detect
        max_radius: Maximum circle radius to detect
        threshold: Minimum votes needed to consider a circle
    
    Returns:
        List of detected circles as (x, y, radius, votes)
    """
    height, width = edges.shape
    
    # Create accumulator space for each radius (use smaller steps for better accuracy)
    radius_step = 1 # Smaller steps for better detection
    radii_to_check = list(range(min_radius, max_radius + 1, radius_step))
    
    accumulators = {}
    for r in radii_to_check:
        accumulators[r] = np.zeros((height, width), dtype=np.int32)
    
    # Find edge pixels
    edge_pixels = np.where(edges > 0)
    edge_y, edge_x = edge_pixels
    
    print(f"Found {len(edge_x)} edge pixels")
    print(f"Checking radii: {radii_to_check}")
    
    # Pre-calculate angles for efficiency (use more angles for better accuracy)
    num_angles = 48
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    # Subsample edge pixels for speed but not too aggressively
    step = max(1, len(edge_x) // 20000)  # Use more edge pixels for better accuracy
    edge_x_sub = edge_x[::step]
    edge_y_sub = edge_y[::step]
    
    print(f"Using {len(edge_x_sub)} subsampled edge pixels (step={step})")
    
    # For each edge pixel, vote for possible circle centers
    for x, y in zip(edge_x_sub, edge_y_sub):
        # For each possible radius
        for r in radii_to_check:
            # Calculate possible center coordinates using pre-calculated angles
            center_x = x - r * cos_angles
            center_y = y - r * sin_angles
            
            # Convert to integers and check bounds
            center_x_int = center_x.astype(int)
            center_y_int = center_y.astype(int)
            
            # Vectorized bounds checking
            valid_mask = ((center_x_int >= 0) & (center_x_int < width) & 
                         (center_y_int >= 0) & (center_y_int < height))
            
            # Add votes for valid centers
            valid_x = center_x_int[valid_mask]
            valid_y = center_y_int[valid_mask]
            
            for cx, cy in zip(valid_x, valid_y):
                accumulators[r][cy, cx] += 1
    
    
    # Find circles by looking for peaks in accumulator space
    circles = []
    
    for r in radii_to_check:
        acc = accumulators[r]
        max_votes = np.max(acc)
        print(f"Radius {r}: max votes = {max_votes}")
        
        # Find peaks above threshold using a more efficient method
        # Apply a simple maximum filter for faster peak detection
        if HAS_SCIPY:
            # Use scipy if available for faster processing
            local_maxima = maximum_filter(acc, size=max(3, r//20)) == acc
            peaks = np.where((acc >= threshold) & local_maxima)
        else:
            # Fallback to manual method if scipy not available
            peaks = np.where(acc >= threshold)
            
        peak_y, peak_x = peaks
        
        for x, y in zip(peak_x, peak_y):
            votes = acc[y, x]
            
            # Quick local maximum check (smaller neighborhood for speed)
            neighborhood_size = max(3, r // 20)
            
            # Check if this is a local maximum in a smaller neighborhood
            y_min = max(0, y - neighborhood_size)
            y_max = min(height, y + neighborhood_size + 1)
            x_min = max(0, x - neighborhood_size)
            x_max = min(width, x + neighborhood_size + 1)
            
            local_region = acc[y_min:y_max, x_min:x_max]
            
            if votes == np.max(local_region):
                circles.append((x, y, r, votes))
    
    # Sort circles by number of votes (confidence)
    circles.sort(key=lambda c: c[3], reverse=True)
    
    print(f"Total circles found before filtering: {len(circles)}")
    for i, (x, y, r, votes) in enumerate(circles[:15]):  # Show top 15
        print(f"  Circle {i+1}: Center=({x}, {y}), Radius={r}, Votes={votes}")
    
    # Remove overlapping circles (non-maximum suppression) - more efficient
    filtered_circles = []
    for circle in circles:
        x, y, r, votes = circle
        
        # Check if this circle overlaps significantly with any already accepted circle
        is_unique = True
        for existing in filtered_circles:
            ex, ey, er, _ = existing
            distance = np.sqrt((x - ex)**2 + (y - ey)**2)
            
            # Make non-maximum suppression less aggressive for close circles
            if distance < max(r, er) * 0.5 and abs(r - er) < max(r, er) * 0.3:
                is_unique = False
                break
        
        if is_unique:
            filtered_circles.append(circle)
            print(f"Accepted circle: Center=({x}, {y}), Radius={r}, Votes={votes}")
        else:
            print(f"Rejected circle: Center=({x}, {y}), Radius={r}, Votes={votes} (too close to existing)")
            
        # Increase limit to detect all circles in your image
        if len(filtered_circles) >= 15:  # Increased from 10 to 15
            break
    
    return filtered_circles

def draw_circles(image, circles, color=(0, 0, 255), thickness=5):
    """
    Draw detected circles on the image
    """
    result = image.copy()
    
    for i, (x, y, r, votes) in enumerate(circles):
        # Draw circle
        cv2.circle(result, (x, y), r, color, thickness)
        # Draw center point
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
        # Add text with circle info
        text = f"R={r}, V={votes}"
        cv2.putText(result, text, (x - 50, y - r - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result

def main():
    # Load the image
    img_name = "bicicleta.jpg"
    image_path = f"img/{img_name}"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Loaded image with shape: {image.shape}")
    
    # Apply edge detection without downscaling to maintain accuracy
    print("Applying edge detection...")
    edges = canny_edge_detection(image, low_threshold=30, high_threshold=100)  # Lower thresholds for better edge detection
    
    # Parameters for circle detection
    # circulos
    # min_radius = 95
    # max_radius = 105
    # threshold = 10  # Lower threshold to catch all circles
    
    # bicicleta
    # min_radius = 160
    # max_radius = 170
    # threshold = 13  # Lower threshold to catch all circles
    
    #roda gigante
    min_radius = 160
    max_radius = 170
    threshold = 13  # Lower threshold to catch all circles
    
    print(f"Detecting circles with radius {min_radius}-{max_radius}px...")
    
    # Apply Hough Circle Transform
    circles = hough_circle_transform(edges, min_radius, max_radius, threshold)
    
    print(f"Detected {len(circles)} circles")
    
    # If no circles detected, try with lower threshold
    if len(circles) == 0:
        print("No circles detected, trying with lower threshold...")
        circles = hough_circle_transform(edges, min_radius, max_radius, threshold//2)
        print(f"Detected {len(circles)} circles with lower threshold")
    
    # Print details of detected circles
    for i, (x, y, r, votes) in enumerate(circles):
        print(f"Circle {i+1}: Center=({x}, {y}), Radius={r}, Votes={votes}")
    
    # Draw circles on the original image
    result = draw_circles(image, circles)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # Edge detection result
    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    plt.axis('off')
    
    # Final result with detected circles
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Circles ({len(circles)} found)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Also save the result
    output_path = f"results/{img_name}"
    cv2.imwrite(output_path, result)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()