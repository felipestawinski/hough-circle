# Hough Circle Detection from Scratch

A complete implementation of the Hough Transform for circle detection without using OpenCV's built-in `HoughCircles` function.

## 🎯 Features

- **Complete Hough Transform Implementation**: Built from mathematical foundations using only basic OpenCV and NumPy functions
- **Edge Detection**: Implements Canny edge detection as preprocessing step
- **Configurable Parameters**: Easily adjust radius range, detection threshold, and sensitivity
- **Non-Maximum Suppression**: Removes overlapping duplicate circles
- **Detailed Visualization**: Shows original image, edges, and detected circles with confidence scores
- **Comprehensive Output**: Displays circle parameters and saves results

## 🚀 Usage

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the program:
```bash
python main.py
```

The program will process `img/circles.jpg` and display the results in a matplotlib window, plus save the output as `detected_circles.jpg`.

## 🔬 How It Works

### Mathematical Foundation

The Hough Transform for circles is based on the circle equation:
```
(x - a)² + (y - b)² = r²
```

### Algorithm Steps

1. **Edge Detection**: Apply Canny edge detection to find object boundaries
2. **Accumulator Space**: Create 3D accumulator space (x, y, radius)
3. **Voting Process**: For each edge pixel, vote for possible circle centers
4. **Peak Detection**: Find local maxima in accumulator space above threshold
5. **Non-Maximum Suppression**: Remove overlapping duplicate detections
6. **Visualization**: Draw detected circles on original image

### Key Implementation Details

- **Optimized Voting**: Uses adaptive sampling based on radius size
- **Memory Efficient**: Separate 2D accumulators for each radius
- **Robust Detection**: Local maximum filtering with neighborhood suppression
- **Configurable Range**: Designed for ~200px radius circles but easily adjustable

Where `(a, b)` is the circle center and `r` is the radius.

For each edge pixel `(x, y)`, we can find all possible circle centers:
```
a = x - r * cos(θ)
b = y - r * sin(θ)
```

Where `θ` varies from 0 to 2π.

### Algorithm Steps

1. **Image Preprocessing**
   - Convert to grayscale
   - Apply bilateral filtering (noise reduction + edge preservation)
   - Histogram equalization for contrast enhancement
   - Gaussian blur for smoothing
   - Adaptive Canny edge detection

2. **Hough Transform Voting**
   - Create 3D accumulator array `[height, width, num_radii]`
   - For each edge pixel and radius combination:
     - Calculate all possible circle centers using trigonometry
     - Vote in accumulator for each valid center

3. **Peak Detection**
   - Find local maxima in accumulator above threshold
   - Extract circle parameters `(x, y, radius, confidence)`

4. **Non-Maximum Suppression**
   - Remove overlapping detections
   - Keep highest confidence circles

## 🚀 Usage

### Basic Usage

```python
from main import HoughCircleDetector

# Load image
image = cv.imread('your_image.jpg')

# Create detector
detector = HoughCircleDetector(
    min_radius=10,
    max_radius=100,
    accumulator_threshold=25,
    angle_step=8
)

# Detect circles
circles, edges = detector.detect_circles(image)

# Results: list of (x, y, radius, votes) tuples
```

### Running the Complete Program

```bash
python main.py
```

This will:
- Load `img/bicicleta.jpg`
- Detect circles automatically
- Display results in windows
- Save output images

## ⚙️ Configuration Parameters

| Parameter | Description | Typical Values | Effect |
|-----------|-------------|----------------|--------|
| `min_radius` | Minimum circle radius | 10-20 | Smaller = detect tiny circles |
| `max_radius` | Maximum circle radius | 50-100 | Larger = detect big circles |
| `accumulator_threshold` | Minimum votes for detection | 15-50 | Lower = more sensitive |
| `angle_step` | Angular resolution (degrees) | 5-10 | Smaller = more accurate but slower |

### Parameter Tuning Guide

**For small circles** (coins, buttons):
```python
detector = HoughCircleDetector(min_radius=5, max_radius=30, accumulator_threshold=15)
```

**For large circles** (wheels, plates):
```python
detector = HoughCircleDetector(min_radius=30, max_radius=150, accumulator_threshold=35)
```

**For maximum sensitivity**:
```python
detector = HoughCircleDetector(accumulator_threshold=10, angle_step=5)
```

**For maximum speed**:
```python
detector = HoughCircleDetector(angle_step=15, max_radius=50)
```