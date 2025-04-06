import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import argparse

class EDLinesDetector:
    def __init__(self, gradient_threshold=2, anchor_threshold=1, 
                 min_line_length=5, nfa_epsilon=1.0):
        """
        Initialize the EDLines detector with parameters
        
        Parameters:
        -----------
        gradient_threshold : int
            Threshold for gradient magnitude to be considered an edge pixel
        anchor_threshold : int
            Threshold for considering a pixel as an anchor in Edge Drawing
        min_line_length : int
            Minimum number of pixels to consider a line segment
        nfa_epsilon : float
            Threshold for Number of False Alarms (NFA), typically 1.0
        """
        self.gradient_threshold = gradient_threshold
        self.anchor_threshold = anchor_threshold
        self.min_line_length = min_line_length
        self.nfa_epsilon = nfa_epsilon
        
        # For storing timing information
        self.timing = {
            'preprocessing': 0,
            'edge_drawing': 0,
            'line_fitting': 0,
            'validation': 0,
            'total': 0
        }
    
    def detect(self, image_path):
        """
        Apply EDLines detection to an image
        
        Parameters:
        -----------
        image_path : str
            Path to the input image
            
        Returns:
        --------
        edges_image : np.ndarray
            Image with detected edges (black on white)
        lines_image : np.ndarray
            Image with detected line segments
        line_segments : list
            List of detected line segments as (x1, y1, x2, y2)
        """
        # Record total time
        total_start_time = time.time()
        
        # Load and preprocess image
        preprocess_start = time.time()
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0.5)
        
        # Compute gradients
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        # Compute gradient direction (in range [0, pi])
        direction = np.mod(np.arctan2(gy, gx) + np.pi, np.pi)
        
        self.timing['preprocessing'] = time.time() - preprocess_start
        
        # Apply Edge Drawing algorithm
        edge_drawing_start = time.time()
        edge_segments = self._edge_drawing_improved(magnitude, direction)
        
        # Create a white image for edges with the same dimensions as the input image
        edge_map = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

        # Create a copy of the original image for line visualization
        lines_image = edge_map.copy()
        
        # Draw edge segments on the white image with single-pixel black lines
        for segment in edge_segments:
            if len(segment) < 2:
                continue
                
            # Convert segment to array of points for drawing
            points = np.array(segment, dtype=np.int32)
            
            # Connect points with lines to ensure continuity
            for i in range(len(points) - 1):
                p1 = (points[i][0], points[i][1])
                p2 = (points[i+1][0], points[i+1][1])
                cv2.line(edge_map, p1, p2, 0, 1)  # Draw black (0) thin lines
        
        # Convert edge map to color image for visualization consistency
        edges_image = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        
        self.timing['edge_drawing'] = time.time() - edge_drawing_start
        
        # Fit line segments to edge segments
        line_fitting_start = time.time()
        candidate_lines = self._fit_lines(edge_segments)
        self.timing['line_fitting'] = time.time() - line_fitting_start
        
        # Validate lines using a-contrario approach
        validation_start = time.time()
        valid_lines = self._validate_lines(candidate_lines, magnitude.shape)
        self.timing['validation'] = time.time() - validation_start
        
        # Visualize the line segments
        for line in valid_lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        self.timing['total'] = time.time() - total_start_time
        
        return edges_image, lines_image, valid_lines
    
    def _edge_drawing_improved(self, magnitude, direction):
        """
        Improved Edge Drawing algorithm implementation
        
        Parameters:
        -----------
        magnitude : np.ndarray
            Gradient magnitude image
        direction : np.ndarray
            Gradient direction image
            
        Returns:
        --------
        edge_segments : list
            List of edge segments (list of points)
        """
        height, width = magnitude.shape
        
        # Step 1: Create edge map using gradient magnitude
        edge_map = np.zeros_like(magnitude, dtype=np.uint8)
        edge_map[magnitude > self.gradient_threshold] = 255
        
        # Step 2: Perform non-maximum suppression to thin edges
        nms_edge_map = np.zeros_like(edge_map)
        
        # Define 8 direction vectors (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        # Convert direction from radians to indices (0-7)
        dir_indices = np.round(direction * 8 / np.pi) % 8
        dir_indices = dir_indices.astype(np.int32)
        
        # Pad magnitude for easier neighbor access
        padded_mag = np.pad(magnitude, 1, mode='constant')
        
        # Apply non-maximum suppression
        for y in range(1, height + 1):
            for x in range(1, width + 1):
                if padded_mag[y, x] <= self.gradient_threshold:
                    continue
                    
                dir_idx = dir_indices[y-1, x-1]
                dx_pos, dy_pos = directions[dir_idx]
                dx_neg, dy_neg = directions[(dir_idx + 4) % 8]  # Opposite direction
                
                # Check if it's a local maximum along the gradient direction
                if (padded_mag[y, x] >= padded_mag[y + dy_pos, x + dx_pos] and
                    padded_mag[y, x] >= padded_mag[y + dy_neg, x + dx_neg]):
                    nms_edge_map[y-1, x-1] = 255
        
        # Step 3: Find anchor points (strong edges)
        anchors = np.zeros_like(nms_edge_map)
        anchors[nms_edge_map == 255] = 255  # Start with all edge pixels
        
        # Only keep pixels with magnitude above anchor threshold
        anchors[magnitude < self.anchor_threshold] = 0
        
        # Step 4: Link edges from anchor points
        visited = np.zeros_like(nms_edge_map, dtype=np.uint8)
        edge_segments = []
        
        # Process each anchor point
        for y in range(height):
            for x in range(width):
                if anchors[y, x] == 0 or visited[y, x] > 0:
                    continue
                
                # Start a new edge segment
                segment = []
                stack = [(y, x)]
                visited[y, x] = 1
                
                while stack:
                    cy, cx = stack.pop()
                    segment.append([cx, cy])  # Note: x,y order for drawing
                    
                    # Find the best continuation pixel
                    best_y, best_x = None, None
                    best_value = -1
                    
                    # Check 8-connected neighbors
                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < height and 0 <= nx < width and 
                            nms_edge_map[ny, nx] > 0 and visited[ny, nx] == 0):
                            
                            # Use gradient magnitude as priority
                            if magnitude[ny, nx] > best_value:
                                best_y, best_x = ny, nx
                                best_value = magnitude[ny, nx]
                    
                    # Add best continuation point to stack
                    if best_y is not None:
                        stack.append((best_y, best_x))
                        visited[best_y, best_x] = 1
                
                # Add segment if it's long enough
                if len(segment) >= self.min_line_length:
                    edge_segments.append(segment)
        
        return edge_segments
    
    def _fit_lines(self, edge_segments):
        """
        Fit line segments to edge segments using least squares
        
        Parameters:
        -----------
        edge_segments : list
            List of edge segments (list of points)
            
        Returns:
        --------
        candidate_lines : list
            List of candidate line segments as (x1, y1, x2, y2, length)
        """
        candidate_lines = []
        
        for segment in edge_segments:
            if len(segment) < self.min_line_length:
                continue
                
            # Convert segment to numpy array
            points = np.array(segment)
            x = points[:, 0]
            y = points[:, 1]
            
            # Fit a line using least squares
            if len(x) > 1:  # Need at least 2 points
                # Calculate mean values
                mean_x = np.mean(x)
                mean_y = np.mean(y)
                
                # Calculate linear regression parameters
                sum_xy = np.sum((x - mean_x) * (y - mean_y))
                sum_xx = np.sum((x - mean_x) ** 2)
                
                if sum_xx != 0:  # Avoid division by zero
                    slope = sum_xy / sum_xx
                    intercept = mean_y - slope * mean_x
                    
                    # Calculate endpoints of the line segment
                    x1, x2 = int(min(x)), int(max(x))
                    y1 = int(slope * x1 + intercept)
                    y2 = int(slope * x2 + intercept)
                    
                    # Calculate segment length
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Add to candidate lines if long enough
                    if length >= self.min_line_length:
                        candidate_lines.append((x1, y1, x2, y2, length))
                else:
                    # Vertical line case
                    y1, y2 = int(min(y)), int(max(y))
                    x1 = x2 = int(mean_x)
                    length = y2 - y1
                    
                    if length >= self.min_line_length:
                        candidate_lines.append((x1, y1, x2, y2, length))
        
        # Sort candidates by length (descending)
        candidate_lines.sort(key=lambda x: x[4], reverse=True)
        
        return candidate_lines
    
    def _validate_lines(self, candidate_lines, image_shape):
        """
        Validate line segments using a-contrario approach
        
        Parameters:
        -----------
        candidate_lines : list
            List of candidate line segments as (x1, y1, x2, y2, length)
        image_shape : tuple
            Shape of the image (height, width)
            
        Returns:
        --------
        valid_lines : list
            List of validated line segments as (x1, y1, x2, y2)
        """
        valid_lines = []
        
        # Calculate the number of tests (approximated)
        height, width = image_shape
        num_pixels = height * width
        num_tests = num_pixels * (num_pixels - 1) // 2  # Approximate maximum number of possible line segments
        
        # Precision parameter (typical value)
        precision = 0.1  # radians (about 5.7 degrees)
        
        for x1, y1, x2, y2, length in candidate_lines:
            # Calculate NFA based on segment length
            # Probability of a point having the right orientation under the uniform background model
            p = 2 * precision / np.pi
            
            # NFA formula: num_tests * (p)^(length-1)
            nfa = num_tests * (p ** (length - 1))
            
            # Accept line if NFA is below threshold
            if nfa <= self.nfa_epsilon:
                valid_lines.append((x1, y1, x2, y2))
        
        return valid_lines
    
    def print_timing(self):
        """Print timing information for each step"""
        print("\nTiming information:")
        print(f"  Preprocessing:  {self.timing['preprocessing']*1000:.2f} ms")
        print(f"  Edge Drawing:   {self.timing['edge_drawing']*1000:.2f} ms")
        print(f"  Line Fitting:   {self.timing['line_fitting']*1000:.2f} ms")
        print(f"  Validation:     {self.timing['validation']*1000:.2f} ms")
        print(f"  Total:          {self.timing['total']*1000:.2f} ms")


def process_image(image_path, output_dir=None, gradient_threshold=25, 
                  anchor_threshold=8, min_line_length=15, display=True):
    """
    Process an image with EDLines and save/display results
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    output_dir : str, optional
        Directory to save output images
    gradient_threshold : int
        Threshold for gradient magnitude
    anchor_threshold : int
        Threshold for anchor points
    min_line_length : int
        Minimum length of line segments
    display : bool
        Whether to display results
    """
    # Create output directory if needed
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize detector
    detector = EDLinesDetector(
        gradient_threshold=gradient_threshold, 
        anchor_threshold=anchor_threshold,
        min_line_length=min_line_length
    )
    
    try:
        # Process image
        print(f"Processing {image_path}...")
        edges_image, lines_image, lines = detector.detect(image_path)
        
        # Print timing information
        detector.print_timing()
        
        # Save results if output directory is provided
        if output_dir:
            img_name = Path(image_path).stem
            cv2.imwrite(str(output_path / f"{img_name}_edges.jpg"), edges_image)
            cv2.imwrite(str(output_path / f"{img_name}_lines.jpg"), lines_image)
            print(f"Results saved to {output_path}")
        
        # Display results if requested
        if display:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(edges_image, cv2.COLOR_BGR2RGB))
            plt.title('Edge Drawing Result')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB))
            plt.title('Line Segments')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        print(f"Detected {len(lines)} line segments.")
        return edges_image, lines_image, lines
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description='EDLines: Line Segment Detector')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output images')
    parser.add_argument('--gradient_threshold', type=int, default=36, help='Gradient magnitude threshold')
    parser.add_argument('--anchor_threshold', type=int, default=8, help='Anchor point threshold')
    parser.add_argument('--min_line_length', type=int, default=10, help='Minimum line segment length')
    parser.add_argument('--no_display', action='store_true', help='Do not display results')
    
    args = parser.parse_args()
    
    process_image(
        args.image_path,
        args.output_dir,
        args.gradient_threshold,
        args.anchor_threshold,
        args.min_line_length,
        not args.no_display
    )


if __name__ == "__main__":
    main()