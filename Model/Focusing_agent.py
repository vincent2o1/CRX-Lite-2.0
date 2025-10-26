# Focusing agent to capture the spots in images

import cv2
import numpy as np
import matplotlib.pyplot as plt
import zlib
import base64
import os
import json
from datetime import datetime
import hashlib
from skimage.metrics import structural_similarity as ssim
import time
import struct

# Global storage for agent movements and spot IDs
agent_movements = {}
spot_ids = {}

def image_to_pixels(image_path):
    """
    Convert an image to pixel values and resize it to 64x64, preserving color.
    """
    image = cv2.imread(image_path)  # Default is color (BGR)
    if image is None:
        raise ValueError("Image not found or invalid format.")

    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    return image

def visualize_agent_movements(image, movements, title="Agent Movements"):
    """
    Visualizes agent movements on a resized image.

    Parameters:
        image (numpy array): The input grayscale image.
        movements (list of tuples): List of (y, x) coordinates.
        title (str): Title for the plot.
    """
    original_h, original_w = image.shape[:2]  # Original image size
    resized_h, resized_w = 64, 64  # Target resized size

    # Resize image
    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # Scale movement coordinates to match resized image
    scaled_movements = [(int(y * (resized_h / original_h)), int(x * (resized_w / original_w))) for y, x in movements]

    # Extract x and y values for plotting
    y_vals, x_vals = zip(*scaled_movements)

    # Plot resized image
    plt.figure(figsize=(6, 6))
    plt.imshow(resized_image, cmap='gray')

    # Plot movements with proper scaling
    plt.plot(x_vals, y_vals, marker='o', color='red', linestyle='-')

    plt.title(title)
    plt.show()

def visualize_spots(image, movements, focus_size=(16, 16), save_path="focused_spots.png"):
    """
    Visualizes focused spots in a grid format and saves the visualization as a PNG file.

    Parameters:
    - image: 2D numpy array (grayscale image)
    - movements: List of (y, x) coordinates for the focus spots
    - focus_size: Tuple (height, width) defining the size of each focused region
    - save_path: File path to save the output image
    """
    focus_h, focus_w = focus_size
    num_spots = len(movements)

    if num_spots == 0:
        print("No focused spots to visualize.")
        return

    cols = min(4, num_spots)
    rows = (num_spots + cols - 1) // cols  # Compute required rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if rows == 1:
        axes = np.array([axes])  # Ensure it's an array for consistency

    for i, ax in enumerate(axes.flatten()):
        if i < num_spots:
            y, x = movements[i]
            spot = image[y:y+focus_h, x:x+focus_w]
            ax.imshow(spot, cmap='gray')
            ax.set_title(f"Spot {i+1}")
            ax.axis('off')
        else:
            ax.axis('off')  # Hide unused subplots

    plt.suptitle("Focused Spots", fontsize=14)
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as {save_path}")
    plt.close()  # Close the figure to free memory

def list_to_reversible_id(pixel_list):
    array = np.array(pixel_list, dtype=np.uint8)
    compressed = zlib.compress(array.tobytes())
    encoded = base64.urlsafe_b64encode(compressed).decode()
    return encoded

def id_to_list(image_id, original_length):
    compressed = base64.urlsafe_b64decode(image_id)
    image_bytes = zlib.decompress(compressed)
    pixel_list = np.frombuffer(image_bytes, dtype=np.uint8).tolist()
    return pixel_list[:original_length]

def calculate_variance(region):
    return np.var(region)

def is_unique_spot(y, x, image, focus_h, focus_w):
    spot = image[y:y+focus_h, x:x+focus_w].flatten().tolist()
    spot_id = list_to_reversible_id(spot)
    if spot_id in spot_ids:
        return False
    spot_ids[spot_id] = (y, x)
   # save_spot_ids_to_drive(spot_id)
    return True

def generate_importance_map(image, image1):
    """
    Generate an importance map using color thresholding to identify colored objects
    on a dark grid background.

    Args:
        image: Input color image (BGR)

    Returns:
        Binary importance map where 1 indicates important regions
    """
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract saturation and value channels
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Identify colored regions (high saturation or high value)
    # Black grid will have low saturation and low value
    colored_pixels = np.logical_or(s > 50, v > 80).astype(np.uint8)

    # Clean up the map using morphological operations
    kernel = np.ones((2, 2), np.uint8)
    clean_map = cv2.morphologyEx(colored_pixels, cv2.MORPH_OPEN, kernel)
    clean_map = cv2.dilate(clean_map, kernel, iterations=1)

    # Remove isolated pixels using connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        clean_map, connectivity=8)

    # Filter out small regions
    final_map = np.zeros_like(clean_map)
    for i in range(1, num_labels):  # Skip label 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= 10:
            final_map[labels == i] = 1

    return final_map

def determine_optimal_focus_size(image, image1):
    """
    Quickly determine optimal focus size based on image content analysis.
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Use blob detection to estimate object sizes
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 25
    params.maxArea = 500
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    if len(keypoints) > 0:
        # Calculate average blob size
        sizes = [kp.size for kp in keypoints]
        avg_size = int(np.mean(sizes))

        # Use average size with constraints
        size = max(16, min(32, avg_size))
        return (size, size)
    else:
        # Default size if no blobs detected
        return (16, 16)

def detect_objects_parallel_with_training(image, image1, importance_map, focus_size, similarity_threshold, channels, model=None):
    """
    Detect objects in the image with user-guided training to learn which objects are useful.
    Finds objects that differ between image and image1, searching the entire second image.
    Allows user feedback to train a model on what constitutes a "good" object.
    Only considers regions highlighted in the importance map.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    from skimage.measure import label, regionprops

    height, width = image.shape[:2]
   # height1, width1 = image1.shape[:2]
    focus_h, focus_w = focus_size

    # Initialize model if needed
    if model is None:
        model = {'positive_examples': [], 'negative_examples': []}

    # Process the importance map to find contiguous object regions
    # Label the connected components in the importance map
    labeled_importance = label(importance_map)
    props = regionprops(labeled_importance)

    # Store candidate objects
    candidate_objects = []

    # For each connected region in the importance map
    for prop in props:
        # Get bounding box of the region
        min_row, min_col, max_row, max_col = prop.bbox

        # Ensure the bounding box has reasonable dimensions
        box_height = max_row - min_row
        box_width = max_col - min_col

        # Skip very small regions
        if box_height < 3 or box_width < 3:
            continue

        # Adjust box dimensions to match focus size or be slightly larger
        # This helps ensure we capture the full object
        if box_height < focus_h:
            # Center the box vertically
            center_row = (min_row + max_row) // 2
            min_row = max(0, center_row - focus_h // 2)
            max_row = min(height, min_row + focus_h)
            # Adjust min_row if max_row hit the boundary
            if max_row == height:
                min_row = max(0, height - focus_h)

        if box_width < focus_w:
            # Center the box horizontally
            center_col = (min_col + max_col) // 2
            min_col = max(0, center_col - focus_w // 2)
            max_col = min(width, min_col + focus_w)
            # Adjust min_col if max_col hit the boundary
            if max_col == width:
                min_col = max(0, width - focus_w)

        # Extract the region
        region = image[min_row:max_row, min_col:max_col]

        # Verify this region is significant (contains non-background pixels)
        if channels == 3:
            # For color images, check if there are significant non-black pixels
            if np.sum(np.max(region, axis=2) > 20) < (region.shape[0] * region.shape[1] * 0.05):
                continue
        else:
            # For grayscale
            if np.sum(region > 20) < (region.shape[0] * region.shape[1] * 0.05):
                continue

        # Check uniqueness against already found objects
        is_unique = True
        for existing_region, _, _ in candidate_objects:
            # If regions are different sizes, resize the smaller one for comparison
            if existing_region.shape != region.shape:
                continue  # Skip comparison if sizes differ too much

            # Compare regions
            if channels == 3:
                try:
                    similarities = [ssim(region[:,:,c], existing_region[:,:,c], data_range=255)
                                 for c in range(channels)]
                    similarity = np.mean(similarities)
                except:
                    similarity = 0
            else:
                try:
                    similarity = ssim(region, existing_region, data_range=255)
                except:
                    similarity = 0

            if similarity > similarity_threshold:
                is_unique = False
                break

        # If unique, add to candidates
        if is_unique:
            candidate_objects.append((region, (min_row, min_col), (max_row - min_row, max_col - min_col)))

    # Present each candidate to the user
    objects = []

    for region, (y, x), (h, w) in candidate_objects:
        print(f"Found potential object at position ({y}, {x}), size: {h}x{w}")

        # Show the object
        plt.figure(figsize=(6, 6))

        # Display the full image with rectangle around the object
        plt.subplot(2, 1, 1)
        plt.imshow(image if channels == 3 else image, cmap='gray' if channels == 1 else None)
        plt.gca().add_patch(plt.Rectangle((x, y), w, h,
                                        edgecolor='red', facecolor='none', linewidth=2))
        plt.title("Image with Detected Object")

        # Display the zoomed object
        plt.subplot(2, 1, 2)
        plt.imshow(region if channels == 3 else region,
                 cmap='gray' if channels == 1 else None)
        plt.title("Detected Object")

        plt.tight_layout()
        plt.show()

        # Extract features for this object
        features = extract_features(region, channels)

        # Predict based on previous examples if we have enough data
        prediction = None
        if len(model['positive_examples']) > 3 and len(model['negative_examples']) > 3:
            prediction = predict_object_usefulness(features, model)
            print(f"Model prediction: {'Useful' if prediction else 'Not useful'}")

        # Ask for user feedback
        response = input("Is this a useful object? (y/n): ").strip().lower()

        if response == 'y':
            objects.append((region, (y, x)))
            # Add to positive examples
            model['positive_examples'].append(features)
            print("Object added to collection.")
        else:
            # Add to negative examples
            model['negative_examples'].append(features)
            print("Object ignored.")

    return objects, model

def extract_features(region, channels):
    """
    Extract enhanced features from the region for model training.
    """
    import numpy as np
    features = []

    # Calculate basic statistics
    if channels == 3:
        # For color images
        for c in range(channels):
            features.append(np.mean(region[:,:,c]))  # Mean color value
            features.append(np.std(region[:,:,c]))   # Color standard deviation

        # Color distribution features
        unique_colors, color_counts = np.unique(region.reshape(-1, channels), axis=0, return_counts=True)
        features.append(len(unique_colors))  # Number of unique colors
        features.append(np.max(color_counts) / np.sum(color_counts))  # Dominant color ratio
    else:
        # For grayscale images
        features.append(np.mean(region))
        features.append(np.std(region))

        # Value distribution features
        unique_values, value_counts = np.unique(region, return_counts=True)
        features.append(len(unique_values))  # Number of unique values
        features.append(np.max(value_counts) / np.sum(value_counts))  # Dominant value ratio

    # Pattern complexity features
    edge_h = np.sum(np.abs(np.diff(region, axis=0)))
    edge_v = np.sum(np.abs(np.diff(region, axis=1)))
    features.append(edge_h)  # Horizontal edge strength
    features.append(edge_v)  # Vertical edge strength

    # Symmetry features
    h, w = region.shape[:2]
    if channels == 3:
        # Ensure both halves are the same width
        mid_w = w // 2
        left = region[:, :mid_w, :]
        right = region[:, -mid_w:, :]  # Always take the last mid_w columns
        h_sym = np.mean([
            np.mean(np.abs(left[:, :, c] - np.fliplr(right[:, :, c])))
            for c in range(channels)
        ])

        mid_h = h // 2
        top = region[:mid_h, :, :]
        bottom = region[-mid_h:, :, :]
        v_sym = np.mean([
            np.mean(np.abs(top[:, :, c] - np.flipud(bottom[:, :, c])))
            for c in range(channels)
        ])
    else:
        mid_w = w // 2
        left = region[:, :mid_w]
        right = region[:, -mid_w:]
        h_sym = np.mean(np.abs(left - np.fliplr(right)))

        mid_h = h // 2
        top = region[:mid_h, :]
        bottom = region[-mid_h:, :]
        v_sym = np.mean(np.abs(top - np.flipud(bottom)))

    features.append(h_sym)  # Horizontal symmetry
    features.append(v_sym)  # Vertical symmetry

    return np.array(features)

def predict_object_usefulness(features, model):
    """
    Predict if an object is useful based on previously labeled examples.
    Improved with weighted distance calculation.
    """
    import numpy as np

    if not model['positive_examples'] or not model['negative_examples']:
        return None

    # Convert to numpy arrays for easier calculation
    pos_examples = np.array(model['positive_examples'])
    neg_examples = np.array(model['negative_examples'])

    # Normalize features for better comparison
    feature_means = np.mean(np.vstack([pos_examples, neg_examples]), axis=0)
    feature_stds = np.std(np.vstack([pos_examples, neg_examples]), axis=0)
    feature_stds[feature_stds == 0] = 1  # Avoid division by zero

    normalized_features = (features - feature_means) / feature_stds
    normalized_pos = (pos_examples - feature_means) / feature_stds
    normalized_neg = (neg_examples - feature_means) / feature_stds

    # Calculate distances to positive and negative examples
    pos_distances = np.mean([np.linalg.norm(normalized_features - ex) for ex in normalized_pos])
    neg_distances = np.mean([np.linalg.norm(normalized_features - ex) for ex in normalized_neg])

    # Return prediction based on which set is closer
    return pos_distances < neg_distances

def detect_objects_parallel_with_training_(output_image, input_image, importance_map, focus_size,
                                         similarity_threshold, channels, input_region=None, model=None):
    """
    Modified to detect how a specific input object appears in the output image.
    Now strongly prioritizes color matching first, then shape.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    from skimage.measure import label, regionprops
    import cv2

    height, width = output_image.shape[:2]
    focus_h, focus_w = focus_size

    # Initialize model if needed
    if model is None:
        model = {'positive_examples': [], 'negative_examples': []}

    if input_region is None:
        return [], model

    # STEP 1: EXTRACT DOMINANT COLORS FROM INPUT OBJECT
    # ------------------------------------------------
    # First identify the significant colors in the input region (ignoring black/background)
    input_significant_pixels = []
    if channels == 3:
        # For RGB images
        # Create a mask for non-black pixels
        mask = np.max(input_region, axis=2) > 30
        # Get the RGB values for these pixels
        for c in range(channels):
            if np.sum(mask) > 0:
                input_significant_pixels.append(input_region[:,:,c][mask])
            else:
                input_significant_pixels.append(np.array([0]))
    else:
        # For grayscale
        mask = input_region > 30
        if np.sum(mask) > 0:
            input_significant_pixels.append(input_region[mask])
        else:
            input_significant_pixels.append(np.array([0]))

    # Find the dominant color(s) in the input region
    input_dominant_colors = []
    for channel_pixels in input_significant_pixels:
        # Use histogram to find most common values
        if len(channel_pixels) > 10:
            hist, bins = np.histogram(channel_pixels, bins=25, range=(0, 255))
            # Get the most common color values
            for i in range(min(3, len(hist))):  # Take up to 3 dominant values
                if np.max(hist) > 0:
                    bin_idx = np.argmax(hist)
                    color_value = (bins[bin_idx] + bins[bin_idx+1]) / 2
                    input_dominant_colors.append(color_value)
                    # Zero out this peak to find the next one
                    hist[max(0, bin_idx-1):min(len(hist), bin_idx+2)] = 0
        else:
            # If few pixels, use the mean
            input_dominant_colors.append(np.mean(channel_pixels))

    # STEP 2: COLOR-BASED SEARCH ACROSS THE ENTIRE OUTPUT IMAGE
    # --------------------------------------------------------
    # Look for regions with similar color profile across the entire output image
    color_matches = []

    # Create a sliding window search
    window_step = max(2, min(focus_h, focus_w) // 4)  # Smaller step for more thorough search

    for y in range(0, height - focus_h + 1, window_step):
        for x in range(0, width - focus_w + 1, window_step):
            # Extract region
            region = output_image[y:y+focus_h, x:x+focus_w]

            # Skip if region is mostly background/black
            if channels == 3:
                if np.sum(np.max(region, axis=2) > 30) < (focus_h * focus_w * 0.05):
                    continue
            else:
                if np.sum(region > 30) < (focus_h * focus_w * 0.05):
                    continue

            # Check for color similarity with input dominant colors
            region_significant_pixels = []
            if channels == 3:
                mask = np.max(region, axis=2) > 30
                for c in range(channels):
                    if np.sum(mask) > 0:
                        region_significant_pixels.append(region[:,:,c][mask])
                    else:
                        region_significant_pixels.append(np.array([0]))
            else:
                mask = region > 30
                if np.sum(mask) > 0:
                    region_significant_pixels.append(region[mask])
                else:
                    region_significant_pixels.append(np.array([0]))

            # Find dominant colors in this region
            region_dominant_colors = []
            for channel_pixels in region_significant_pixels:
                if len(channel_pixels) > 10:
                    hist, bins = np.histogram(channel_pixels, bins=25, range=(0, 255))
                    for i in range(min(3, len(hist))):
                        if np.max(hist) > 0:
                            bin_idx = np.argmax(hist)
                            color_value = (bins[bin_idx] + bins[bin_idx+1]) / 2
                            region_dominant_colors.append(color_value)
                            hist[max(0, bin_idx-1):min(len(hist), bin_idx+2)] = 0
                else:
                    region_dominant_colors.append(np.mean(channel_pixels))

            # Calculate color similarity
            color_match_score = 0
            if channels == 3 and len(input_dominant_colors) >= 3 and len(region_dominant_colors) >= 3:
                # RGB color distance
                input_color = np.array(input_dominant_colors[:3])
                region_color = np.array(region_dominant_colors[:3])
                color_distance = np.sqrt(np.sum((input_color - region_color)**2))
                # Normalize to 0-1 range
                color_match_score = 1 - min(1.0, color_distance / 441.7)  # 441.7 = sqrt(3 * 255^2)
            elif len(input_dominant_colors) > 0 and len(region_dominant_colors) > 0:
                # Grayscale or single channel match
                color_distance = abs(input_dominant_colors[0] - region_dominant_colors[0])
                color_match_score = 1 - min(1.0, color_distance / 255.0)

            # If color match is good, store this region
            if color_match_score > 0.85:  # High threshold for color match
                color_matches.append((region, (y, x), (focus_h, focus_w), color_match_score))

    # STEP 3: SHAPE MATCHING
    # ---------------------
    # For each color match, now evaluate shape similarity
    candidate_objects = []

    for region, (y, x), (h, w), color_score in color_matches:
        # Convert both input and region to binary to focus on shape
        if channels == 3:
            input_binary = (np.max(input_region, axis=2) > 30).astype(np.uint8)
            region_binary = (np.max(region, axis=2) > 30).astype(np.uint8)
        else:
            input_binary = (input_region > 30).astype(np.uint8)
            region_binary = (region > 30).astype(np.uint8)

        # Try to match shapes
        try:
            # If sizes differ, resize input to match region
            if input_binary.shape != region_binary.shape:
                resized_input = cv2.resize(input_binary,
                                         (region_binary.shape[1], region_binary.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
            else:
                resized_input = input_binary

            # Calculate shape similarity
            shape_score = ssim(resized_input, region_binary, data_range=1)

            # Calculate final score: heavily weighted toward color match
            final_score = 0.7 * color_score + 0.3 * shape_score

            if final_score > similarity_threshold * 0.7:
                candidate_objects.append((region, (y, x), (h, w), final_score))
        except:
            # If shape matching fails, still include based on color alone if very strong match
            if color_score > 0.95:
                candidate_objects.append((region, (y, x), (h, w), color_score * 0.7))

    # Sort candidates by similarity score
    candidate_objects.sort(key=lambda x: x[3], reverse=True)

    # Limit to top matches
    candidate_objects = candidate_objects[:10]

    # Present each candidate to the user
    objects = []

    for region, (y, x), (h, w), similarity in candidate_objects:
        print(f"Found potential match at position ({y}, {x}), size: {h}x{w}, similarity: {similarity:.3f}")

        # Show the object
        plt.figure(figsize=(10, 6))

        # Display the input region we're looking for
        plt.subplot(2, 2, 1)
        plt.imshow(input_region if channels == 3 else input_region, cmap='gray' if channels == 1 else None)
        plt.title("Input Object")

        # Display the full output image with rectangle around the object
        plt.subplot(2, 2, 2)
        plt.imshow(output_image if channels == 3 else output_image, cmap='gray' if channels == 1 else None)
        plt.gca().add_patch(plt.Rectangle((x, y), w, h,
                                        edgecolor='red', facecolor='none', linewidth=2))
        plt.title("Output Image with Matched Object")

        # Display the zoomed object
        plt.subplot(2, 2, 3)
        plt.imshow(region if channels == 3 else region, cmap='gray' if channels == 1 else None)
        plt.title(f"Matched Object (Similarity: {similarity:.3f})")

        plt.tight_layout()
        plt.show()

        # Ask for user feedback
        response = input("Is this a correct match for the input object? (y/n): ").strip().lower()

        if response == 'y':
            objects.append((region, (y, x)))
            # Add to positive examples
            features = extract_features(region, channels)
            model['positive_examples'].append(features)
            print("Object added to collection.")
        else:
            # Add to negative examples
            features = extract_features(region, channels)
            model['negative_examples'].append(features)
            print("Object ignored.")

    return objects, model

def find_best_match(input_region, output_image, focus_size, similarity_threshold, channels, output_importance_map=None):
    """
    Find the best matching position of the input region in the output image.
    Uses connected components analysis, shape descriptors, and improved similarity metrics.
    Handles objects of different shapes and transformations.
    """
    import numpy as np
    import cv2
    from skimage.metrics import structural_similarity as ssim
    from skimage import measure

    focus_h, focus_w = focus_size
    best_match_pos = None
    best_similarity = 0
    transformation_data = {}

    # Step 1: Use connected components to extract each colored object
    def extract_objects(image, mask=None):
        if channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if mask is None else mask
        else:
            gray = image if mask is None else mask

        # Threshold to create binary image
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        # Find connected components
        labels = measure.label(binary, connectivity=2)
        regions = measure.regionprops(labels)

        # Extract each object (component) with area greater than min_size
        min_size = 25  # Minimum size to consider as valid object
        objects = []

        for region in regions:
            if region.area >= min_size:
                # Get object mask and bounding box
                minr, minc, maxr, maxc = region.bbox
                obj_mask = labels[minr:maxr, minc:maxc] == region.label

                # Get the actual object from the original image
                if channels == 3:
                    obj = np.zeros((maxr-minr, maxc-minc, 3), dtype=np.uint8)
                    for c in range(3):
                        obj[:,:,c] = image[minr:maxr, minc:maxc, c] * obj_mask
                else:
                    obj = image[minr:maxr, minc:maxc] * obj_mask

                objects.append({
                    'image': obj,
                    'mask': obj_mask.astype(np.uint8) * 255,
                    'bbox': (minr, minc, maxr, maxc),
                    'area': region.area
                })

        return objects

    # Step 2: Calculate Hu Moments and binary template matching
    def calculate_shape_descriptors(obj_data):
        mask = obj_data['mask']
        # Calculate Hu Moments
        moments = cv2.moments(mask)

        # Handle case where moments don't exist (empty contour)
        if moments['m00'] == 0:
            return {
                'hu_moments': np.zeros(7),
                'centroid': (0, 0),
                'orientation': 0,
                'aspect_ratio': 1.0
            }

        # Calculate Hu Moments
        hu_moments = cv2.HuMoments(moments).flatten()

        # Sign-transform Hu moments for better matching
        for i in range(7):
            if hu_moments[i] != 0:
                hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]))

        # Calculate additional shape metrics
        centroid_x = moments['m10'] / moments['m00']
        centroid_y = moments['m01'] / moments['m00']

        # Calculate orientation using central moments
        if (moments['mu20'] - moments['mu02']) != 0:
            orientation = 0.5 * np.arctan2(2 * moments['mu11'],
                                          (moments['mu20'] - moments['mu02']))
        else:
            orientation = 0

        # Find contours for aspect ratio
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and len(contours[0]) > 5:
            try:
                # Fit ellipse to get aspect ratio
                ellipse = cv2.fitEllipse(contours[0])
                aspect_ratio = max(ellipse[1]) / (min(ellipse[1]) + 1e-5)  # Avoid div by 0
            except:
                aspect_ratio = 1.0
        else:
            aspect_ratio = 1.0

        return {
            'hu_moments': hu_moments,
            'centroid': (centroid_x, centroid_y),
            'orientation': orientation,
            'aspect_ratio': aspect_ratio
        }

    # Step 3: Measure shape similarity between objects
    def compare_shapes(shape1, shape2):
        # Compare Hu moments
        hu_distance = cv2.matchShapes(shape1['hu_moments'].reshape(7, 1),
                                      shape2['hu_moments'].reshape(7, 1),
                                      cv2.CONTOURS_MATCH_I3, 0)

        # Calculate orientation similarity (normalized from 0-1, where 1 is perfect match)
        # Angle difference: map to [0, π/2] and normalize to [0, 1]
        angle_diff = abs(shape1['orientation'] - shape2['orientation']) % np.pi
        if angle_diff > np.pi/2:
            angle_diff = np.pi - angle_diff
        orientation_sim = 1 - (angle_diff / (np.pi/2))

        # Calculate aspect ratio similarity
        aspect_ratio_sim = min(shape1['aspect_ratio'], shape2['aspect_ratio']) / max(shape1['aspect_ratio'], shape2['aspect_ratio'])

        # Combine metrics with weights
        # Lower hu_distance is better, so we need to transform it (e^(-d) gives 1 for perfect match and near 0 for poor match)
        hu_similarity = np.exp(-5 * hu_distance)  # The coefficient 5 controls sensitivity

        # Weighted combination
        shape_similarity = 0.6 * hu_similarity + 0.2 * orientation_sim + 0.2 * aspect_ratio_sim

        return shape_similarity

    # Handle rotation without shape mismatch
    def rotate_image(img, angle):
        """Rotate image by angle degrees and preserve dimensions appropriately"""
        if channels == 3:
            h, w, c = img.shape
            # Create output image with the SAME dimensions as input
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
            return rotated
        else:
            h, w = img.shape
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
            return rotated

    # Get overall binary mask of input region
    if channels == 3:
        input_mask = (np.max(input_region, axis=2) > 20).astype(np.uint8) * 255
    else:
        input_mask = (input_region > 20).astype(np.uint8) * 255

    # Extract objects from input region
    input_objects = extract_objects(input_region, input_mask)

    if not input_objects:
        # Fallback for when no distinct objects found - treat entire region as one object
        input_objects = [{
            'image': input_region,
            'mask': input_mask,
            'bbox': (0, 0, input_region.shape[0], input_region.shape[1]),
            'area': np.sum(input_mask > 0)
        }]

    # Compute descriptors for input objects
    for obj in input_objects:
        obj['descriptors'] = calculate_shape_descriptors(obj)

    focus_h, focus_w = focus_size
    best_match_pos = None
    best_similarity = 0
    transformation_data = {}

    # Use the modified detection function to find matches for our specific input_region
    detected_objects, _ = detect_objects_parallel_with_training_(
        output_image,
        None,  # We're passing input_region directly
        output_importance_map,
        (focus_h, focus_w),
        similarity_threshold,
        channels,
        input_region=input_region
    )

    # Now use detected object positions as search positions
    search_positions = []
    for object_region, (y, x) in detected_objects:
        search_positions.append((y, x))

    # Process each candidate position
    for y, x in search_positions:
        try:
            if y + focus_h > output_image.shape[0] or x + focus_w > output_image.shape[1]:
                continue

            output_region = output_image[y:y+focus_h, x:x+focus_w]

            # Get binary mask for output region
            if channels == 3:
                output_mask = (np.max(output_region, axis=2) > 20).astype(np.uint8) * 255
            else:
                output_mask = (output_region > 20).astype(np.uint8) * 255

            # Extract objects from output region
            output_objects = extract_objects(output_region, output_mask)

            if not output_objects:
                # Fallback when no distinct objects found
                output_objects = [{
                    'image': output_region,
                    'mask': output_mask,
                    'bbox': (0, 0, output_region.shape[0], output_region.shape[1]),
                    'area': np.sum(output_mask > 0)
                }]

            # Compute descriptors for output objects
            for obj in output_objects:
                obj['descriptors'] = calculate_shape_descriptors(obj)

            # Try to match the global shape first for efficiency
            if input_mask.shape == output_mask.shape:
                iou = np.sum(np.logical_and(input_mask > 0, output_mask > 0)) / max(1, np.sum(np.logical_or(input_mask > 0, output_mask > 0)))
                if iou < 0.2:  # Skip if overall shapes are very different
                    continue

            # Compare objects between input and output regions
            match_scores = []
            for input_obj in input_objects:
                best_obj_score = 0

                for output_obj in output_objects:
                    # Skip if there's a large size difference
                    size_ratio = min(input_obj['area'], output_obj['area']) / (max(input_obj['area'], output_obj['area']) + 1e-5)
                    if size_ratio < 0.2:  # More permissive size ratio
                        continue

                    # Compare shapes
                    shape_similarity = compare_shapes(input_obj['descriptors'], output_obj['descriptors'])

                    # Skip detailed comparison if shape is very different
                    if shape_similarity < 0.2:  # More permissive shape threshold
                        continue

                    try:
                        # Calculate IoU (Intersection over Union)
                        if input_obj['mask'].shape != output_obj['mask'].shape:
                            # Handle different shapes by using contour matching
                            input_contours, _ = cv2.findContours(input_obj['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            output_contours, _ = cv2.findContours(output_obj['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            if input_contours and output_contours:
                                # Match shapes using contours
                                contour_match = cv2.matchShapes(input_contours[0], output_contours[0], cv2.CONTOURS_MATCH_I3, 0)
                                contour_similarity = np.exp(-5 * contour_match)

                                # Use contour similarity instead of overlap for different sized objects
                                overlap = contour_similarity
                            else:
                                # Fallback if contours can't be found
                                overlap = 0.3  # Default fallback
                        else:
                            # Direct comparison if same size
                            intersection = np.logical_and(input_obj['mask'] > 0, output_obj['mask'] > 0).sum()
                            union = np.logical_or(input_obj['mask'] > 0, output_obj['mask'] > 0).sum()
                            overlap = intersection / (union + 1e-5)

                        # Calculate color similarity
                        try:
                            if channels == 3:
                                # Normalize histograms for different sized objects
                                i_hist = cv2.calcHist([input_obj['image']], [0, 1, 2], input_obj['mask'], [8, 8, 8], [0, 256, 0, 256, 0, 256])
                                i_hist = i_hist / (np.sum(i_hist) + 1e-5)

                                o_hist = cv2.calcHist([output_obj['image']], [0, 1, 2], output_obj['mask'], [8, 8, 8], [0, 256, 0, 256, 0, 256])
                                o_hist = o_hist / (np.sum(o_hist) + 1e-5)

                                color_sim = cv2.compareHist(i_hist, o_hist, cv2.HISTCMP_CORREL)
                            else:
                                i_hist = cv2.calcHist([input_obj['image']], [0], input_obj['mask'], [16], [0, 256])
                                i_hist = i_hist / (np.sum(i_hist) + 1e-5)

                                o_hist = cv2.calcHist([output_obj['image']], [0], output_obj['mask'], [16], [0, 256])
                                o_hist = o_hist / (np.sum(o_hist) + 1e-5)

                                color_sim = cv2.compareHist(i_hist, o_hist, cv2.HISTCMP_CORREL)

                            if np.isnan(color_sim):
                                color_sim = 0.5  # Default if calculation fails
                        except:
                            color_sim = 0.5  # Default if histogram fails

                        # Combined object match score - weighted for shape and structure
                        obj_score = 0.5 * shape_similarity + 0.3 * overlap + 0.2 * color_sim

                        if obj_score > best_obj_score:
                            best_obj_score = obj_score

                    except Exception as e:
                        continue

                if best_obj_score > 0:
                    match_scores.append(best_obj_score)

            # Overall region match score
            if match_scores:
                # Weight by number of matched objects
                match_fraction = len(match_scores) / len(input_objects)

                # Take weighted average of individual match scores
                match_scores.sort(reverse=True)
                weights = np.linspace(1.0, 0.5, len(match_scores))  # Linear weighting
                weighted_scores = np.multiply(match_scores, weights)
                mean_score = np.sum(weighted_scores) / np.sum(weights)

                # Combine with match fraction
                region_score = 0.7 * mean_score + 0.3 * match_fraction

                # Add importance map bonus if applicable
                if output_importance_map is not None:
                    importance_region = output_importance_map[y:y+focus_h, x:x+focus_w]
                    importance_score = np.sum(importance_region) / (focus_h * focus_w * 255)
                    region_score = 0.85 * region_score + 0.15 * min(1.0, importance_score * 10)

                # Check if this is the best match so far
                if region_score > best_similarity:
                    best_similarity = region_score
                    best_match_pos = (y, x)

                    # Record transformation data
                    transformation_data["object_count"] = len(match_scores)
                    transformation_data["transformed"] = best_similarity < 0.8  # Lower scores suggest transformation

        except Exception as e:
            continue

    # Try rotated versions to handle transformations if needed
    if best_similarity < 0.7 and best_match_pos is not None:
        y, x = best_match_pos
        output_region = output_image[y:y+focus_h, x:x+focus_w]

        best_rotation = 0
        for angle in [90, 180, 270]:
            try:
                # Use proper rotation that preserves dimensions
                rotated_input = rotate_image(input_region, angle)

                # Get mask for rotated input
                if channels == 3:
                    rotated_mask = (np.max(rotated_input, axis=2) > 20).astype(np.uint8) * 255
                else:
                    rotated_mask = (rotated_input > 20).astype(np.uint8) * 255

                # Extract objects from rotated input
                rotated_objects = extract_objects(rotated_input, rotated_mask)

                if not rotated_objects:
                    rotated_objects = [{
                        'image': rotated_input,
                        'mask': rotated_mask,
                        'bbox': (0, 0, rotated_input.shape[0], rotated_input.shape[1]),
                        'area': np.sum(rotated_mask > 0)
                    }]

                # Compute descriptors for rotated objects
                for obj in rotated_objects:
                    obj['descriptors'] = calculate_shape_descriptors(obj)

                # Compare objects
                rot_match_scores = []
                for rot_obj in rotated_objects:
                    best_obj_score = 0
                    for output_obj in output_objects:
                        # Shape comparison
                        shape_similarity = compare_shapes(rot_obj['descriptors'], output_obj['descriptors'])

                        if shape_similarity < 0.2:
                            continue

                        try:
                            # Calculate overlap using contours for differently sized objects
                            if rot_obj['mask'].shape != output_obj['mask'].shape:
                                rot_contours, _ = cv2.findContours(rot_obj['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                out_contours, _ = cv2.findContours(output_obj['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                if rot_contours and out_contours:
                                    contour_match = cv2.matchShapes(rot_contours[0], out_contours[0], cv2.CONTOURS_MATCH_I3, 0)
                                    contour_similarity = np.exp(-5 * contour_match)
                                    overlap = contour_similarity
                                else:
                                    overlap = 0.3
                            else:
                                intersection = np.logical_and(rot_obj['mask'] > 0, output_obj['mask'] > 0).sum()
                                union = np.logical_or(rot_obj['mask'] > 0, output_obj['mask'] > 0).sum()
                                overlap = intersection / (union + 1e-5)

                            # Color similarity
                            try:
                                if channels == 3:
                                    r_hist = cv2.calcHist([rot_obj['image']], [0, 1, 2], rot_obj['mask'], [8, 8, 8], [0, 256, 0, 256, 0, 256])
                                    r_hist = r_hist / (np.sum(r_hist) + 1e-5)

                                    o_hist = cv2.calcHist([output_obj['image']], [0, 1, 2], output_obj['mask'], [8, 8, 8], [0, 256, 0, 256, 0, 256])
                                    o_hist = o_hist / (np.sum(o_hist) + 1e-5)

                                    color_sim = cv2.compareHist(r_hist, o_hist, cv2.HISTCMP_CORREL)
                                else:
                                    r_hist = cv2.calcHist([rot_obj['image']], [0], rot_obj['mask'], [16], [0, 256])
                                    r_hist = r_hist / (np.sum(r_hist) + 1e-5)

                                    o_hist = cv2.calcHist([output_obj['image']], [0], output_obj['mask'], [16], [0, 256])
                                    o_hist = o_hist / (np.sum(o_hist) + 1e-5)

                                    color_sim = cv2.compareHist(r_hist, o_hist, cv2.HISTCMP_CORREL)

                                if np.isnan(color_sim):
                                    color_sim = 0.5
                            except:
                                color_sim = 0.5

                            obj_score = 0.5 * shape_similarity + 0.3 * overlap + 0.2 * color_sim

                            if obj_score > best_obj_score:
                                best_obj_score = obj_score
                        except Exception as e:
                            continue

                    if best_obj_score > 0:
                        rot_match_scores.append(best_obj_score)

                # Calculate overall score for this rotation
                if rot_match_scores:
                    # Consider match fraction for rotated objects
                    rot_match_fraction = len(rot_match_scores) / len(rotated_objects)

                    rot_match_scores.sort(reverse=True)
                    weights = np.linspace(1.0, 0.5, len(rot_match_scores))
                    weighted_scores = np.multiply(rot_match_scores, weights)
                    rot_mean_score = np.sum(weighted_scores) / np.sum(weights)

                    rot_score = 0.7 * rot_mean_score + 0.3 * rot_match_fraction

                    if rot_score > best_similarity:
                        best_similarity = rot_score
                        best_rotation = angle
                        transformation_data["rotation"] = angle
            except Exception as e:
                continue

    # Handle special case - if no good match found but there must be one
    if (best_match_pos is None or best_similarity < 0.4) and similarity_threshold < 0.5:
        # When objects are heavily transformed, use simpler matching as fallback
        best_fallback_score = 0
        best_fallback_pos = None

        for y, x in search_positions:
            try:
                if y + focus_h > output_image.shape[0] or x + focus_w > output_image.shape[1]:
                    continue

                output_region = output_image[y:y+focus_h, x:x+focus_w]

                # Use simple presence of non-background pixels as indicator
                if channels == 3:
                    input_pixels = np.sum(np.max(input_region, axis=2) > 20)
                    output_pixels = np.sum(np.max(output_region, axis=2) > 20)
                else:
                    input_pixels = np.sum(input_region > 20)
                    output_pixels = np.sum(output_region > 20)

                # Simple density comparison
                if input_pixels > 0 and output_pixels > 0:
                    density_ratio = min(input_pixels, output_pixels) / max(input_pixels, output_pixels)

                    if density_ratio > best_fallback_score:
                        best_fallback_score = density_ratio
                        best_fallback_pos = (y, x)
            except:
                continue

        # Use fallback if it's better than our current best
        if best_fallback_pos is not None and (best_match_pos is None or best_fallback_score > 0.5):
            best_match_pos = best_fallback_pos
            best_similarity = max(best_similarity, best_fallback_score * 0.5)  # Scale fallback score
            transformation_data["fallback_match"] = True

    return best_match_pos, best_similarity, transformation_data

def analyze_transformation(image, image1, input_region, input_pos, output_pos,
                          focus_size, similarity_threshold, channels, transformation_data):
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    focus_h, focus_w = focus_size
    input_y, input_x = input_pos
    output_y, output_x = output_pos

    # Initialize result data
    transformation_types = []
    explanation_spots = []
    description = ""
    detailed_description = ""

    dx = output_x - input_x
    dy = output_y - input_y
    distance = np.sqrt(dx**2 + dy**2)
    output_spot = image1[output_y:output_y+focus_h, output_x:output_x+focus_w]

    # Rule: Nearby objects
    search_radius = max(focus_h, focus_w) * 2
    nearby_objects = []

    for y in range(max(0, output_y-search_radius), min(height, output_y+search_radius+1), focus_h//2):
        for x in range(max(0, output_x-search_radius), min(width, output_x+search_radius+1), focus_w//2):
            if abs(y-output_y) < focus_h//2 and abs(x-output_x) < focus_w//2:
                continue
            if y+focus_h > height or x+focus_w > width:
                continue

            nearby_region = image1[y:y+focus_h, x:x+focus_w]

            if channels == 3:
                rgb_variance = np.sum([np.var(nearby_region[:,:,c]) for c in range(channels)])
                if rgb_variance > 100:
                    nearby_objects.append((nearby_region, (y, x)))
            else:
                if np.var(nearby_region) > 50:
                    nearby_objects.append((nearby_region, (y, x)))

    attraction_detected = False
    if nearby_objects:
        closest_y, closest_x = min(nearby_objects, key=lambda obj:
            np.sqrt((obj[1][0]-output_y)**2 + (obj[1][1]-output_x)**2))[1]

        att_y_start = max(0, min(output_y, closest_y))
        att_x_start = max(0, min(output_x, closest_x))
        att_y_end = min(height, max(output_y+focus_h, closest_y+focus_h))
        att_x_end = min(width, max(output_x+focus_w, closest_x+focus_w))

        if att_y_end - att_y_start >= focus_h and att_x_end - att_x_start >= focus_w:
            was_close_in_input = check_proximity_in_input(
                image, nearby_objects[0][0], input_pos, closest_y, closest_x,
                focus_size, similarity_threshold, channels
            )
            if not was_close_in_input:
                attraction_detected = True
                transformation_types.append("attraction")
                attraction_spot = image1[att_y_start:att_y_end, att_x_start:att_x_end]
                if attraction_spot.shape[:2] != (focus_h, focus_w):
                    attraction_spot = cv2.resize(attraction_spot, (focus_w, focus_h))
                explanation_spots.append(attraction_spot)
                description += "Objects appear to be attracted to each other in the output image. "
                detailed_description += "attracted "

    # Rule: Color matching
    color_matching_detected = False
    if channels == 3:
        input_color_avg = np.mean(input_region, axis=(0,1))
        output_surrounding = image1[max(0, output_y-focus_h):min(height, output_y+2*focus_h),
                                    max(0, output_x-focus_w):min(width, output_x+2*focus_w)]
        surround_color_avg = np.mean(output_surrounding, axis=(0,1))
        color_distance = np.sqrt(np.sum((input_color_avg - surround_color_avg)**2))

        if color_distance < 40:
            color_matching_detected = True
            transformation_types.append("color_matching")
            y_start = max(0, output_y-focus_h//2)
            y_end = min(height, output_y+focus_h+focus_h//2)
            x_start = max(0, output_x-focus_w//2)
            x_end = min(width, output_x+focus_w+focus_w//2)
            color_spot = image1[y_start:y_end, x_start:x_end]
            if color_spot.shape[:2] != (focus_h, focus_w):
                color_spot = cv2.resize(color_spot, (focus_w, focus_h))
            explanation_spots.append(color_spot)
            description += "The object appears to have moved to a region with similar colors. "
            detailed_description += "color_matched "

    # Rule: Fitting into structure
    fitting_detected = False
    output_surrounding = image1[max(0, output_y-focus_h):min(height, output_y+2*focus_h),
                                max(0, output_x-focus_w):min(width, output_x+2*focus_w)]
    if channels == 3:
        surrounding_gray = cv2.cvtColor(output_surrounding, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(surrounding_gray, 50, 150)
    else:
        edges = cv2.Canny(output_surrounding, 50, 150)

    if np.mean(edges) > 20:
        fitting_detected = True
        transformation_types.append("fitting")
        explanation_spots.append(output_spot)
        description += "The object appears to fit into a defined structure or pattern. "
        detailed_description += "fitted to region "

    # Intermediate motion steps
    if distance > focus_h:
        steps = min(3, max(1, int(distance // (focus_h//2))))
        for i in range(1, steps):
            interp_y = int(input_y + (dy * i / steps))
            interp_x = int(input_x + (dx * i / steps))
            if 0 <= interp_y < height-focus_h and 0 <= interp_x < width-focus_w:
                motion_spot = image1[interp_y:interp_y+focus_h, interp_x:interp_x+focus_w]
                explanation_spots.append(motion_spot)

        if not transformation_types:
            if abs(dx) > abs(dy):
                transformation_types.append("horizontal_movement")
                description += "The object has moved horizontally. "
                detailed_description += "moved horizontally "
            else:
                transformation_types.append("vertical_movement")
                description += "The object has moved vertically. "
                detailed_description += "moved vertically "
    elif not transformation_types:
        transformation_types.append("minimal_movement")
        description += "The object has moved minimally. "
        detailed_description += "moved minimally "

    # Rotation
    if "rotation" in transformation_data:
        angle = transformation_data["rotation"]
        transformation_types.append(f"rotation_{angle}")
        description += f"The object has rotated {angle} degrees. "
        detailed_description += f"rotated {angle} degrees "

    # ✅ NEW: Check shape or size change
    def get_contour_area(region):
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if channels == 3 else region
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.contourArea(contours[0]) if contours else 0

    input_area = get_contour_area(input_region)
    output_area = get_contour_area(output_spot)
    shape_or_size_changed = abs(input_area - output_area) > (0.15 * input_area)

    if shape_or_size_changed:
        transformation_types.append("shape_or_size_changed")
        description += "The object's shape or size has changed. "
        detailed_description += "changed size/shape "

    # ✅ NEW: Rule about whether the object should be moved or not
    if (distance < focus_h//3 and not (attraction_detected or color_matching_detected or fitting_detected or shape_or_size_changed)):
        description += "This object should not be moved or changed. "
        detailed_description += "should remain static "
    else:
        description += "This object can be moved. "
        detailed_description += "can be moved "

    # Finalize spot inclusion
    if explanation_spots and not np.array_equal(explanation_spots[-1], output_spot):
        explanation_spots.append(output_spot)
    elif not explanation_spots:
        explanation_spots.append(output_spot)

    if not detailed_description:
        detailed_description = "object transformed with no specific pattern"
    else:
        detailed_description = "object " + detailed_description.strip()

    return {
        "spots": explanation_spots,
        "types": transformation_types,
        "description": description,
        "detailed_description": detailed_description
    }

def check_proximity_in_input(image, nearby_object, input_pos, nearby_y, nearby_x,
                           focus_size, similarity_threshold, channels):
    """
    Check if the nearby object in output was also close in input.
    """
    height, width = image.shape[:2]
    focus_h, focus_w = focus_size
    input_y, input_x = input_pos

    # Check positions around the input object
    for offset_y, offset_x in [(0, focus_w), (0, -focus_w), (focus_h, 0), (-focus_h, 0)]:
        check_y = input_y + offset_y
        check_x = input_x + offset_x

        if 0 <= check_y < height-focus_h+1 and 0 <= check_x < width-focus_w+1:
            input_nearby = image[check_y:check_y+focus_h, check_x:check_x+focus_w]

            # Compare with the nearby object from output
            if channels == 3:
                similarities = [ssim(input_nearby[:,:,c], nearby_object[:,:,c], data_range=255)
                              for c in range(channels)]
                similarity = np.mean(similarities)
            else:
                similarity = ssim(input_nearby, nearby_object, data_range=255)

            if similarity > similarity_threshold:
                return True

    return False

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_final_spots(image, movements, focus_h=32, focus_w=32, title="Final Unique Spots"):
    """
    Visualize the final selected spots on the original image

    Args:
        image: Original image array
        movements: List of (y, x) coordinates of selected spots
        focus_h: Height of focus region (default 32)
        focus_w: Width of focus region (default 32)
        title: Title for the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Display the original image
    ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)

    # Draw rectangles for each selected spot
    colors = plt.cm.Set3(np.linspace(0, 1, len(movements)))

    for i, (y, x) in enumerate(movements):
        # Create rectangle patch
        rect = patches.Rectangle(
            (x, y), focus_w, focus_h,
            linewidth=2,
            edgecolor=colors[i],
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)

        # Add spot number
        ax.text(x + focus_w//2, y + focus_h//2, str(i+1),
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='circle', facecolor=colors[i], alpha=0.7))

    ax.set_title(f"{title} - Total: {len(movements)} spots")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_spots_grid(focused_regions, movements, cols=5, title="Extracted Spots"):
    """
    Display all extracted spots in a grid layout

    Args:
        focused_regions: List of extracted spot images
        movements: List of (y, x) coordinates
        cols: Number of columns in the grid
        title: Title for the plot
    """
    if not focused_regions:
        print("No spots to visualize")
        return

    n_spots = len(focused_regions)
    rows = (n_spots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, (spot, (y, x)) in enumerate(zip(focused_regions, movements)):
        row = i // cols
        col = i % cols

        ax = axes[row, col]
        ax.imshow(spot, cmap='gray' if len(spot.shape) == 2 else None)
        ax.set_title(f'Spot {i+1}\n({y}, {x})', fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for i in range(n_spots, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.suptitle(f"{title} - {n_spots} spots of grid", fontsize=14)
    plt.tight_layout()
    plt.show()

def focusing_agent(image, image1, importance_map=None, importance_map1=None, dual_process=False,
                  request_input=None, training=False, focus_size=(32, 32),
                  similarity_threshold=0.80, testing=False, understanding=False, branch=False):
    """
    Focusing agent that analyzes transformations between input and output images.

    Understanding mode: Returns a sequence of spots showing object transformations
    from input to output, including spots explaining the transformation.

    Parameters:
    - image: Input image (s1)
    - image1: Output image (s2)
    - importance_map: Map highlighting important regions (optional)
    - focus_size: Size of the focus window
    - similarity_threshold: Threshold for determining object similarity
    - understanding: Whether to run in understanding mode

    Returns:
    - focused_regions, focused_regions1: Lists of spots showing transformations
    """
    # Get image dimensions - handle both RGB and grayscale
    if len(image.shape) == 3:  # RGB image
        height, width, channels = image.shape
    else:  # Grayscale image
        height, width = image.shape
        channels = 1

    # Create importance map if not provided
    if importance_map is None:
        if channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Use edge detection to create importance map
        importance_map = cv2.Canny(gray, 50, 150)

        # Dilate to include more surrounding areas
        kernel = np.ones((3, 3), np.uint8)
        importance_map = cv2.dilate(importance_map, kernel, iterations=1)

    # Dynamic focus size determination based on image content
    #if understanding:
        # Calculate optimal focus size using faster, more efficient approach
        #optimal_size = determine_optimal_focus_size(image, image1)
        #focus_size = optimal_size
        #focus_h, focus_w = focus_size
        #print(f"Using optimal focus size: {focus_size}")

    #else:
       # focus_h, focus_w = focus_size
    focus_h, focus_w = focus_size
    print(f"Using optimal focus size: {focus_size}")
    # Understanding mode - analyze transformations between input and output images
    if understanding:
        print("Analyzing transformations between images...")
        start_time = time.time()  # Track execution time

        # Store sequences of transformation spots
        transformation_sequences = []

        # OPTIMIZATION: Use parallel processing for object detection
        input_objects, model = detect_objects_parallel_with_training(image, image1, importance_map, focus_size, similarity_threshold, channels)
        print(f"Found {len(input_objects)} distinct objects in input image in {time.time() - start_time:.2f} seconds")

        # For each object in input image, find its transformation
        for idx, (input_region, input_pos) in enumerate(input_objects):
            print(f"Processing object {idx+1}/{len(input_objects)}...")

            # When calling find_best_match:
            best_match_pos, best_similarity, transformation_data = find_best_match(
                input_region, image1, focus_size, similarity_threshold, channels,
                output_importance_map=importance_map1  # Pass the second image's importance map
            )

            # If we found a match, create a transformation sequence
            if best_match_pos:
                input_y, input_x = input_pos
                output_y, output_x = best_match_pos

                # Initialize sequence with input object spot
                sequence = [image[input_y:input_y+focus_h, input_x:input_x+focus_w]]

                # Analyze transformation type
                transformation_info = analyze_transformation(
                    image, image1, input_region, input_pos, best_match_pos,
                    focus_size, similarity_threshold, channels, transformation_data
                )

                # Add transformation spots to sequence
                sequence.extend(transformation_info["spots"])

                # Save the sequence with information about transformation type
                transformation_sequences.append({
                    "sequence": sequence,
                    "transformation_types": transformation_info["types"],
                    "input_pos": input_pos,
                    "output_pos": best_match_pos,
                    "description": transformation_info["description"]
                })

        print(f"Found {len(transformation_sequences)} transformation sequences")
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
        #visualize_transformation_sequences(transformation_sequences)
        # Combine all sequences into a single list of spots
        all_spots = []
        for seq_info in transformation_sequences:
            all_spots.extend(seq_info["sequence"])

        # Return the sequences for both input and output (maintaining expected return format)
        focused_regions = all_spots
        focused_regions1 = list(all_spots)  # Make a copy

        return transformation_sequences

    # Testing mode - extract all familiar spots with selective matching
    if testing:
        abc = retrieve_all_spot_ids_from_drive_all()
        #print(f"abc - {abc}")
        spot_ids = abc
        focused_regions = []   # Will store the matched stored spots
        focused_regions1 = []  # Will store the matched stored spots (secondary image)
        movements = []         # Will store the positions where matches were found
        matched_positions = set()  # Track positions we've already checked to avoid duplicates
        print(f"spot_ids - {spot_ids}")
        # Convert all stored spot IDs to images once for efficiency
        stored_spots = {}
        for stored_id in spot_ids.keys():
            try:
                stored_spot_pixels, shape = id_to_list_(stored_id)
               # print(f"stored_spot_pixels - {stored_spot_pixels}, shape - {shape}")
                stored_spots[stored_id] = np.array(stored_spot_pixels, dtype=np.uint8).reshape(shape)
            except Exception as e:
                print(f"Error loading stored spot {stored_id}: {e}")
                continue
       # print(f"stored_spots - {stored_spots}")
        # For each position in the image
        step_size = 2  # Reduced from 2 to ensure we don't miss matches
        for y in range(0, height - focus_h + 1, step_size):
            for x in range(0, width - focus_w + 1, step_size):
                # Create position key for tracking
                pos_key = (y, x)
                #print(f"pos_key - {pos_key}")
                if pos_key in matched_positions:
                    continue

                # Skip if not an important region
                region_importance = importance_map[y:y+focus_h, x:x+focus_w]
                if not np.any(region_importance > 0):
                    continue

                # Extract the current spot
                current_spot = image[y:y+focus_h, x:x+focus_w]
                best_match = None
                best_similarity = 0

                # Check against all stored spots
                for stored_id, stored_spot in stored_spots.items():
                    #try:
                    if len(current_spot.shape) == 2:
                      current_spot = cv2.cvtColor(current_spot, cv2.COLOR_GRAY2RGB)
                    if len(stored_spot.shape) == 2:
                        stored_spot = cv2.cvtColor(stored_spot, cv2.COLOR_GRAY2RGB)

                    # 1. Basic SSIM matching
                    basic_similarity = ssim(current_spot, stored_spot, data_range=255,channel_axis=-1, win_size=7)
                    similarity = basic_similarity
                   # print(f"basic_similarity - {basic_similarity}")

                    if similarity <= similarity_threshold:
                        # 2. Rotation matching - try different angles
                        for angle in [90, 180, 270]:
                            rotated_spot = np.rot90(current_spot, k=angle//90)
                            if len(rotated_spot.shape) == 2:
                                  rotated_spot = cv2.cvtColor(rotated_spot, cv2.COLOR_GRAY2RGB)
                            #print(f"rotated_spot - {rotated_spot.shape}")
                           # print(f"stored_spot - {stored_spot.shape}")
                            rot_similarity = ssim(rotated_spot, stored_spot, data_range=255, channel_axis=-1, win_size=7)
                            if rot_similarity > similarity:
                                similarity = rot_similarity

                        # 3. Limited zoom in/out if still no match
                        if similarity <= similarity_threshold:
                            for scale_factor in [0.95, 1.05]:
                                # Scale the current spot
                                scaled_size = int(focus_h * scale_factor)
                                if scaled_size <= 0 or scaled_size >= height:
                                    continue

                                scaled_spot = cv2.resize(current_spot, (scaled_size, scaled_size))

                                # Crop or pad to match original size
                                if scale_factor > 1:  # Larger than original - crop center
                                    start = (scaled_size - focus_h) // 2
                                    scaled_spot = scaled_spot[start:start+focus_h, start:start+focus_h]
                                else:  # Smaller than original - pad with zeros
                                    pad_size = (focus_h - scaled_size) // 2
                                    if len(scaled_spot.shape) == 3:
                                        scaled_spot = np.pad(scaled_spot, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                                                            mode='constant', constant_values=0)
                                    else:
                                        scaled_spot = np.pad(scaled_spot, ((pad_size, pad_size), (pad_size, pad_size)),
                                                            mode='constant', constant_values=0)

                                    # Ensure exact dimensions
                                    scaled_spot = cv2.resize(scaled_spot, (focus_h, focus_w))
                                if len(scaled_spot.shape) == 2:
                                      scaled_spot = cv2.cvtColor(scaled_spot, cv2.COLOR_GRAY2RGB)
                                #print(f"scaled_spot - {scaled_spot.shape}")

                                scale_similarity = ssim(scaled_spot, stored_spot, data_range=255, channel_axis=-1, win_size=7)
                                if scale_similarity > similarity:
                                    similarity = scale_similarity

                    # Track the best match for this position
                    if similarity > similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = stored_id
                       # print(f"best_match - {best_match}")

                    #except Exception as e:
                       # print(f"Error in spot matching: {e}")
                       # continue

               # print(f" best_match - {best_match}")
                # If we found a match at this position
                if best_match:
                    # Add the stored spot to the results
                    matched_spot = stored_spots[best_match]
                    focused_regions.append(matched_spot)
                    focused_regions1.append(matched_spot)
                    movements.append((y, x))

                    # Mark this position as matched
                    matched_positions.add(pos_key)

                    # Also mark nearby positions to avoid redundant matches (optional)
                    buffer = 2  # Adjust based on how close matches should be allowed
                    for by in range(max(0, y-buffer), min(height-focus_h+1, y+buffer+1)):
                        for bx in range(max(0, x-buffer), min(width-focus_w+1, x+buffer+1)):
                            matched_positions.add((by, bx))

        if not movements:
            print("No familiar spots found in the image")
            return None

        print(f"Found {len(movements)} familiar spots in the image")
        #visualize_agent_movements(image, movements)
        #visualize_spots(image, movements)

        # Return the stored spots that matched with the input image
        return focused_regions, focused_regions1

    # Storage for already processed spots to ensure uniqueness
    unique_spots = []
    unique_spot_hashes = set()

    if training:
        abc = retrieve_all_spot_ids_from_drive_all()
        #print(f"abc - {abc}")
        spot_ids = abc
        focused_regions = []
        focused_regions1 = []
        movements = []
        checked_positions = set()

        # Find all possible spot positions with stricter criteria
        possible_positions = []
        min_importance_threshold = 5  # Require at least 5 important pixels

        for y in range(0, height - focus_h + 1, focus_h//4):  # Step by focus_h//4 to reduce overlap
            for x in range(0, width - focus_w + 1, focus_w//4):  # Step by focus_w//4 to reduce overlap
                # Consider this position only if it contains sufficient important pixels
                region_importance = importance_map[y:y+focus_h, x:x+focus_w]
                important_pixel_count = np.sum(region_importance > 0)

                if important_pixel_count >= min_importance_threshold:
                    # Calculate importance score (weighted by pixel values)
                    importance_score = np.sum(region_importance)
                    possible_positions.append((y, x, importance_score))

        print(f"Initial candidate positions: {len(possible_positions)}")

        # Sort positions by importance score
        sorted_positions = sorted(
            possible_positions,
            key=lambda pos: pos[2],  # Sort by importance_score
            reverse=True
        )

        # Take only top candidates to limit processing
        max_candidates = min(200, len(sorted_positions))  # Limit to top 200 candidates
        sorted_positions = sorted_positions[:max_candidates]

        print(f"Processing top {len(sorted_positions)} candidates")

        # Process positions in order of importance
        for y, x, importance_score in sorted_positions:
            if (y, x) in checked_positions:
                continue

            region = image[y:y+focus_h, x:x+focus_w]

            # Skip if variance is too low (increased threshold for 32x32)
            if calculate_variance(region) <= 100:  # Increased from 50 for larger regions
                checked_positions.add((y, x))
                continue

            # Check if this spot is unique compared to all previously found spots
            spot_is_unique = True
            region_flat = region.flatten()
            spot_hash = hash(region_flat.tobytes())

            # First check simple hash for quick elimination
            if spot_hash in unique_spot_hashes:
                spot_is_unique = False
            else:
                # Also check using structural similarity to catch visually similar spots
                for prev_region, _ in unique_spots:
                    try:
                        # For 32x32 regions, SSIM should work fine with default settings
                        similarity = ssim(region, prev_region, data_range=255)

                        if similarity > similarity_threshold:
                            spot_is_unique = False
                            break

                    except ValueError:
                        # Fallback: skip SSIM comparison for problematic cases
                        continue

            if spot_is_unique:
                unique_spot_hashes.add(spot_hash)
                unique_spots.append((region, (y, x)))
                movements.append((y, x))

                # Create spot ID for the global dictionary
                spot_id = list_to_reversible_id_(region.flatten().tolist())
                spot_ids[spot_id] = (y, x)

                # Mark surrounding positions as checked to avoid nearly duplicate spots
                # Increased exclusion radius for 32x32 regions
                exclusion_radius = max(focus_h//2, focus_w//2)
                for dy in range(-exclusion_radius, exclusion_radius + 1, focus_h//8):
                    for dx in range(-exclusion_radius, exclusion_radius + 1, focus_w//8):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny <= height - focus_h and
                            0 <= nx <= width - focus_w):
                            checked_positions.add((ny, nx))
            else:
                checked_positions.add((y, x))

            # Optional: limit total number of unique spots found
            if len(movements) >= 50:  # Stop after finding 50 unique spots
                print(f"Reached maximum limit of {len(movements)} unique spots")
                break

        print(f"Found {len(movements)} unique spots")

        if not movements:
            print("Input fully processed. No new unique important portions found.")
            return None, None, spot_ids

        #visualize_agent_movements(image, movements)
        #visualize_spots(image, movements)
        # Take only the first 5 movements
       # movements = movements[:3]

        #focused_regions = [image[y:y+focus_h, x:x+focus_w] for y, x in movements]
        #focused_regions1 = [image1[y:y+focus_h, x:x+focus_w] for y, x in movements]
        for pos in movements[:5]:  # take first 5
            y, x = pos  # explicit unpacking
           # print(f"Extracting spot at position: ({y}, {x})")
            focused_regions.append(image[y:y+focus_h, x:x+focus_w])
            #focused_regions1.append(image1[y:y+focus_h, x:x+focus_w])

        # After running your training code and getting focused_regions and movements
        #if focused_regions and movements:
            # Main visualization showing spots on original image
            #visualize_final_spots(image, movements, focus_h, focus_w)

            # Grid view of all extracted spots
            #visualize_spots_grid(focused_regions, movements)

        #else:
           # print("No spots found to visualize")

        return focused_regions, focused_regions1, spot_ids

    # Existing logic for dual_process - CORRECTED VERSION
    if dual_process and request_input is not None:
        match_found = False
        requested_position = None
        spot_ids = retrieve_all_spot_ids_from_drive_all()
        #print(f"request_input - {request_input}")
        # Properly decode the stored region with metadata
        try:
            decoded_data, original_shape = id_to_list_(request_input)
            if decoded_data is None:
                print("Failed to decode request input")
                return False

            #print(f"Decoded data shape info: {original_shape}")
            #print(f"Decoded data size: {len(decoded_data)}")

            # Reconstruct the image with proper dimensions
            request_input_2d = unflatten_pixel_values_(decoded_data, original_shape, focus_h, focus_w)
            request_input_2d = request_input_2d.astype(np.uint8)

           # print(f"Reconstructed region shape: {request_input_2d.shape}")
           # print(f"Reconstructed region dtype: {request_input_2d.dtype}")

            # Verify the reconstruction makes sense
            #print(f"Reconstructed stats - Min: {request_input_2d.min()}, Max: {request_input_2d.max()}, Mean: {request_input_2d.mean():.2f}")

        except Exception as e:
            print(f"Error reconstructing stored region: {e}")
            return False

        # Ensure dimensions match focus size
        if request_input_2d.shape != (focus_h, focus_w):
          #  print(f"Resizing from {request_input_2d.shape} to ({focus_h}, {focus_w})")
            request_input_resized = cv2.resize(request_input_2d, (focus_w, focus_h), interpolation=cv2.INTER_NEAREST)
        else:
            request_input_resized = request_input_2d.copy()

        # Rest of your processing code remains the same...
        # Generate importance map and candidate positions
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        importance_map = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        importance_map = cv2.dilate(importance_map, kernel, iterations=1)

        # Find candidate positions
        possible_positions = []
        min_importance_threshold = 5

        for y in range(0, height - focus_h + 1, focus_h//4):
            for x in range(0, width - focus_w + 1, focus_w//4):
                region_importance = importance_map[y:y+focus_h, x:x+focus_w]
                important_pixel_count = np.sum(region_importance > 0)

                if important_pixel_count >= min_importance_threshold:
                    importance_score = np.sum(region_importance)
                    possible_positions.append((y, x, importance_score))

        #print(f"Initial candidate positions: {len(possible_positions)}")

        # Sort and limit candidates
        sorted_positions = sorted(possible_positions, key=lambda pos: pos[2], reverse=True)
        max_candidates = min(200, len(sorted_positions))
        sorted_positions = sorted_positions[:max_candidates]

        print(f"Processing top {len(sorted_positions)} candidates")

        checked_positions = set()

        # Process positions in order of importance
        for y, x, importance_score in sorted_positions:
            if (y, x) in checked_positions:
                continue

            checked_positions.add((y, x))

            region = image[y:y+focus_h, x:x+focus_w]

            # Convert to grayscale if needed and ensure consistent data type
            if len(region.shape) == 3:
                region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                region_gray = region.copy()

            region_gray = region_gray.astype(np.uint8)  # Ensure consistent data type

           # print(f"Region shape: {region_gray.shape}, dtype: {region_gray.dtype}")
           # print(f"Request input shape: {request_input_resized.shape}, dtype: {request_input_resized.dtype}")

            # Debug: Show pixel value ranges
           # print(f"Region stats - Min: {region_gray.min()}, Max: {region_gray.max()}, Mean: {region_gray.mean():.2f}")
           # print(f"Target stats - Min: {request_input_resized.min()}, Max: {request_input_resized.max()}, Mean: {request_input_resized.mean():.2f}")

            # Visualize comparison (optional)
           # visualize_comparison(region_gray, request_input_resized, y, x, similarity=0.0)

            # Calculate similarity with proper data types
            try:
                similarity = ssim(
                    region_gray.astype(np.float32),
                    request_input_resized.astype(np.float32),
                    data_range=255.0
                )
            except Exception as e:
                print(f"SSIM calculation error: {e}")
                continue

           # print(f"Position ({y}, {x}), importance: {importance_score}, similarity: {similarity}")

            if similarity > 0.95:  # You might need to adjust this threshold
                print(f"Match found in image source at position ({y}, {x}) with similarity {similarity}")
                requested_position = (y, x)
                match_found = True
                break

        print(f"Match found: {match_found}")
        if not match_found:
            return match_found

        # Rest of your existing code for handling the match...
        #request_input_id = list_to_reversible_id(request_input_2d.flatten().tolist())
       # print(f"Generated ID from reconstructed region: {request_input_id}")
       
        # Convert spot_ids to a list of tuples (key, value)
        spot_id_items = list(spot_ids.items())
       # print(f"request_input_id - {request_input_id}")
        #print(type(request_input_id))
        # Use tolerance-based matching instead of exact match
        matching_id, matching_position = find_matching_id_with_tolerance(
            request_input, spot_ids, tolerance=5.0  # Adjust tolerance as needed (inc number = inc pixel difference)
        )

        if matching_id:
           # print("Matching ID found")
            current_position = spot_ids[matching_id]
            print(f"Current position: {current_position}")
            print(f"matching_id - {matching_id}")
            current_index = list(spot_ids.values()).index(current_position)
            #make sure this next id is present in input source.
            if current_index + 1 < len(spot_ids):
                #print("3")
                if not branch:
                    next_position = list(spot_ids.values())[current_index + 1]
                    print(f"next_position - {next_position}")
                    next_spot_id_key, next_position_ = spot_id_items[current_index + 1]
                    print("Connected spot found")
                  # spot_image = image[next_position[0]:next_position[0]+focus_h,
                                        #next_position[1]:next_position[1]+focus_w]
                    #plt.figure()
                    #plt.imshow(spot_image, cmap='gray')
                  # plt.title("Connected Spot Visualization")
                    #plt.show()
                    return (next_spot_id_key, match_found)

                else:
                    # When branching is true, return two consecutive IDs
                    if current_index + 2 < len(spot_ids):  # Make sure we have at least 2 more items
                        # Get the next two consecutive positions
                        next_position_1 = list(spot_ids.values())[current_index + 1]
                        next_position_2 = list(spot_ids.values())[current_index + 2]

                        # Get the corresponding spot ID keys
                        next_spot_id_key_1, _ = spot_id_items[current_index + 1]
                        next_spot_id_key_2, _ = spot_id_items[current_index + 2]

                        print(f"Branch mode: returning two consecutive IDs")
                        print(f"First next position: {next_position_1}")
                        print(f"Second next position: {next_position_2}")
                        print("Two consecutive spots found for branching")

                        # Return both IDs as a tuple along with match_found flag
                        return ((next_spot_id_key_1, next_spot_id_key_2), match_found)

                    elif current_index + 1 < len(spot_ids):  # If only one more item available
                        # Return just one ID if we don't have two consecutive ones
                        next_position = list(spot_ids.values())[current_index + 1]
                        next_spot_id_key, _ = spot_id_items[current_index + 1]

                        print(f"Branch mode: only one consecutive ID available")
                        print(f"Next position: {next_position}")
                        print("Only one consecutive spot found for branching")

                        # Return single ID in a tuple format for consistency
                        return ((next_spot_id_key, None), match_found)

                    else:
                        print("Branch mode: no consecutive spots available")
                        return ((None, None), match_found)

def decode_id_to_array(encoded_id):
    """Decode the reversible ID back to numpy array using the correct format"""
    if isinstance(encoded_id, (list, tuple)):
        if len(encoded_id) == 0:
            print("Empty encoded_id list/tuple")
            return None, None
        encoded_id = encoded_id[0]  # assume first element is the actual ID
    elif not isinstance(encoded_id, str):
        encoded_id = str(encoded_id)  # fallback: force to string

    try:
        # Decode base64
        compressed_data = base64.urlsafe_b64decode(encoded_id)

        # Decompress
        decompressed_data = zlib.decompress(compressed_data)

        # Unpack: [metadata_length(4 bytes)][metadata][pixel_data]
        metadata_length = struct.unpack('I', decompressed_data[:4])[0]

        # Extract metadata
        metadata_bytes = decompressed_data[4:4+metadata_length]
        metadata = json.loads(metadata_bytes.decode('utf-8'))

        # Extract pixel data
        pixel_data_bytes = decompressed_data[4+metadata_length:]

        # Convert back to numpy array
        dtype = np.dtype(metadata['dtype'])
        array_flat = np.frombuffer(pixel_data_bytes, dtype=dtype)

        # Reshape if we have shape info
        if metadata['shape'] is not None:
            array = array_flat.reshape(metadata['shape'])
        else:
            array = array_flat

        return array

    except Exception as e:
        print(f"Error decoding ID: {e}")
        return None

def find_matching_id_with_tolerance(request_input_id, spot_ids, tolerance=0.01):
    """Find matching ID in spot_ids with tolerance for small differences"""
    
    print(f"Finding match for request_input_id with tolerance {tolerance}")
    request_array = decode_id_to_array(request_input_id)

    if request_array is None:
        print("Failed to decode request_input_id")
        return None, None

   # print(f"Request array shape: {request_array.shape}")
   # print(f"Request array sample: {request_array.flatten()[:10]}")
   # print(f"spot_ids 1 - {spot_ids}")
    for spot_id, position in spot_ids.items():
        print(f"Checking spot_id - {spot_id}")
        print(type(spot_id))
        spot_array = decode_id_to_array(spot_id)
        if spot_array is None:
            continue

        # Check if arrays have same shape
        if request_array.shape != spot_array.shape:
            continue

        # Calculate similarity
        # Method 1: Mean absolute difference (good for pixel data)
        diff = np.mean(np.abs(request_array.astype(float) - spot_array.astype(float)))

        if diff <= tolerance:
            print(f"Match found with mean absolute difference: {diff}")
            return spot_id, position

        # Alternative: Structural Similarity (better for images)
        # You could also use normalized cross-correlation
        max_val = max(np.max(request_array), np.max(spot_array))
        print(f"Max value for normalization: {max_val}")
        if max_val > 0:
            normalized_diff = diff / max_val
            print(f"Normalized difference: {normalized_diff}")
            if normalized_diff <= tolerance:
                print(f"Match found with normalized difference: {normalized_diff}")
                return spot_id, position

    print("No matching ID found within tolerance")
    return None, None

def list_to_reversible_id_(pixel_list):
    """Store pixel data with proper handling of RGB vs grayscale"""
    array = np.array(pixel_list, dtype=np.uint8)

    # Store metadata about the original shape
    if len(array.shape) > 1:
        # If it's already 2D/3D, flatten it
        original_shape = array.shape
        array_flat = array.flatten()
    else:
        # If it's already flattened, try to infer the shape
        array_flat = array
        # Try to determine if it's RGB or grayscale based on size
        total_pixels = len(array_flat)
        if total_pixels == 1024:  # 32x32 grayscale
            original_shape = (32, 32)
        elif total_pixels == 3072:  # 32x32x3 RGB
            original_shape = (32, 32, 3)
        else:
            # Try to infer square dimensions
            side_length = int(np.sqrt(total_pixels))
            if side_length * side_length == total_pixels:
                original_shape = (side_length, side_length)
            elif side_length * side_length * 3 == total_pixels:
                original_shape = (side_length, side_length, 3)
            else:
                original_shape = None

    # Create metadata
    metadata = {
        'shape': original_shape,
        'size': len(array_flat),
        'dtype': str(array_flat.dtype)
    }

    # Combine metadata and data
    metadata_bytes = json.dumps(metadata).encode('utf-8')
    metadata_length = len(metadata_bytes)

    # Pack: [metadata_length(4 bytes)][metadata][pixel_data]
    combined_data = struct.pack('I', metadata_length) + metadata_bytes + array_flat.tobytes()

    compressed = zlib.compress(combined_data)
    encoded = base64.urlsafe_b64encode(compressed).decode()
    return encoded

def id_to_list_(encoded_id, expected_size=None):
    """Properly decode with metadata"""
        # Normalize input to string
    if isinstance(encoded_id, (list, tuple)):
        if len(encoded_id) == 0:
            print("Empty encoded_id list/tuple")
            return None, None
        encoded_id = encoded_id[0]  # assume first element is the actual ID
    elif not isinstance(encoded_id, str):
        encoded_id = str(encoded_id)  # fallback: force to string

    print(f"Decoding ID: {encoded_id[:30]}... (truncated)")
    
    try:
        compressed = base64.urlsafe_b64decode(encoded_id.encode())
        decompressed = zlib.decompress(compressed)

        # Extract metadata
        metadata_length = struct.unpack('I', decompressed[:4])[0]
        metadata_bytes = decompressed[4:4+metadata_length]
        pixel_data = decompressed[4+metadata_length:]

        # Parse metadata
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        original_shape = metadata['shape']
        original_size = metadata['size']

        # Reconstruct array
        array = np.frombuffer(pixel_data, dtype=np.uint8)

        #print(f"Decoded metadata: shape={original_shape}, size={original_size}")
        #print(f"Actual decoded size: {len(array)}")

        return array, original_shape

    except Exception as e:
        print(f"Error decoding ID with metadata: {e}")
        # Fallback to old method
        try:
            compressed = base64.urlsafe_b64decode(encoded_id.encode())
            decompressed = zlib.decompress(compressed)
            array = np.frombuffer(decompressed, dtype=np.uint8)

            # Try to guess the shape based on size
            if len(array) == 1024:
                shape = (32, 32)
            elif len(array) == 3072:
                shape = (32, 32, 3)
            else:
                shape = None

            return array, shape
        except Exception as e2:
            print(f"Fallback decoding also failed: {e2}")
            return None, None

def unflatten_pixel_values_(flattened_pixels, original_shape, focus_h=32, focus_w=32):
    """Properly reshape with original shape information"""
    if isinstance(flattened_pixels, list):
        flattened_pixels = np.array(flattened_pixels, dtype=np.uint8)

    if original_shape is None:
        print("Warning: No shape information available, using default")
        return flattened_pixels.reshape((focus_h, focus_w))

    try:
        # Reshape to original dimensions
        if len(original_shape) == 3:  # RGB image
            reshaped = flattened_pixels.reshape(original_shape)
            # Convert to grayscale for comparison
            if original_shape[2] == 3:
                reshaped_gray = cv2.cvtColor(reshaped, cv2.COLOR_RGB2GRAY)
               # print(f"Converted RGB {original_shape} to grayscale {reshaped_gray.shape}")
                return reshaped_gray
        else:  # Grayscale image
            reshaped = flattened_pixels.reshape(original_shape)
            print(f"Reshaped to original grayscale shape: {original_shape}")
            return reshaped

    except Exception as e:
        print(f"Error reshaping with original shape {original_shape}: {e}")
        # Fallback to default reshape
        expected_size = focus_h * focus_w
        if len(flattened_pixels) >= expected_size:
            return flattened_pixels[:expected_size].reshape((focus_h, focus_w))
        else:
            padded = np.pad(flattened_pixels, (0, expected_size - len(flattened_pixels)), 'constant')
            return padded.reshape((focus_h, focus_w))

# Add this visualization code right before the similarity calculation
def visualize_comparison(region_gray, request_input_resized, y, x, similarity=None):
    """Visualize the region and target side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Show the extracted region
    axes[0].imshow(region_gray, cmap='gray')
    axes[0].set_title(f'Region at ({y}, {x})')
    axes[0].axis('off')

    # Show the target (request_input)
    axes[1].imshow(request_input_resized, cmap='gray')
    axes[1].set_title('Target (Request Input)')
    axes[1].axis('off')

    # Show difference map
    diff = np.abs(region_gray.astype(np.float32) - request_input_resized.astype(np.float32))
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Difference Map')
    axes[2].axis('off')

    if similarity is not None:
        fig.suptitle(f'SSIM Similarity: {similarity:.4f}', fontsize=14)

    plt.tight_layout()
    plt.show()

    # Also print some statistics
    print(f"Region stats - Min: {region_gray.min()}, Max: {region_gray.max()}, Mean: {region_gray.mean():.2f}")
    print(f"Target stats - Min: {request_input_resized.min()}, Max: {request_input_resized.max()}, Mean: {request_input_resized.mean():.2f}")
    print("-" * 50)

def visualize_transformation_sequences(transformation_sequences):
    """Visualizes the transformation sequences for each object"""
    for i, seq_info in enumerate(transformation_sequences):
        sequence = seq_info["sequence"]
        types = seq_info["transformation_types"]

        plt.figure(figsize=(len(sequence) * 3, 3))
        for j, spot in enumerate(sequence):
            plt.subplot(1, len(sequence), j + 1)

            # Display RGB image
            plt.imshow(cv2.cvtColor(spot, cv2.COLOR_BGR2RGB))  # Convert BGR (OpenCV) to RGB (Matplotlib)

            plt.title(f"Step {j + 1}")
            plt.axis('off')

        plt.suptitle(f"Object {i + 1} Transformation: {', '.join(types)}")
        plt.tight_layout()
        plt.show()


def save_spot_ids_to_drive(spot_ids):
    """Save spot IDs with a counter for uniqueness."""
    drive_folder = '/content/drive/MyDrive/ring tree/ids'
    os.makedirs(drive_folder, exist_ok=True)  # Ensure directory exists

    # Find existing files and determine the next counter
    existing_files = [f for f in os.listdir(drive_folder) if f.startswith('spot_ids_') and f.endswith('.json')]
    next_counter = len(existing_files) + 1

    # Create a filename using a counter
    file_path = os.path.join(drive_folder, f'spot_ids_{next_counter:04d}.json')

    try:
        with open(file_path, 'w') as f:
            json.dump(spot_ids, f, indent=4)
        #print(f"Spot IDs successfully saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving spot IDs: {e}")
        return None
# spot id hash increment
# def save_and_store_spot_ids(
#     spot_ids,
#     focus_id=None,
#     ids_dir=r"D:\artist\brainX\CRX\Properties\System1_inputs_eye",
#     imagine_dir=r"D:\artist\brainX\CRX\Properties\Imagination",
#     all_dir=r"D:\artist\brainX\CRX\Properties\System1_inputs_eye",
#     state_file="spot_state.json"
# ):
#     """
#     Save spot IDs individually with counter, with rules:
#     - Same image → same counter
#     - New image → increment counter
#     - Store normally into ids_dir
#     - If the counter is divisible by 10, save current node also into imagine_dir/node.json
#     - Always update consolidated dictionary in all_dir/spot_ids.json
#     """

#     os.makedirs(ids_dir, exist_ok=True)
#     os.makedirs(imagine_dir, exist_ok=True)
#     os.makedirs(all_dir, exist_ok=True)

#     individual_file_path = None
#     consolidated_file_path = None

#     # Track last image hash and counter across runs
#     state_path = os.path.join(ids_dir, state_file)
#     if os.path.exists(state_path):
#         with open(state_path, "r") as f:
#             state = json.load(f)
#         last_hash = state.get("last_hash")
#         last_counter = state.get("last_counter", 0)
#     else:
#         state = {}
#         last_hash = None
#         last_counter = 0

#     # Compute a hash of spot_ids to detect "same image"
#     spot_str = json.dumps(spot_ids, sort_keys=True)
#     spot_hash = hashlib.md5(spot_str.encode()).hexdigest()

#     if spot_hash == last_hash:
#         # Same image → reuse counter
#         next_counter = last_counter
#     else:
#         # New image → increment counter
#         next_counter = last_counter + 1
#         last_hash = spot_hash
#         last_counter = next_counter

#     # --- Step 1: Save individual spot IDs (if focus_id provided) ---
#     if focus_id is not None:
#         file_path = os.path.join(ids_dir, f"spot_ids_{next_counter:04d}.json")

#         try:
#             with open(file_path, "w") as f:
#                 json.dump(focus_id, f, indent=4)
#             print(f"✅ Spot IDs saved to {file_path}")
#             individual_file_path = file_path
#         except Exception as e:
#             print(f"❌ Error saving spot IDs: {e}")
#             return None, None

#         # Step 1b: urgent node (only for new unique image counters divisible by 10)
#         if last_counter % 10 == 0:
#             urgent_node_path = os.path.join(imagine_dir, "node.json")
#             try:
#                 with open(urgent_node_path, "w") as f:
#                     json.dump(focus_id, f, indent=4)
#                 print(f"🚨 Urgent node saved to {urgent_node_path}")
#                 individual_file_path = urgent_node_path
#             except Exception as e:
#                 print(f"❌ Error saving urgent node: {e}")
#                 return None, None

#     # Save updated state
#     with open(state_path, "w") as f:
#         json.dump({"last_hash": last_hash, "last_counter": last_counter}, f)

#     # --- Step 2: Store all spot_ids in unified dictionary ---
#     file_path_all = os.path.join(all_dir, "spot_ids.json")

#     try:
#         if os.path.exists(file_path_all):
#             try:
#                 with open(file_path_all, "r", encoding="utf-8") as f:
#                     existing_data = json.load(f)
#             except json.JSONDecodeError:
#                 backup_path = file_path_all + ".corrupted"
#                 os.replace(file_path_all, backup_path)
#                 print(f"⚠ Corrupted JSON detected. Backed up to {backup_path}. Starting fresh.")
#                 existing_data = {}
#         else:
#             existing_data = {}

#         # Append without overwriting keys
#         for key, value in spot_ids.items():
#             if key not in existing_data:
#                 existing_data[key] = value

#         with open(file_path_all, "w", encoding="utf-8") as f:
#             json.dump(existing_data, f, indent=4, ensure_ascii=False)

#         consolidated_file_path = file_path_all

#     except Exception as e:
#         print(f"❌ Error storing all spot IDs: {e}")
#         return individual_file_path, None

#     return individual_file_path, consolidated_file_path

def save_and_store_spot_ids(
    spot_ids,
    focus_id=None,
    frame_number=None,   # 👈 pass the current frame number here
    importance_map=None,
    ids_dir=r"D:\artist\brainX\CRX\Properties\System1_inputs_eye",
    imagine_dir=r"D:\artist\brainX\CRX\Properties\Imagination",
    importance_dir=r"D:\artist\brainX\CRX\Properties\latest_images",
    all_dir=r"D:\artist\brainX\CRX\Properties\System1_inputs_eye",
    state_file="spot_state.json"
):
    """
    Save spot IDs individually with counter, with rules:
    - Same frame → same counter
    - New frame → increment counter
    - Store normally into ids_dir
    - If the counter is divisible by 10, save current node also into imagine_dir/node.json
    - Always update consolidated dictionary in all_dir/spot_ids.json
    - Also save importance map with the same counter
    """

    os.makedirs(ids_dir, exist_ok=True)
    os.makedirs(imagine_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(importance_dir, exist_ok=True)

    individual_file_path = None
    consolidated_file_path = None

    # Track last frame and counter across runs
    state_path = os.path.join(ids_dir, state_file)
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        last_frame = state.get("last_frame")
        last_counter = state.get("last_counter", 0)
    else:
        state = {}
        last_frame = None
        last_counter = 0

    # --- Decide counter logic ---
    if frame_number is not None:
        if last_frame is None or frame_number != last_frame:
            # New frame detected → increment counter
            next_counter = last_counter + 1
            last_frame = frame_number
            last_counter = next_counter
        else:
            # Same frame → reuse old counter
            next_counter = last_counter
    else:
        # Fallback: no frame number provided → always increment
        next_counter = last_counter + 1
        last_counter = next_counter

    # --- Step 1: Save individual spot IDs (if focus_id provided) ---
    if focus_id is not None:
        file_path = os.path.join(ids_dir, f"spot_ids_{next_counter:04d}.json")
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(focus_id)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            print(f"✅ Spot IDs appended to {file_path}")
            individual_file_path = file_path

        except Exception as e:
            print(f"❌ Error saving spot IDs: {e}")
            return None, None

        # Step 1b: urgent node (only for new unique image counters divisible by 10)
        if last_counter % 10 == 0:
            urgent_node_path = os.path.join(imagine_dir, "node.json")
            try:
                with open(urgent_node_path, "w") as f:
                    json.dump(focus_id, f, indent=4)
                print(f"🚨 Urgent node saved to {urgent_node_path}")
                individual_file_path = urgent_node_path
            except Exception as e:
                print(f"❌ Error saving urgent node: {e}")
                return None, None

    # --- Step 1c: Save importance map (if provided) ---
    if importance_map is not None:
        try:
            importance_filename = f"importance_{next_counter:04d}.npy"
            importance_path = os.path.join(importance_dir, importance_filename)
            np.save(importance_path, importance_map)
            print(f"[System1] Importance map saved at {importance_path}")
        except Exception as e:
            print(f"❌ Error saving importance map: {e}")

    # Save updated state
    with open(state_path, "w") as f:
        json.dump({"last_frame": last_frame, "last_counter": last_counter}, f)

    # --- Step 2: Store all spot_ids in unified dictionary ---
    file_path_all = os.path.join(all_dir, "spot_ids.json")

    try:
        if os.path.exists(file_path_all):
            try:
                with open(file_path_all, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                backup_path = file_path_all + ".corrupted"
                os.replace(file_path_all, backup_path)
                print(f"⚠ Corrupted JSON detected. Backed up to {backup_path}. Starting fresh.")
                existing_data = {}
        else:
            existing_data = {}

        # Append without overwriting keys
        for key, value in spot_ids.items():
            if key not in existing_data:
                existing_data[key] = value

        with open(file_path_all, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

        consolidated_file_path = file_path_all

    except Exception as e:
        print(f"❌ Error storing all spot IDs: {e}")
        return individual_file_path, None

    return individual_file_path, consolidated_file_path

def save_spot_ids_to_drive_test(spot_ids):
    """Save spot IDs with a counter for uniqueness."""
    drive_folder = '/content/drive/MyDrive/ring tree/ids_test'
    os.makedirs(drive_folder, exist_ok=True)  # Ensure directory exists

    # Find existing files and determine the next counter
    existing_files = [f for f in os.listdir(drive_folder) if f.startswith('spot_ids_') and f.endswith('.json')]
    next_counter = len(existing_files) + 1

    # Create a filename using a counter
    file_path = os.path.join(drive_folder, f'spot_ids_{next_counter:04d}.json')

    try:
        with open(file_path, 'w') as f:
            json.dump(spot_ids, f, indent=4)
        #print(f"Spot IDs successfully saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving spot IDs: {e}")
        return None

def store_spot_ids_to_drive_all(spot_ids):
    """Store the given spot IDs in a JSON file in Google Drive within one unified dictionary."""
    drive_folder = r'D:\artist\brainX\CRX\Properties\System1_inputs_eye'
    os.makedirs(drive_folder, exist_ok=True)  # Ensure directory exists

    file_path = os.path.join(drive_folder, 'spot_ids.json')

    try:
        # Load existing data safely
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                # Backup corrupted file and reset
                backup_path = file_path + ".corrupted"
                os.replace(file_path, backup_path)  # overwrite backup if exists
                print(f"⚠ Corrupted JSON detected. Backed up to {backup_path}. Starting fresh.")
                existing_data = {}

        # Append or update data without overwriting keys
        for key, value in spot_ids.items():
            if key not in existing_data:
                existing_data[key] = value

        # Write back clean JSON
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

        return file_path
    except Exception as e:
        print(f"Error storing spot IDs: {e}")
        return None

def retrieve_all_spot_ids_from_drive_all():
    """Retrieve all stored spot IDs from Google Drive."""
    drive_folder = r'D:\artist\brainX\CRX\Properties\System1_inputs_eye'
    file_path = os.path.join(drive_folder, 'spot_ids.json')

    try:
        if not os.path.exists(file_path): 
            print("No spot IDs found.") 
            return {}

        with open(file_path, 'r') as f: 
            spot_ids = json.load(f) 
            print(f"Successfully retrieved {len(spot_ids)} spot ID entries.")
            return spot_ids
        
    except Exception as e:
        print(f"Error retrieving spot IDs: {e}")
        return {}

def compare_focus(focus_list1, focus_list2):
    """
    Compare two lists of focus values and print whether they are the same or different.
    """
    if len(focus_list1) != len(focus_list2):
        print("❗ The focus lists have different lengths. They are not the same.")
        return

    for i, (f1, f2) in enumerate(zip(focus_list1, focus_list2)):
        if np.array_equal(f1, f2):
            print(f"✅ Focus {i + 1} is the same.")
        else:
            print(f"❗ Focus {i + 1} is different.")
           # print(f"  Focus 1: {f1}")
           # print(f"  Focus 2: {f2}")

def visualize_processed_pixels(focused_regions, images_per_row=4, save_path="focused_spots_test_new.png"):
    """
    Visualizes processed image regions in a grid format with 4 images per row and saves the visualization as a PNG file.
    """
    num_regions = len(focused_regions)
    if num_regions == 0:
        print("No processed regions to visualize.")
        return

    rows = (num_regions + images_per_row - 1) // images_per_row
    fig, axes = plt.subplots(rows, images_per_row, figsize=(images_per_row * 3, rows * 3))

    # Handle axes flattening properly
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < num_regions:
            img = np.array(focused_regions[i], dtype=np.uint8)

            # Convert grayscale 2D image to RGB for consistent display
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            elif img.ndim == 3 and img.shape[2] == 3:
                ax.imshow(img)
            else:
                print(f"Skipping unrecognized image shape: {img.shape}")
                ax.axis('off')
                continue

            ax.set_title(f"Spot {i+1}")
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle("Focused Spots", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as {save_path}")
    plt.close()

def visualize_importance_map(image, importance_map):
    """
    Visualize the importance map overlaid on the original color image.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Importance Map")
    plt.imshow(importance_map, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Important Regions Overlay")

    # Create a color overlay where important regions are highlighted in red
    overlay = image.copy()
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = 255  # Red channel

    # Apply red mask where importance_map is non-zero
    mask = importance_map > 0
    overlay[mask] = red_mask[mask]

    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def save_focusing_agent_output(transformation_sequences, drive_folder='/content/drive/MyDrive/ring tree/rule patchs'):
    """
    Save the output of the focusing agent in JSON format, converting image patches to IDs.

    Parameters:
    - transformation_sequences: List of transformation sequences from focusing_agent
    - drive_folder: Path to save the JSON file

    Returns:
    - Path to the saved JSON file
    """
    # Create folder if it doesn't exist
    os.makedirs(drive_folder, exist_ok=True)

    # Create a timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"focusing_agent_output_{timestamp}.json"
    filepath = os.path.join(drive_folder, filename)

    # Prepare data for JSON serialization
    serializable_sequences = []

    for seq in transformation_sequences:
        # Convert each image in the sequence to an ID
        sequence_ids = []
        for img_patch in seq["sequence"]:
            patch_id = list_to_reversible_id(img_patch)
            sequence_ids.append(patch_id)

        serializable_seq = {
            "transformation_types": seq["transformation_types"],
            "input_pos": tuple(seq["input_pos"]),
            "output_pos": tuple(seq["output_pos"]),
            "description": seq["description"],
            "sequence_ids": sequence_ids,  # Store IDs instead of shapes
            "sequence_shapes": [arr.shape for arr in seq["sequence"]]  # Still keep shapes for reference
        }
        serializable_sequences.append(serializable_seq)

    # Create metadata
    output_data = {
        "timestamp": timestamp,
        "num_transformations": len(transformation_sequences),
        "transformation_sequences": serializable_sequences
    }

    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)

    #print(f"Successfully saved focusing agent output to {filepath}")
    return filepath

def save_importance_map(importance_map, save_path="D:/artist/brainX/CRX/Properties/importance_map.npy"):
    """
    Save the latest importance map (overwrite previous one).
    
    Args:
        importance_map (np.ndarray): Binary importance map.
        save_path (str): File path to save the importance map.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, importance_map)
    print(f"[System1] Latest importance map saved at {save_path}")

def load_eye_input_images(eye_dir=r"D:\artist\brainX\CRX\Properties\Eye_input"):
    """
    Continuously yield image paths and loaded images (PNG).
    If no images exist, wait until new images appear.
    """
    if not os.path.exists(eye_dir):
        os.makedirs(eye_dir, exist_ok=True)
        print(f"[load_eye_input_images] Created missing directory: {eye_dir}")

    while True:  # keep waiting until images are found
        files = sorted(
            [f for f in os.listdir(eye_dir) if f.endswith(".png") and f.startswith("frame_")]
        )

        if not files:
            time.sleep(0.2)  # wait a bit before checking again
            continue

        for file_name in files:
            image_path = os.path.join(eye_dir, file_name)

            # Load image as numpy array
            image = cv2.imread(image_path)
            if image is None:
                print(f"[load_eye_input_images] Failed to load {image_path}")
                continue

            yield image_path, image

            # Remove after yielding
            try:
                os.remove(image_path)
                print(f"[load_eye_input_images] Removed {image_path}")
            except FileNotFoundError:
                pass

def store_latest_image(image: np.ndarray):
    """
    Save the latest processed image (overwrites the old one).
    Args:
        image: numpy array (128x128x3 BGR image)
    """
    # --- Storage paths ---
    LATEST_IMAGE_PATH = "D:/artist/brainX/CRX/Properties/latest_image.png"

    os.makedirs(os.path.dirname(LATEST_IMAGE_PATH), exist_ok=True)
    cv2.imwrite(LATEST_IMAGE_PATH, image)
    print(f"[Store] Latest image saved -> {LATEST_IMAGE_PATH}")

training = True
testing = False
understanding = False
focus = []
focus2_ = []
image_path1 = []

if __name__ == "__main__":
    for image_path, image in load_eye_input_images():
        print(f"Now processing: {image_path}")
        # Isolate the file name
        file_name = image_path.split('\\')[-1]

        # Remove the file extension
        file_name_without_extension = file_name.split('.')[0]

        # Extract the frame number by removing the prefix
        frame_number = file_name_without_extension.replace('frame_', '')
        # Assign path into your variable
        #test_image_path = '/content/test 1.2.1.png'
        test_image_path = rf"{image_path}"   # replace with current file
        
       # print("Assigned ->", test_image_path)

        # If you want to show image shape
        #print("Image shape:", image.shape)
                

        if training == True:
            image = image_to_pixels(test_image_path)
            # Save it
            #store_latest_image(image)

            image1 = image_to_pixels(test_image_path)
            importance_map = generate_importance_map(image, image1)
            #visualize_importance_map(image, importance_map)
            # Save only the latest one
            save_importance_map(importance_map)
            focus, focus2, spot_ids = focusing_agent(image, image1, importance_map, training=True)
           # compare_focus(focus, focus2)
            if not focus:
                print("No familiar spots to process")
                continue
            print(f"spot_ids - {spot_ids}")
            focus_ids = []
            for focus1 in focus:
                # Ensure consistent storage - convert to grayscale if RGB
                if len(focus1.shape) == 3:
                # focus1_gray = cv2.cvtColor(focus1, cv2.COLOR_RGB2GRAY)
                    focus_id = list_to_reversible_id_(focus1.flatten().tolist())
                else:
                    focus_id = list_to_reversible_id_(focus1.flatten().tolist())

                # focus_ids.append(focus_id)
                # save_spot_ids_to_drive(focus_id)
                
                individual_path, consolidated_path = save_and_store_spot_ids(spot_ids, focus_id, frame_number, importance_map)

        store_spot_ids_to_drive_all(spot_ids)

        if testing == True:
        # retrieve_all_spot_ids_from_drive_all()
            image = image_to_pixels(image_path)
            image1 = image_to_pixels(image_path1)
            importance_map = generate_importance_map(image, image1)
            visualize_importance_map(image, importance_map)

            result = focusing_agent(image, image1, importance_map, testing=True)

            if result:
                focus1, focus2 = result
                focus2_ = []  # Reset the list before appending

                for focus_ in focus1: # For testing purpose
                    focus__ = list_to_reversible_id_(focus_.flatten().tolist())
                    focus2_.append(focus__)
                    save_spot_ids_to_drive_test(focus__)

                print("Test visualization")
                visualize_processed_pixels(focus2)
            else:
                print("No familiar spots to process")

        if understanding == True:
            image = image_to_pixels(image_path)
            image1 = image_to_pixels(image_path1)
            importance_map = generate_importance_map(image, image1)
            importance_map1 = generate_importance_map(image1, image)
            visualize_importance_map(image, importance_map)
            visualize_importance_map(image1, importance_map1)

            result = focusing_agent(image, image1, importance_map, importance_map1, understanding=True)
            save_focusing_agent_output(result)

        #print("id comparison")
        #compare_focus(focus_ids, focus2_)
        #print("test visualization")
        #visualize_processed_pixels(focus2)

        #in testing, include a threshold of matching spots. most similar images will have more matching spots surely passing the threshold.
        #less similar images will low matching spots not passing threshold hence filtering only the correct input to ring tree.