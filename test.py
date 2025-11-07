import cv2
import numpy as np
import svgwrite
from svgwrite import cm, pt

def jpeg_to_svg_edges(jpeg_path, svg_path):
    """
    Converts a JPEG image to an SVG file by tracking edges and ignoring color/white background.
    """
    # --- 1. Image Loading and Preprocessing ---
    # Load the image in grayscale, which inherently ignores color information.
    # This also helps in dealing with a white background as edges will be the contrast lines.
    try:
        img = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Error: Could not open or find the image at {jpeg_path}")
    except Exception as e:
        print(f"An error occurred during image loading: {e}")
        return

    # Blur the image to reduce noise, which is critical for good edge detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # --- 2. Edge Detection ---
    # Apply Canny edge detection. Adjust the thresholds (100, 200) for different results.
    # This process outputs a binary image where edges are white and everything else is black.
    edges = cv2.Canny(blurred, 100, 200)

    # --- 3. Contour Finding ---
    # Find contours from the edge-detected image.
    # cv2.RETR_EXTERNAL retrieves only the external contours (good for object outlines).
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 4. SVG Generation ---
    dwg = svgwrite.Drawing(svg_path, size=(img.shape[1], img.shape[0]), profile='tiny')

    # Iterate through the detected contours
    for contour in contours:
        # Approximate the contour to a polygon with less points (simplifying the path)
        # 1-3 is the epsilon parameter, a smaller value means a closer approximation.
        epsilon = 3
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Reshape the contour points into a list of (x, y) tuples for svgwrite
        points = approx.reshape(-1, 2).tolist()

        # Create an SVG path from the points.
        # 'fill="none"' ensures only the outline is visible (ignoring 'color' filling).
        # 'stroke="black"' gives the edge a black line.
        if len(points) > 1:
            path_data = "M" + " L".join(f"{x},{y}" for x, y in points)
            dwg.add(dwg.path(d=path_data, stroke='black', fill='none', stroke_width=1))

    # Save the SVG file
    dwg.save()
    print(f"Successfully converted {jpeg_path} to {svg_path}")

# --- Example Usage ---
# NOTE: Replace 'input.jpg' with your JPEG file path.
# The JPEG should ideally have a clear separation between the object and the white background.
jpeg_to_svg_edges('input.jpg', 'output.svg')
