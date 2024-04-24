import cv2
import numpy as np
from typing import Tuple


def dec2rgb(x: int) -> list:
    """
    Decimal value to RGB encoding.
    
    Args:
        x (int): The decimal value to convert.
        
    Returns:
        list: A list containing the RGB values.
    """
    return [x // 65536, (x % 65536) // 256, x % 256]


def rgb2dec(x: list) -> int:
    """
    RGB to decimal decoding.
    
    Args:
        x (tuple): A tuple containing the RGB colour values.
        
    Returns:
        int: The decimal representation of the RGB colour values.
    """
    return x[0] * 65536 + x[1] * 256 + x[2]


def det(a: tuple, b: tuple) -> float:
    """
    Calculates the determinant of a 2x2 matrix formed by vectors a and b.
    
    Args:
        a (tuple): The first vector.
        b (tuple): The second vector.
    
    Returns:
        float: The determinant of the matrix.
    """
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def line_intersection(line1: tuple, line2: tuple) -> tuple:
    """
    Calculate the intersection point of two line sections.

    Args:
        line1 (tuple): Tuple representing the coordinates of the first line's endpoints.
        line2 (tuple): Tuple representing the coordinates of the second line's endpoints.

    Returns:
        tuple: Tuple representing the coordinates of the intersection point (x, y).

    Raises:
        ZeroDivisionError: If the line sections do not intersect each other.
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
       raise ZeroDivisionError("Lines do not intersect!")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def walk_lines(points: list, mask: np.array) -> tuple:
    """
    Walk the polygon points and draw lines. The direction (order of points/lines) is clockwise so 
    the inside of the polygon is always on the right side along each line. The shifts represent the 
    direction of the inside of the polygon. The colour of each line is its index encoded to RGB. 

    Args:
        points (list): List or array of points representing the points of the polygon.
        mask (ndarray): Binary mask.

    Returns:
        tuple: A tuple containing the generated mask with polylines and shifts.

    """
    # Create mask polylines where line colors is its encoded index
    mask_poly = np.zeros((*mask.shape, 3), dtype=np.uint8)
    normals = np.empty_like(points, dtype=float)
    n_lines = len(points)
    for i in range(n_lines):
        j = (i + 1) % n_lines
        a = points[i]
        b = points[j]
        cv2.line(mask_poly, a, b, dec2rgb(i + 1), 1)  # this function accepts only uint8
        dx = b[1] - a[1]
        dy = b[0] - a[0]
        normal = np.array([dy, -dx])  # watch out for the sign and order!
        norm = np.linalg.norm(normal)
        normals[i] = normal / norm if norm != 0 else normal  # unit normals

    # Spatial line shifts in pixels
    shifts = normals.round().astype(int)
    shifts = shifts[:, [1, 0]]  # swap x and y

    return mask_poly, shifts


def polygon2mask(points: list, shape: Tuple[int, int]) -> np.array:
    """
    Convert an enclosed polygon defined by a list of points into a binary mask.
    This implementation works with self-overlapping polygons.

    Args:
        points (list): List of points defining the polygon.
        shape (tuple): Shape of the output mask.

    Returns:
        numpy.ndarray: Binary mask representing the polygon.

    """    
    # Create mask with fillPoly
    mask = np.zeros(shape, dtype=np.uint8)
    mask = cv2.fillPoly(mask, [points], color=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 2:
        return mask

    # Get mask polylines and shifts
    mask_poly, shifts = walk_lines(points, mask)

    # Draw each contour
    n_lines = len(points)
    for i in range(len(contours)):
        mask_contour = np.zeros_like(mask)
        mask_contour = cv2.polylines(mask_contour, [contours[i]], isClosed=True, color=1, thickness=1)

        # Get the component
        component = mask_poly[mask_contour > 0]
        lines_rgb = np.unique(component, axis=0)
        lines_id = [rgb2dec(rgb) - 1 for rgb in lines_rgb]

        # Draw the inside lines (in the direction of the normal)
        mask_insides = np.zeros_like(mask)
        for line_a in lines_id:
            line_z = (line_a - 1) % n_lines
            line_b = (line_a + 1) % n_lines
            line_c = (line_a + 2) % n_lines

            z = points[line_z]
            a = points[line_a]
            b = points[line_b]
            c = points[line_c]

            z1 = z + shifts[line_z]
            a1 = a + shifts[line_z]
            lz1 = (z1, a1)

            a2 = a + shifts[line_a]
            b1 = b + shifts[line_a]
            la1 = (a2, b1)

            b2 = b + shifts[line_b]
            c1 = c + shifts[line_b]
            lb1 = (b2, c1)

            try:
                a0 = np.array(line_intersection(lz1, la1)).round().astype(int)
            except ZeroDivisionError:
                a0 = a2
            try:
                b0 = np.array(line_intersection(la1, lb1)).round().astype(int)
            except ZeroDivisionError:
                b0 = b1

            cv2.line(mask_insides, a0, b0, 1, 1)

        # Find the overlap between the component and the inside lines
        mask_component = cv2.fillPoly(np.zeros_like(mask), [contours[i]], color=1)
        ero_mask_component = cv2.erode(mask_component, np.ones((3, 3), np.uint8), iterations=1)
        mask_intersection = cv2.bitwise_and(mask_insides, ero_mask_component)

        if i == 0:
            # Check the direction of the polygon lines (clockwise or counterclockwise)
            # If the direction is counterclockwise, reverse the points
            mask_component_inv = cv2.bitwise_not(mask_component)
            mask_intersection_inv = cv2.bitwise_and(mask_insides, mask_component_inv)
            if mask_intersection_inv.any():
                # The "inside" lines are outside the polygon, so the points are reversed
                points = points[::-1]
                mask_poly, shifts = walk_lines(points, mask)
            continue

        if mask_intersection.any():
            mask = cv2.bitwise_or(mask, mask_component)

    return mask


if __name__ == "__main__":
    # Example usage
    points = np.array([[100, 100], [150, 200],[400, 200],[300, 400],[150,100],
                       [300, 300], [300, 250], [150, 300], [100, 200], [50, 300], 
                       [150, 150], [150, 170], [100, 250], [250, 250], [294, 258], 
                       [265, 280], [238, 257], [170, 280], [10, 400], [10, 200]])
    shape = (500, 500)

    ## Test the counterclockwise direction of the polygon
    # points = points[::-1]
    ##

    mask = polygon2mask(points, shape)

    # Comparison with the cv2.fillPoly function
    polylines = cv2.polylines(np.zeros(shape, dtype=np.uint8), [points], 
                              isClosed=True, 
                              color=255, 
                              thickness=1)
    fillPoly = cv2.fillPoly(np.zeros(shape, dtype=np.uint8), [points], color=1)

    cv2.imwrite("polygon2mask.png", mask * 255)
    cv2.imwrite("polylines.png", polylines)
    cv2.imwrite("fillPoly.png", fillPoly * 255)
    
