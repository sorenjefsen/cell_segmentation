import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.path.append(r"C:\Users\sjefs\Desktop\Uni\Speciale\Kode\library")
import CV_classes as CV_class


########################## Data loading ##########################

def load_image(name, number, channel = 1):
    """Load a microscopy image as a ``Picture`` instance for a given timepoint and channel.

    Args:
        name (str | None): Identifier for the dataset to load. Supported value is
            ``"timelapze_z"`` (note the existing spelling in the codebase).
        number (int): Time index to load. Must be an integer in the range 1..25.
        channel (int, optional): Image channel to load. Must be either 1 or 2.
            Defaults to ``1``.

    Returns:
        CV_classes.Picture | None: A ``Picture`` object containing the selected
        image if the recognized dataset name is provided; otherwise ``None``.

    Raises:
        ValueError: If ``name`` is ``None``, if ``number`` is outside the valid
        range, or if ``channel`` is outside the valid range.
    """
    if name is None:
        # Raise an error if the name is not provided
        raise ValueError("Image name must be provided.")

    if name == "timelapze_z":
        if number > 25 or number < 1 or  type(number) is not int:
            raise ValueError("Number must be an integer between 1 and 25 for timelapze_z.")
        if channel > 2 or channel < 1 or type(channel) is not int:
            raise ValueError("Channel must be an integer between 1 and 2 for timelapze_z.")

        
        path = rf"C:\Users\sjefs\Desktop\Uni\Speciale\Data\SharangGarudTopBP1data\timelapse_tiff_z_proj\topbp1-timelapse-jf646-10_2_R3D_channel{channel}\topbp1-timelapse-jf646-10_2_R3D_channel{channel}_t{number}_maxZ.tif"
        return CV_class.Picture(path)
    
    return None




########################## CIRCLE ADAPTIVE THRESHOLDING AND TARGET POINT GENERATION ##########################
def generate_circle_points(r_center, r):
    """Generate pixel coordinates for all points inside a circle.

    Args:
        r_center (tuple[int, int]): Center of the circle as ``(x, y)`` pixel
            coordinates.
        r (int): Circle radius in pixels.

    Returns:
        np.ndarray: Array of shape ``(N, 2)`` containing integer ``(x, y)``
        coordinates inside the circle.
    """
    x_in = np.arange(r_center[0] - r, r_center[0] + r)
    y_in = np.arange(r_center[1] - r, r_center[1] + r)
    points_in = np.array(np.meshgrid(x_in, y_in)).T.reshape(-1, 2)
    points_in_circle = points_in[np.linalg.norm(points_in - r_center, axis=1) <= r]
    return points_in_circle


def calculate_circle_mean(test_array, points_in_circle):
    """Compute the mean pixel value at a set of circle coordinates.

    Args:
        test_array (np.ndarray): 2D image array.
        points_in_circle (np.ndarray): Array of shape ``(N, 2)`` with ``(x, y)``
            coordinates.

    Returns:
        float: Mean intensity value of the pixels at the specified points.
    """
    # array indexing is [row, col] = [y, x], so we swap the point columns here  
    circle_pixels = test_array[points_in_circle[:, 1], points_in_circle[:, 0]]
    circle_mean = circle_pixels.mean()
    return circle_mean

# Test it
def generate_circle(test_array, r_min=10, r_max=101):
    """Generate a random circular region and compute its average pixel value.

    Args:
        test_array (np.ndarray): Image data from which the circle is sampled.
        r_min (int, optional): Minimum radius. Defaults to ``10``.
        r_max (int, optional): Maximum radius. Defaults to ``101``.

    Returns:
        tuple: ``(r_center, r, points_in_circle, circle_mean)`` where ``r_center``
        is the circle center, ``r`` is the radius, ``points_in_circle`` are the
        sampled coordinates, and ``circle_mean`` is the mean pixel value.
    """
    r = np.random.randint(r_min, r_max)
    r_center = np.random.randint(r, test_array.shape[0] - r, 2)
    points_in_circle = generate_circle_points(r_center, r)
    circle_mean = calculate_circle_mean(test_array, points_in_circle)
    return r_center, r, points_in_circle, circle_mean

# Lets write that into a function also
def generate_target_points(test_array, num_circles=50, threshold_factor=1.2, r_min=10, r_max=101, direction = "High"):
    """Generate candidate target pixels from random circular neighborhoods.

    Args:
        test_array (np.ndarray): Image array to analyze.
        num_circles (int, optional): Number of random circles to sample.
            Defaults to ``50``.
        threshold_factor (float, optional): Multiplicative threshold used to
            decide whether a point is unusually bright or dark relative to its
            local circle mean. Defaults to ``1.2``.
        r_min (int, optional): Minimum circle radius. Defaults to ``10``.
        r_max (int, optional): Maximum circle radius. Defaults to ``101``.
        direction (str, optional): Either ``"High"`` for bright outliers or
            ``"Low"`` for dark outliers. Defaults to ``"High"``.

    Returns:
        np.ndarray: Array of ``(x, y)`` points satisfying the threshold criterion.
    """
    target_points = np.array([])
    for _ in range(num_circles):
        r_center, r, points_in_circle, circle_mean = generate_circle(test_array, r_min=r_min, r_max=r_max)
        # We sort out the target points based on the direction
        if direction == "High":
            Targets = points_in_circle[test_array[points_in_circle[:, 1], points_in_circle[:, 0]] > circle_mean*threshold_factor]
        else:
            Targets = points_in_circle[test_array[points_in_circle[:, 1], points_in_circle[:, 0]] < circle_mean/threshold_factor]
        target_points = np.append(target_points, Targets, axis=0) if target_points.size else Targets
    return target_points


# test it
#target_points = generate_target_points(test_array, num_circles=100, threshold_factor=1.2, r_min=40, r_max=80)
#plt.scatter(target_points[:, 0], target_points[:, 1], s=1, c='blue', alpha=0.3)
#plt.imshow(test_array, cmap='gray')




##############  Growing areas for segmenting ############################

def adaptive_threshold(array, blockSize=101, C=-20):
    """Apply adaptive Gaussian thresholding to a grayscale image.

    Args:
        array (np.ndarray): 2D grayscale image array.
        blockSize (int, optional): Size of the pixel neighborhood used by the
            adaptive thresholding algorithm. Must be odd. Defaults to ``101``.
        C (int, optional): Constant subtracted from the mean. Defaults to ``-20``.

    Returns:
        np.ndarray: Binary thresholded image with foreground pixels set to 255.
    """

    return cv2.adaptiveThreshold(
        array,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=blockSize,
        C=-C
    )

def get_marked_points(th1):
    """Return all non-zero foreground pixels from a thresholded image.

    Args:
        th1 (numpy.ndarray): Binary or thresholded image array.

    Returns:
        list[tuple[int, int]]: Coordinates ``(x, y)`` for all foreground pixels.
    """
    marked_points = []
    for y in range(th1.shape[0]):
        for x in range(th1.shape[1]):
            if th1[y, x] > 0:
                marked_points.append((x, y))
    return marked_points


def region_growing(th1, start_points, threshold=0, min_size = 1):
    """Grow 8-connected foreground regions from candidate seed points.

    Args:
        th1 (numpy.ndarray): Binary image where foreground pixels are non-zero.
        start_points (list[tuple[int, int]]): Seed points ``(x, y)`` used to
            start region expansion.
        threshold (int, optional): Minimum pixel value required for a neighbor to
            be included in the region. Defaults to ``0``.
        min_size (int, optional): Minimum number of pixels required for a region
            to be kept. Defaults to ``1``.

    Returns:
        list[list[tuple[int, int]]]: A list of segmented regions, where each
        region is a list of ``(x, y)`` pixel coordinates.
    """

    visited = set()
    segments = []

    for start_point in start_points:
        x, y = start_point

        # Skip invalid, background, or already-segmented starting points
        if (
            not (0 <= x < th1.shape[1] and 0 <= y < th1.shape[0])
            or th1[y, x] == 0
            or start_point in visited
        ):
            continue

        segment = []
        points_to_try = [start_point]
        visited.add(start_point)

        while points_to_try:
            current_x, current_y = points_to_try.pop()
            segment.append((current_x, current_y))

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = (current_x + dx, current_y + dy)
                    nx, ny = neighbor

                    if (
                        0 <= nx < th1.shape[1]
                        and 0 <= ny < th1.shape[0]
                        and th1[ny, nx] > threshold
                        and neighbor not in visited
                    ):
                        visited.add(neighbor)
                        points_to_try.append(neighbor)

        # If segment is too small, skip it
        if len(segment) >= min_size:
            segments.append(segment)

    return segments


#Using the region growing function would look like this:
# segments = region_growing(th1, marked_points)
#
# # Visualize the segments
# import matplotlib.pyplot as plt
# import random
#
# plt.imshow(th1, cmap="gray")
# for segment in segments:
#     color = (random.random(), random.random(), random.random())
#     plt.scatter(
#         [p[0] for p in segment],
#         [p[1] for p in segment],
#         marker=".",
#         color=color
#     )
# plt.show()

def plot_segment(segment):
    """Plot a single image segment as a random-colored scatter cloud.

    Args:
        segment (list[tuple[int, int]]): Coordinates ``(x, y)`` belonging to one
            connected segment.

    Returns:
        None: Displays the segment in the active matplotlib axes via ``plt.scatter``.
    """

    color = (random.random(), random.random(), random.random())
    plt.scatter(
        [p[0] for p in segment],
        [p[1] for p in segment],
        marker=".",
        color=color
    )


def get_largest_segment(segments):
    """Return the segment with the greatest number of pixels.

    Args:
        segments (list[list[tuple[int, int]]]): A list of segmented regions, each
            containing ``(x, y)`` coordinates.

    Returns:
        list[tuple[int, int]]: The largest segment in the input list.
    """
    segment_areas = [len(segment) for segment in segments]
    largest_segment = segments[segment_areas.index(max(segment_areas))]

    # Convert to numpy
    largest_segment = np.array(largest_segment)
    return largest_segment

# Example usage:
# largest_segment = get_largest_segment(segments)


def get_center_of_mass(segment):
    """Compute the centroid of a connected segment.

    Args:
        segment (list[tuple[int, int]]): Pixel coordinates ``(x, y)`` belonging
            to the segment.

    Returns:
        tuple[float, float]: The centroid coordinates ``(x_center, y_center)``.
    """
    center_of_mass_x = sum(p[0] for p in segment) / len(segment)
    center_of_mass_y = sum(p[1] for p in segment) / len(segment)
    return (center_of_mass_x, center_of_mass_y)


def get_area(segment):
    """Return the number of pixels in a segment.

    Args:
        segment (list[tuple[int, int]]): Pixel coordinates ``(x, y)`` belonging
            to the segment.

    Returns:
        int: Number of points in the segment.
    """
    return len(segment)

def get_spread(segment):
    """Compute the bounding-box spread of a segment in x and y directions.

    Args:
        segment (list[tuple[int, int]]): Pixel coordinates ``(x, y)`` belonging
            to the segment.

    Returns:
        tuple[int, int]: ``(x_spread, y_spread)`` as the difference between max
        and min coordinates in each axis.
    """
    x_coords = [p[0] for p in segment]
    y_coords = [p[1] for p in segment]
    x_spread = max(x_coords) - min(x_coords)
    y_spread = max(y_coords) - min(y_coords)
    return (x_spread, y_spread)



def get_bounding_box(original_array, segment, scale = 1):
    """
    Compute a scaled, centered bounding box (ROI) around a segment.

    Parameters
    ----------
    original_array : np.ndarray
        The image array the segment belongs to (used for boundary clipping).
    segment : np.ndarray
        The segment points as an array with shape ``(N, 2)`` containing
        ``(x, y)`` pairs.
    scale : float
        Scale factor applied to the segment's bounding-box size.

    Returns
    -------
    min_x, max_x, min_y, max_y : int
        The clipped bounding box coordinates.
    """
    segment_array = np.asarray(segment)
    if segment_array.ndim != 2 or segment_array.shape[1] != 2 or segment_array.shape[0] == 0:
        raise ValueError("segment must be a non-empty array with shape (N, 2)")

    # NumPy arrays are indexed row-wise, so calculate the centroid by column.
    center_of_mass = segment_array.mean(axis=0)

    # Bounding box of the segment
    min_coords = segment_array.min(axis=0)
    max_coords = segment_array.max(axis=0)

    # Use one side length for both axes so the ROI is square.
    segment_size = max_coords - min_coords + 1
    roi_side = max(1, int(np.ceil(np.max(segment_size) * scale)))

    # Keep the ROI centered on the segment's center of mass
    center = np.asarray(center_of_mass)
    circle_min_coords = np.floor(center - roi_side / 2).astype(int)
    circle_max_coords = circle_min_coords + roi_side

    # Shift (rather than independently clip) the ROI to preserve its square
    # shape when the center is close to an image boundary.
    height, width = original_array.shape[:2]
    image_size = np.array([width, height])
    roi_side = min(roi_side, width, height)
    circle_min_coords = np.floor(center - roi_side / 2).astype(int)
    circle_min_coords = np.maximum(circle_min_coords, 0)
    circle_min_coords = np.minimum(circle_min_coords, image_size - roi_side)
    circle_max_coords = circle_min_coords + roi_side

    # Crop channel 2
    cropped_image = original_array[
        circle_min_coords[1]:circle_max_coords[1],
        circle_min_coords[0]:circle_max_coords[0]
    ]
    return cropped_image