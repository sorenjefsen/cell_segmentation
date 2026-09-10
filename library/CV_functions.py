import numpy as np
def generate_circle_points(r_center, r):
    """Return integer pixel coordinates within a circle."""
    x_in = np.arange(r_center[0] - r, r_center[0] + r)
    y_in = np.arange(r_center[1] - r, r_center[1] + r)
    points_in = np.array(np.meshgrid(x_in, y_in)).T.reshape(-1, 2)
    points_in_circle = points_in[np.linalg.norm(points_in - r_center, axis=1) <= r]
    return points_in_circle

def calculate_circle_mean(test_array, points_in_circle):
    """Return the mean array value at the specified circle points."""
    # array indexing is [row, col] = [y, x], so we swap the point columns here  
    circle_pixels = test_array[points_in_circle[:, 1], points_in_circle[:, 0]]
    circle_mean = circle_pixels.mean()
    return circle_mean

# Test it
def generate_circle(test_array, r_min=10, r_max=101):
    """Generate a random circle and calculate its mean pixel value."""
    r = np.random.randint(r_min, r_max)
    r_center = np.random.randint(r, test_array.shape[0] - r, 2)
    points_in_circle = generate_circle_points(r_center, r)
    circle_mean = calculate_circle_mean(test_array, points_in_circle)
    return r_center, r, points_in_circle, circle_mean

# Lets write that into a function also
def generate_target_points(test_array, num_circles=50, threshold_factor=1.2, r_min=10, r_max=101, direction = "High"):
    """Return circle points whose values exceed a scaled circle mean."""
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