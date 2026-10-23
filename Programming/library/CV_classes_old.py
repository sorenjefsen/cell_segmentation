import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import mrc
import os
import re
import sys
sys.path.append(r"C:\Users\sjefs\Desktop\Uni\Speciale\Kode\library")
import CV_functions as CV_func

# OLD VERSION OF CV_classes.py. HERE WE CAN LOAD IMAGES DIRECTLY FROM PATH. 


class Time_series():
    """
    Representation of a time series of microscopy images loaded from disk.

    This class stores the sequence of images as a 3D numpy array and provides
    methods for accessing and processing the time series data.

    Takes in the location of the folder containing the time series image files.
    """
    def __init__(self, path: str, run_number, channel):
        self.name = path.split("\\")[-1]
        self.load_time_series(path, run_number, channel)
        self.stacks = np.array([None])

    def load_time_series(self, path: str, run_number, channel) -> None:
        """
        Load the time series images from disk for the specified channel. Sets the array
        ``self.array`` that can be acessed as a 3D numpy array with shape (time, z, height, width).

        Args:
            path (str): Path to the folder containing the image files.
            run_number (str): Run number identifier for the image files.
            channel (int): Channel number to load (1 or 2).

        Raises:
            ValueError: If an invalid channel number is provided.
        """
        path_to_folder = r"C:\Users\sjefs\Desktop\Uni\Speciale\Data\SharangGarudTopBP1data_raw"
        filename = os.path.join(path_to_folder, f"topbp1-timelapse-jf646-{run_number}_R3D.dv")

        raw_folder = r"C:\Users\sjefs\Desktop\Uni\Speciale\Data\SharangGarudTopBP1data_raw\\"

        filename = f"topbp1-timelapse-jf646-{run_number}_R3D.dv"
        path_data = raw_folder + filename
        path_brightfield = raw_folder + filename[:-3] + "_REF.dv"

        image = mrc.load(path_data)
        image_brightfield = mrc.load(path_brightfield)

        channel_1 = image[::2].reshape(-1, 20, *image.shape[1:])
        channel_2 = image[1::2].reshape(-1, 20, *image.shape[1:])


        if channel == 1:
            self.array = channel_1
        elif channel == 2:
            self.array = channel_2
        else:
            raise ValueError("Invalid channel number. Must be 1 or 2.")


    def display(self):
        """Interactive plot of the time series with sliders for time and z."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        # The inline backend renders static images, so sliders do nothing.
        if "inline" in plt.get_backend().lower():
            try:
                from IPython import get_ipython

                ip = get_ipython()
                if ip is not None:
                    for backend in ("widget", "qt"):
                        try:
                            ip.run_line_magic("matplotlib", backend)
                            break
                        except Exception:
                            continue
            except (ImportError, NameError):
                pass

        arr = self.array
        n_t, n_z = arr.shape[0], arr.shape[1]

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        im = ax.imshow(arr[0, 0], cmap="gray", vmin=arr.min(), vmax=arr.max())

        ax_t = plt.axes([0.2, 0.12, 0.6, 0.03])
        ax_z = plt.axes([0.2, 0.05, 0.6, 0.03])
        s_t = Slider(ax_t, "Time", 0, max(n_t - 1, 1), valinit=0, valstep=1)
        s_z = Slider(ax_z, "Z", 0, max(n_z - 1, 1), valinit=0, valstep=1)

        def update(_):
            t, z = int(s_t.val), int(s_z.val)
            t = min(t, n_t - 1)
            z = min(z, n_z - 1)
            im.set_data(arr[t, z])
            ax.set_title(f"t={t}, z={z}")
            fig.canvas.draw_idle()

        s_t.on_changed(update)
        s_z.on_changed(update)
        update(None)
        # keep references so widgets are not garbage collected
        self._sliders = (s_t, s_z)
        plt.show()


class Stack():
    def __init__(self, array):

        self.array = array
        self.parent = None






class Picture():
    """Representation of a single microscopy image loaded from disk.

    This class stores the original image array, the current working image array,
    and metadata parsed from the filename. It supports resizing, visualisation,
    basic statistics, and common preprocessing operations used in image analysis.

    Args:
        path (str): Path to the image file. The filename is expected to contain
            a timestamp in the form "_t<integer>" so that ``self.t`` can be
            extracted.

    Attributes:
        path (str): Original file path.
        name (str): File name without directory information.
        t (int): Timestamp parsed from the filename.
        array (np.ndarray): Current working image array.
        array_original (np.ndarray): Copy of the original image before any
            preprocessing.
        blured (bool): Whether Gaussian blur has been applied.
        normalized (bool): Whether the array has been min-max normalized.
        contrasted (bool): Whether CLAHE contrast enhancement has been applied.
        downsized (bool): Flag indicating whether downsampling was performed.
    """

    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    def __init__(self, path: str):
        """Initialize the picture object and load the image data.

        Args:
            path (str): File path to the image.

        Returns:
            None: Stores the loaded image and metadata on the instance.
        """
        self.path = path
        self.name = path.split("\\")[-1]
        self.load_image(path)
        self.array_original = self.array.copy()

        # Things related to segments

        self.segments = []
        self.used_labels = [0]

        self.largest_segment = None

        # (x, y) offset of self.array's origin within array_original, set by crop_from_original
        self.offset = (0, 0)

        # Make sure we are not performing preprocessing more than once
        self.blured = False
        self.normalized = False
        self.contrasted = False
        self.downsized = False

    def load_image(self, path: str) -> None:
        """Load the image from disk and extract its timestamp.

        Args:
            path (str): File path to the image.

        Returns:
            None: Updates ``self.array`` and ``self.t``.
        """
        if path.lower().endswith(".npy"):
            self.array = np.load(path)
        else:
            self.array = np.array(Image.open(path))

        # Find timestamp, supports both "_t0" and "_t_0" style filenames
        match = re.search(r"_t_?(\d+)", self.name)
        self.t = int(match.group(1)) if match else None

    def add_segment(self, segment) -> None:
        """Add a new segment to the image.

        Args:
            segment (Segment): The segment object to be added.

        Returns:
            None: Updates ``self.segments`` and ``self.used_labels``.
        """
        self.segments.append(segment)
        self.used_labels.append(segment.label)
        if self.largest_segment is None or segment.size > self.largest_segment.size:
            self.largest_segment = segment

    def downsize(self, factor: float) -> None:
        """Reduce the image size for faster testing or previewing.

        Args:
            factor (float): If less than 1, the image is resized by interpolation.
                If greater than 1, a central crop is taken using the provided
                integer size. Values <= 0 are invalid.

        Returns:
            None: Modifies ``self.array`` in place.

        Raises:
            ValueError: If ``factor`` is less than or equal to zero.
        """
        if factor <= 0:
            raise ValueError("Downsize factor must be positive")
        from cv2 import resize

        if factor < 1:
            new_size = (int(self.array.shape[1] * factor), int(self.array.shape[0] * factor))
            self.array = resize(self.array, new_size)
        if factor > 1:
            self.array = self.array[:factor, :factor]
        if factor == 1:
            pass

    def zoom(self, min_x, max_x, min_y, max_y) -> None:
        """Crop the image to the specified rectangular region.

        Args:
            min_x (int): Minimum x-coordinate of the crop region.
            max_x (int): Maximum x-coordinate of the crop region.
            min_y (int): Minimum y-coordinate of the crop region.
            max_y (int): Maximum y-coordinate of the crop region.

        Returns:
            None: Updates ``self.array`` in place.
        """
        self.array = self.array[min_y:max_y, min_x:max_x]
        offset_x, offset_y = self.offset
        self.offset = (offset_x + min_x, offset_y + min_y)

    def crop_from_original(self, min_y, max_y, min_x, max_x) -> None:
        """Crop ``array_original`` to the given region and record the crop offset.

        Recording the offset lets segments created from this crop be compared
        with segments from differently-cropped images using global coordinates.

        Args:
            min_y (int): Minimum y-coordinate of the crop region.
            max_y (int): Maximum y-coordinate of the crop region.
            min_x (int): Minimum x-coordinate of the crop region.
            max_x (int): Maximum x-coordinate of the crop region.

        Returns:
            None: Updates ``self.array`` and ``self.offset`` in place.

        Raises:
            ValueError: If the resulting crop region is empty.
        """
        height, width = self.array_original.shape[:2]
        # Clamp defensively: negative indices would otherwise be interpreted as
        # "from the end" by numpy/Python slicing instead of being clipped to 0.
        min_y = max(0, min(min_y, height))
        max_y = max(0, min(max_y, height))
        min_x = max(0, min(min_x, width))
        max_x = max(0, min(max_x, width))

        if min_y >= max_y or min_x >= max_x:
            raise ValueError(
                f"Invalid crop region: min_y={min_y}, max_y={max_y}, min_x={min_x}, max_x={max_x}"
            )

        self.array = self.array_original[min_y:max_y, min_x:max_x]
        self.offset = (min_x, min_y)


    def display(self, show_segments: bool = False, original: bool = False, name: str = None, show_axis: bool = True, show_colorbar: bool = True, show_data: bool = False) -> None:
        """Display the current image or the original image.

        Args:
            show_segments (bool, optional): If ``True``, display the image segments. Defaults to ``False``.
            original (bool, optional): If ``True``, display the unprocessed image
                stored in ``array_original``. Defaults to ``False``.
            name (str, optional): Title for the displayed image. Defaults to ``None``.
            show_axis (bool, optional): If ``False``, hide the axis in the display. Defaults to ``True``.
            show_colorbar (bool, optional): If ``True``, display a colorbar alongside the image. Defaults to ``True``.
            show_data (bool, optional): If ``True``, display the image data values on the plot. Defaults to ``False``.

        Returns:
            None: Opens a matplotlib figure window.
        """
        if original:
            plt.imshow(self.array_original, cmap='gray')
        else:
            plt.imshow(self.array, cmap='gray')
        plt.title(name if name is not None else f"{self.name} + Time = {self.t}")
        if not show_axis:
            plt.axis('off')
        if show_data:
            plt.text(0, 0, f"Average: {self.average(original=original):.2f}", ha='left', va='top', color='red')
        if show_colorbar:
            plt.colorbar()
        if show_segments:
            for segment in self.segments:
                segment.display(annotate_label=True)

    def reset(self, all = False):
        """Reset the current image array to the original image array.

        Args:
            all (bool, optional): If ``True``, reloads the image from ``self.path``
                and restores all instance state, as if freshly constructed.
        """
        self.array = self.array_original.copy()
        if all:
            self.load_image(self.path)
            self.array_original = self.array.copy()

            self.segments = []
            self.used_labels = [0]
            self.largest_segment = None
            self.offset = (0, 0)

            self.blured = False
            self.normalized = False
            self.contrasted = False
            self.downsized = False

    ################# Extract data ######################

    def average(self, original: bool = False) -> float:
        """Return the mean pixel intensity of the image.

        Args:
            original (bool, optional): If ``True``, compute the mean over the
                original image array instead of the current processed one.

        Returns:
            float: Mean intensity value across all pixels in the selected array.
        """
        if original:
            return np.mean(self.array_original)
        return np.mean(self.array)

    def histogram(self, plot = True):
        """ Option to generate histogram of intensities"""
        # ravel
        image_ravel = self.array.ravel()
        hist, bins = np.histogram(image_ravel, bins=256)

        if plot:
            plt.plot(bins[:-1], hist)
            plt.title(f"Histogram of {self.name}")
        return hist, bins

    ################### PRE PROCESSING ##########################
    def pre_process(self, ksize=(5, 5), sigma=1.4, clip_limit=2.0, tile_grid_size=(8, 8)) -> None:
        """Run the standard preprocessing pipeline for the image.

        This method executes blur, grayscale conversion, normalization, and CLAHE
        contrast enhancement in sequence.

        Args:
            ksize (tuple[int, int], optional): Gaussian kernel size. Must be odd
                in both dimensions. Defaults to ``(5, 5)``.
            sigma (float, optional): Standard deviation for Gaussian blur.
                Defaults to ``1.4``.
            clip_limit (float, optional): CLAHE contrast limit. Defaults to ``2.0``.
            tile_grid_size (tuple[int, int], optional): CLAHE grid size.
                Defaults to ``(8, 8)``.

        Returns:
            None: Updates ``self.array`` in place.
        """
        self.blur(ksize=ksize, sigma=sigma)
        self.gray()
        self.normalize()
        self.contrast(clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    def blur(self, ksize=(5, 5), sigma=1.4) -> None:
        """Apply Gaussian blur to the current image array.

        Args:
            ksize (tuple[int, int], optional): Blur kernel size in pixels.
            sigma (float, optional): Gaussian standard deviation.

        Returns:
            None: Replaces ``self.array`` with the blurred version.

        Raises:
            ValueError: If either kernel dimension is even.
        """
        if ksize[0] % 2 == 0 or ksize[1] % 2 == 0:
            raise ValueError("Kernel size must be odd in both dimensions")
        from cv2 import GaussianBlur
        self.array = GaussianBlur(self.array, ksize, sigma)
        self.array = self.array.astype(np.float32)
        self.blured = True

    def diff_of_gauss(self, ksize1=(5, 5), sigma1=1.0, ksize2=(5, 5), sigma2=2.0) -> None:
        """Apply Difference of Gaussians (DoG) to the current image array.

        Args:
            ksize1 (tuple[int, int], optional): Kernel size for the first Gaussian blur.
                Defaults to ``(5, 5)``.
            sigma1 (float, optional): Standard deviation for the first Gaussian blur.
                Defaults to ``1.0``.
            ksize2 (tuple[int, int], optional): Kernel size for the second Gaussian blur.
                Defaults to ``(5, 5)``.
            sigma2 (float, optional): Standard deviation for the second Gaussian blur.
                Defaults to ``2.0``.

        Returns:
            None: Updates ``self.array`` with the DoG result.
        """
        from cv2 import GaussianBlur
        blur_1 = GaussianBlur(self.array, ksize1, sigma1)
        blur_2 = GaussianBlur(self.array, ksize2, sigma2)
        self.array = blur_1 - blur_2
        self.array = self.array.astype(np.float32)

    def gray(self) -> None:
        """Convert the image array to floating-point grayscale format.

        Returns:
            None: Casts ``self.array`` to ``np.float32``.
        """
        self.array = self.array.astype(np.float32)
        self.grayed = True

    def normalize(self) -> None:
        """Normalize the image intensity range to ``[0, 255]``.

        Returns:
            None: Converts the current array to uint8 with min-max normalization.
        """
        self.array = cv2.normalize(self.array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.normalized = True

    def contrast(self, clip_limit=2.0, tile_grid_size=(8,8)) -> None:
        """Apply CLAHE contrast enhancement to the image.

        Args:
            clip_limit (float, optional): Threshold for contrast limiting.
                Defaults to ``2.0``.
            tile_grid_size (tuple[int, int], optional): Grid partition size for
                CLAHE. Defaults to ``(8, 8)``.

        Returns:
            None: Updates ``self.array`` with higher-contrast values.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.array = clahe.apply(self.array).astype(np.float32)
        self.contrasted = True

    def format(self, dtype=np.float32) -> None:
        """Cast the current image array to a requested NumPy dtype.

        Args:
            dtype (type, optional): Target NumPy dtype, such as ``np.float32`` or
                ``np.uint8``. Defaults to ``np.float32``.

        Returns:
            None: Replaces ``self.array`` with the new dtype.
        """
        self.array = self.array.astype(dtype)

    def erosion(self, ksize=(3, 3), iterations=1, shape=cv2.MORPH_RECT) -> None:

        """Apply morphological erosion to the image.

        Args:
            kernel_size (tuple[int, int], optional): Size of the structuring element.
                Defaults to ``(3, 3)``.
            iterations (int, optional): Number of times erosion is applied.
                Defaults to ``1``.
            shape (int, optional): Structuring element shape, e.g. ``cv2.MORPH_RECT``
                or ``cv2.MORPH_ELLIPSE`` for a circular kernel. Defaults to ``cv2.MORPH_RECT``.

        Returns:
            None: Updates ``self.array`` with the eroded image.
        """
        kernel = cv2.getStructuringElement(shape, ksize)

        self.array = cv2.erode(self.array, kernel, iterations=iterations)

    def dilation(self, ksize=(3, 3), iterations=1, shape=cv2.MORPH_RECT) -> None:

        """Apply morphological dilation to the image.

        Args:
            ksize (tuple[int, int], optional): Size of the structuring element.
                Defaults to ``(3, 3)``.
            iterations (int, optional): Number of times dilation is applied.
                Defaults to ``1``.
            shape (int, optional): Structuring element shape, e.g. ``cv2.MORPH_RECT``
                or ``cv2.MORPH_ELLIPSE`` for a circular kernel. Defaults to ``cv2.MORPH_RECT``.

        Returns:
            None: Updates ``self.array`` with the dilated image.
        """
        kernel = cv2.getStructuringElement(shape, ksize)
        self.array = cv2.dilate(self.array, kernel, iterations=iterations)

    def opening(self, ksize=(3, 3), shape=cv2.MORPH_RECT) -> None:

        """Apply morphological opening to the image.

        Args:
            ksize (tuple[int, int], optional): Size of the structuring element.
                Defaults to ``(3, 3)``.
            shape (int, optional): Structuring element shape, e.g. ``cv2.MORPH_RECT``
                or ``cv2.MORPH_ELLIPSE`` for a circular kernel. Defaults to ``cv2.MORPH_RECT``.

        Returns:
            None: Updates ``self.array`` with the opened image.
        """
        kernel = cv2.getStructuringElement(shape, ksize)
        self.array = cv2.morphologyEx(self.array, cv2.MORPH_OPEN, kernel)

    def closing(self, ksize=(3, 3), shape=cv2.MORPH_RECT) -> None:

        """Apply morphological closing to the image.

        Args:
            ksize (tuple[int, int], optional): Size of the structuring element.
                Defaults to ``(3, 3)``.
            shape (int, optional): Structuring element shape, e.g. ``cv2.MORPH_RECT``
                or ``cv2.MORPH_ELLIPSE`` for a circular kernel. Defaults to ``cv2.MORPH_RECT``.

        Returns:
            None: Updates ``self.array`` with the closed image.
        """
        kernel = cv2.getStructuringElement(shape, ksize)
        self.array = cv2.morphologyEx(self.array, cv2.MORPH_CLOSE, kernel)

    def white_tophat(self, ksize=(3, 3), shape=cv2.MORPH_RECT) -> None:

        """Apply morphological white tophat to the image.

        Args:
            ksize (tuple[int, int], optional): Size of the structuring element.
                Defaults to ``(3, 3)``.
            shape (int, optional): Structuring element shape, e.g. ``cv2.MORPH_RECT``
                or ``cv2.MORPH_ELLIPSE`` for a circular kernel. Defaults to ``cv2.MORPH_RECT``.

        Returns:
            None: Updates ``self.array`` with the white tophat image.
        """
        kernel = cv2.getStructuringElement(shape, ksize)
        self.array = cv2.morphologyEx(self.array, cv2.MORPH_TOPHAT, kernel)

    def black_tophat(self, ksize=(3, 3), shape=cv2.MORPH_RECT) -> None:

        """Apply morphological black tophat to the image.

        Args:
            ksize (tuple[int, int], optional): Size of the structuring element.
                Defaults to ``(3, 3)``.
            shape (int, optional): Structuring element shape, e.g. ``cv2.MORPH_RECT``
                or ``cv2.MORPH_ELLIPSE`` for a circular kernel. Defaults to ``cv2.MORPH_RECT``.

        Returns:
            None: Updates ``self.array`` with the black tophat image.
        """
        kernel = cv2.getStructuringElement(shape, ksize)
        self.array = cv2.morphologyEx(self.array, cv2.MORPH_BLACKHAT, kernel)

    def morphological_gradient(self, ksize=(3, 3), shape=cv2.MORPH_RECT) -> None:

        """Apply morphological gradient to the image.

        Args:
            ksize (tuple[int, int], optional): Size of the structuring element.
                Defaults to ``(3, 3)``.
            shape (int, optional): Structuring element shape, e.g. ``cv2.MORPH_RECT``
                or ``cv2.MORPH_ELLIPSE`` for a circular kernel. Defaults to ``cv2.MORPH_RECT``.

        Returns:
            None: Updates ``self.array`` with the morphological gradient image.
        """
        kernel = cv2.getStructuringElement(shape, ksize)
        self.array = cv2.morphologyEx(self.array, cv2.MORPH_GRADIENT, kernel)


###################### Image CV processing ##########################
    def adaptive_threshold(self, block_size=101, C=-20, method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C) -> None:
        """Apply adaptive thresholding to the image.

        Args:
            block_size (int, optional): Size of the neighborhood used to calculate the threshold.
                Must be an odd number. Defaults to ``101``.
            C (int, optional): Constant subtracted from the mean or weighted mean.
                Defaults to ``-20``.

        Returns:
            th (np.ndarray): The binary image resulting from adaptive thresholding.
        """
        # Create a copy of the array
        temp_array = self.array.copy()

        # Perform some format conversions to ensure the array is suitable for adaptive thresholding
        if temp_array.ndim == 3:
            temp_array = cv2.cvtColor(temp_array, cv2.COLOR_BGR2GRAY)
        if temp_array.dtype != np.uint8:
            temp_array = cv2.normalize(
                temp_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

        th = cv2.adaptiveThreshold(
            temp_array,
            255,
            method,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C
        )
        self.th = th
        return th

    def thresholding(self, thresh: int = 127, max_value: int = 255, invert = False) -> None:
        """Apply simple thresholding to the image.

        Args:
            thresh (int, optional): Threshold value. Defaults to ``127``.
            max_value (int, optional): Maximum value to use with the threshold. Defaults to ``255``.

        Returns:
            th (np.ndarray): The binary image resulting from thresholding.
        """
        if invert:
            _, th = cv2.threshold(self.array, thresh, max_value, cv2.THRESH_BINARY_INV)
        else:
            _, th = cv2.threshold(self.array, thresh, max_value, cv2.THRESH_BINARY)
        self.th = th
        return th



class Segment:
    def __init__(self, points: np.ndarray, parent: "Picture") -> None:
        self.mask = np.zeros_like(parent.array, dtype=np.uint8)
        self.points = np.array(points)
        for point in points:
            # points are (x, y); mask indexing is [row, col] = [y, x]
            self.mask[point[1], point[0]] = 1
        self.parent = parent
        # 0 means unassigned
        self.label = 0
        self.size = len(points)
        self.center_of_mass = self.get_center_of_mass()

    def get_global_points(self) -> np.ndarray:
        """Return this segment's points shifted into the parent image's original coordinate frame.

        This makes segments from differently-cropped images comparable.
        """
        offset_x, offset_y = self.parent.offset
        return self.points + np.array([offset_x, offset_y])

    def get_center_of_mass(self) -> np.ndarray:
        """Return the center of mass of the segment in the parent image's original coordinate frame."""
        global_points = self.get_global_points()
        if len(global_points) == 0:
            return np.array([0, 0])
        return np.mean(global_points, axis=0)

    def get_area(self) -> int:
        """Calculate the area of the segment based on its mask."""
        return np.sum(self.mask)

    def get_mean_intensity(self) -> float:
        """Calculate the mean intensity of the segment within the parent image."""
        return np.mean(self.parent.array[self.mask > 0])

    def display(self, details = "on top", annotate_label: bool = True) -> None:
        """Display the segment as an image."""
        # tab20 has 20 distinct colors, cycle through them based on the label
        color = plt.get_cmap('tab20')(self.label % 20)

        if details == "with original":
            plt.imshow(self.parent.array, cmap='gray')
            plt.scatter(self.points[:, 0], self.points[:, 1], c=[color], s=1)
        if details == "mask only":
            plt.imshow(self.mask, cmap='gray')
        if details == "on top":
            plt.scatter(self.points[:, 0], self.points[:, 1], c=[color], s=1)

        else:
            
            plt.imshow(self.mask, cmap='gray')

        # Also display center of mass as a blue cross (converted back to local/plot coordinates)
        offset_x, offset_y = self.parent.offset
        center_of_mass = self.get_center_of_mass() - np.array([offset_x, offset_y])
        plt.scatter(center_of_mass[0], center_of_mass[1], c='blue', s=10, marker='x')
  
  


        # Annotate the label next to the center of the segment
        if annotate_label:
            y_indices, x_indices = np.where(self.mask > 0)
            if len(x_indices) > 0 and len(y_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                plt.text(center_x, center_y, self.label, color='red', fontsize=12)


    def get_overlap(self, other_segment: "Segment") -> float:
        """Calculate the overlap ratio between this segment and another segment.

        Points are compared in the parent images' original coordinate frame, so
        this works even if the two segments come from differently-cropped images.

        Args:
            other_segment (Segment): The other segment to compare with.

        Returns:
            float: The overlap ratio, defined as the intersection area divided by the union area.
        """
        self_points = {tuple(point) for point in self.get_global_points()}
        other_points = {tuple(point) for point in other_segment.get_global_points()}
        union = self_points | other_points
        if not union:
            return 0.0
        intersection = self_points & other_points
        return len(intersection) / len(union)


    def assign_label(self, other_image: "Picture" = None, max_range: float = 10) -> None:
        """Assign a label to this segment.

        If this segment has no overlap with any segment in ``other_image``, it is
        assigned the label of the closest segment (by centroid distance) within
        ``max_range``. If no such segment exists, a new label is created.

        If ``other_image`` is not given, the label is instead assigned by this
        segment's size rank among its parent image's other segments (largest = 1).

        Args:
            other_image (Picture, optional): The previous image to match this
                segment against. If ``None``, the label is assigned by size rank
                instead.
            max_range (float, optional): Maximum centroid distance allowed when
                falling back to nearest-segment matching. Defaults to ``10``.
        """
        if other_image is None:
            # Re-rank every segment in this frame together (largest = label 1), including
            # this segment even if it hasn't been added to the parent image yet. id() breaks
            # ties between equally-sized segments so labels stay unique.
            siblings = list(self.parent.segments)
            if all(segment is not self for segment in siblings):
                siblings.append(self)
            siblings.sort(key=lambda segment: (-segment.size, id(segment)))
            for label, segment in enumerate(siblings, start=1):
                segment.label = label
            return

        # Labels already claimed this frame (previously processed segments) or in the previous frame
        used_labels = [segment.label for segment in other_image.segments]
        used_labels += [segment.label for segment in self.parent.segments if segment.label != 0]

        max_overlap = 0
        best_match = None
        for other_segment in other_image.segments:
            overlap = self.get_overlap(other_segment)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = other_segment

        if best_match is None:
            # Fall back to the closest previous segment within max_range (in global coordinates)
            center = self.get_global_points().mean(axis=0)
            min_distance = max_range
            for other_segment in other_image.segments:
                other_center = other_segment.get_global_points().mean(axis=0)
                distance = np.linalg.norm(center - other_center)
                if distance < min_distance:
                    min_distance = distance
                    best_match = other_segment

        if best_match is not None:
            self.label = best_match.label
        else:
            self.label = (max(used_labels) if used_labels else 0) + 1


    def get_bounding_box(self, scale: float = 1.0) -> tuple[int, int, int, int]:
        """Get a square bounding box centered on the segment's center of mass.

        Returns:
            tuple[int, int, int, int]: The bounding box coordinates (min_y, max_y, min_x, max_x).
        """
        y_indices, x_indices = np.where(self.mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return 0, 0, 0, 0
        center_y = np.mean(y_indices)
        center_x = np.mean(x_indices)

        # Half-size is the largest distance from the center needed to contain all points
        half_size = max(
            np.max(np.abs(y_indices - center_y)),
            np.max(np.abs(x_indices - center_x)),
        )

        min_y = int(round(center_y - half_size * scale))
        max_y = int(round(center_y + half_size * scale))
        min_x = int(round(center_x - half_size * scale))
        max_x = int(round(center_x + half_size * scale))

        # Clamp to valid bounds: negative indices would otherwise wrap around
        # instead of clipping when used for array slicing.
        height, width = self.mask.shape[:2]
        min_y = max(0, min_y)
        min_x = max(0, min_x)
        max_y = min(height, max_y)
        max_x = min(width, max_x)
        return min_y, max_y, min_x, max_x