import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

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
        image_name (str): File name without directory information.
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
        self.image_name = path.split("\\")[-1]
        self.load_image(path)
        self.array_original = self.array.copy()

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
        self.array = np.array(Image.open(self.path))

        # Find timestamp
        start_pos = self.image_name.find("_t")
        end_pos = self.image_name.find("_", start_pos + 2)
        self.t = int(self.image_name[start_pos + 2:end_pos])

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

    def display(self, color_map: str = 'gray', original: bool = False) -> None:
        """Display the current image or the original image.

        Args:
            color_map (str, optional): Matplotlib colormap name to use for display.
                Defaults to ``'gray'``.
            original (bool, optional): If ``True``, display the unprocessed image
                stored in ``array_original``. Defaults to ``False``.

        Returns:
            None: Opens a matplotlib figure window.
        """
        if original:
            plt.imshow(self.array_original, cmap=color_map)
        else:
            plt.imshow(self.array, cmap=color_map)
        plt.title(f"{self.image_name} + Time = {self.t}")
        plt.show()

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


###################### Image CV processing ##########################
    def adaptive_threshold(self, block_size=101, C=-20) -> None:
        """Apply adaptive thresholding to the image.

        Args:
            block_size (int, optional): Size of the neighborhood used to calculate the threshold.
                Must be an odd number. Defaults to ``101``.
            C (int, optional): Constant subtracted from the mean or weighted mean.
                Defaults to ``-20``.

        Returns:
            threshold_array (np.ndarray): The binary image resulting from adaptive thresholding.
        """
        # Create a copy of the array
        temp_array = self.array.copy()

        if temp_array.ndim == 3:
            temp_array = cv2.cvtColor(temp_array, cv2.COLOR_BGR2GRAY)
        if temp_array.dtype != np.uint8:
            temp_array = cv2.normalize(
                temp_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

        threshold_array = cv2.adaptiveThreshold(
            temp_array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C
        )
        return threshold_array