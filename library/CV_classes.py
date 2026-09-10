import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Picture():
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    def __init__(self, path):
        self.path = path
        self.image_name = path.split("\\")[-1]
        self.load_image(path)


    def load_image(self, path):
        self.array = np.array(Image.open(self.path))

        # Find timestamp
        start_pos = self.image_name.find("_t")
        end_pos = self.image_name.find("_", start_pos + 2)
        self.t = int(self.image_name[start_pos + 2:end_pos])

    def display(self):
        plt.imshow(self.array)
        plt.title("Time = " + f"{self.t}")
        plt.show()

    def get_average_signal(self):
        return np.mean(self.array)

    def blur(self, ksize=(5, 5), sigma=1.4):
        if ksize[0] % 2 == 0 or ksize[1] % 2 == 0:
            raise ValueError("Kernel size must be odd in both dimensions")
        from cv2 import GaussianBlur
        self.array = GaussianBlur(self.array, ksize, sigma)
