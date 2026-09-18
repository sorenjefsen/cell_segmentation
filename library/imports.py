"""Common imports used across the project.

This module can be imported from a notebook with:
    from imports import *
    # or
    import imports
"""

import numpy as np
import os
import sys
import cv2

sys.path.append(r"C:\Users\sjefs\Desktop\Uni\Speciale\Kode\library")
import CV_functions as CV_func
import CV_classes as CV_class
from PIL import Image
from PIL.TiffTags import TAGS
import matplotlib.pyplot as plt
import random

__all__ = [
    "np",
    "os",
    "sys",
    "cv2",
    "CV_func",
    "CV_class",
    "Image",
    "TAGS",
    "plt",
    "random",
]

