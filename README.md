# Cell Segmentation

Master's thesis project on segmenting and tracking chromosomes/nuclei in fluorescence microscopy timelapse imaging.

## Project structure

```
Kode/
  code/       Jupyter notebooks (main pipeline, exploratory analysis, ML experiments)
  library/    Reusable Python modules (CV_functions.py, CV_classes.py, ...)
  Arkiv/      Older/archived notebooks
Artikler/     Reference articles
Møder/        Meeting notes
```

Raw data, exported images, and processed arrays (`Data/`, `Kode/image_export/`, and image files)
are excluded from version control via `.gitignore` — they are too large / not suited for git.

## Pipeline overview

1. Load a timelapse image (channel 1 = chromosomes, channel 2 = marked region of interest).
2. Preprocess (blur, normalize).
3. Adaptive thresholding to find foreground pixels.
4. Region growing to segment connected components.
5. Identify the largest segment and compute its center of mass, area, and bounding box.
6. Use the channel 1 bounding box to crop the channel 2 image and repeat segmentation.
7. Track segments across frames and collect statistics over time.

See [Kode/code/main_pipeline.ipynb](Kode/code/main_pipeline.ipynb) for the current implementation.

## Setup

```bash
pip install -r requirements.txt
```

(Key dependencies: numpy, opencv-python, matplotlib, Pillow, ipywidgets)
