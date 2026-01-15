# ThinVesselSeg

This is a repository for Antanas Bukauskas Master's Thesis "Impact of Downscaling Fundus Images and Vessel Masks on Thin Retinal Vessel Segmentation". Here I provide code used for training segmentation models on multiple resolutions (under `HRFormer/seg`) as well a code for separating thin and thick vessels from ground truth masks (`thin_vessel_separator.py`). There are also implementation file for measuring ground truth downscaling structural information preservation of thin and thick vessels (`label_resample_stats.py`). Other simpler and smaller scripts used in this study were not provided.

Dataset used for the experiments can be downloaded here: https://figshare.com/ndownloader/files/34969398

This repository depends on a fork of HRFormer segmentation model repository, so please clone using `--recurse-submodules` flag.

```
git clone --recurse-submodules https://github.com/AntanasBuk/ThinVesselSeg.git
```

