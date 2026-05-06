# IP_project

# Automated Glaucoma Detection Using BPS, LBP, and GLCM

A Hybrid Computer Vision Approach with Novel Enhancements for Improved Accuracy and Robustness in Glaucoma Screening. 

This project was developed as part of **BITSF311: Image Processing** (Group P23).

## 👥 Team Members
* **Veer Reyansh Paka** (2023A7PS0664P)
* **Sree Sai Deepak. Y** (2023A7PS0594P)
* **Daksh Tyagi** (2023A7PS0647P)

## 📖 Overview
Glaucoma is a leading cause of irreversible blindness worldwide. Early detection is critical to slow its progression. This repository contains the Python implementation of an automated glaucoma screening system that extracts robust texture features from retinal fundus images. 

The pipeline builds upon traditional methodologies by combining **Bit-Plane Slicing (BPS)**, **Local Binary Patterns (LBP)**, and **Gray-Level Co-occurrence Matrix (GLCM)** features, enhanced with novel additions like Optic Disc Segmentation, Adaptive BPS, and Ensemble Classification.

## ✨ Key Features & Pipeline
1. **Preprocessing:** Green channel extraction and CLAHE application for non-uniform illumination correction.
2. **Optic Disc Segmentation:** Coarse-to-fine morphological segmentation to isolate the diagnostically relevant optic nerve head.
3. **Adaptive Bit-Plane Slicing (BPS):** Decomposes images into binary planes, automatically filtering out low-order noise planes based on entropy.
4. **Local Binary Pattern (LBP):** Encodes local textural fluctuations on the selected structural bit planes.
5. **GLCM Feature Extraction:** Computes Contrast, Correlation, Energy, and Homogeneity properties.
6. **Dimensionality Reduction:** Min-Max scaling followed by Principal Component Analysis (PCA) retaining 95% variance.
7. **Ensemble Classification:** A Soft-Voting Classifier combining Support Vector Machines (SVM), Random Forests (RF), and Gradient Boosting (GB) for robust prediction.

## 🛠️ Prerequisites
Ensure you have Python 3.8+ installed. You will need the following libraries to run the pipeline:

```bash
pip install opencv-python numpy scikit-image scikit-learn
```
  ## 📥 Dataset Acquisition
 The easiest way to acquire standardized datasets is through Kaggle. First, ensure you have the Kaggle CLI installed:

  ```bash
  pip install kaggle
  (Make sure your kaggle.json API token is placed in ~/.kaggle/)
  ```
  You can then download popular public glaucoma datasets directly to your project folder:
  
  ```bash
  # Example 1: Download the SMDG (Standardized Multi-Channel Dataset for Glaucoma)
  kaggle datasets download -d crarojas/smdg-19
  
  # Example 2: Download a compiled Glaucoma classification dataset
  kaggle datasets download -d rnstch/glaucoma
  ```
  After downloading, unzip the files into your project directory.


Once you have acquired a dataset, extract the images into a data/ directory in the root of your project. Organize the .jpg, .png,       or .tif images into healthy and glaucoma subfolders as shown below:

    project_root/
    │
    ├── glaucoma_detection.py    # Main pipeline script
    ├── README.md                # Project documentation
    │
    └── data/
        ├── healthy/             # Normal/Healthy fundus images
        │   ├── image_001.jpg
        │   └── ...
        │
        └── glaucoma/            # Positive glaucoma fundus images
            ├── image_050.jpg
            └── ...
