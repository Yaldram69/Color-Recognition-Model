# Color Recognition Model

A computer vision project for **color recognition** using a CNN trained on a synthetic color image dataset, with **real-time webcam-based color detection and labeling** using OpenCV.

---

## Project Overview

This project has three main parts:

1. **Dataset Generation**
   - Generates synthetic images for predefined color classes
   - Adds random shade variation and noise for diversity

2. **Model Training**
   - Trains a CNN using TensorFlow/Keras on the generated dataset
   - Saves:
     - `color_classification_model.h5`
     - `class_indices.json`

3. **Real-Time Color Detection**
   - Captures webcam frames using OpenCV
   - Segments candidate color regions using HSV ranges
   - Classifies each detected region using the trained CNN
   - Displays bounding boxes and predicted color labels

---

## Features

- Synthetic dataset generation for color classes
- Color variation using shade scaling and noise
- CNN-based color classification
- Real-time webcam color detection and labeling
- HSV-based segmentation + model-based recognition
- Saved model + class index mapping for inference

---

## Tech Stack

- Python
- NumPy
- OpenCV
- Pillow (PIL)
- TensorFlow / Keras

---
