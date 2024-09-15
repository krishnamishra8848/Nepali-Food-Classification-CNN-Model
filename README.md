# Nepali Food Image Classification Model

## Overview

This repository contains a pre-trained image classification model using MobileNetV2, designed to classify images of Nepali food dishes. The model has been trained to recognize the following food classes:

- **Chatamari**
- **Chhoila**
- **Dalbhat**
- **Dhindo**
- **Gundruk**
- **Kheer**
- **Momo**
- **Sekuwa**
- **Selroti**

## Model Details

- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Input Shape:** 150x150 RGB images
- **Number of Classes:** 9
- **Regularization:** L2 regularization with a factor of 0.01
- **Dropout Rate:** 50% in the dense layer
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Callbacks:**
  - **ReduceLROnPlateau:** Adjusts learning rate based on validation loss
  - **EarlyStopping:** Stops training when validation loss plateaus

## Data

The model was trained using images from the Nepali food dataset. The dataset is organized into subdirectories for each food class.



