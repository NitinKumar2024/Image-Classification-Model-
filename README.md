# Image Classification Model using CIFAR-10

![Image Classification](https://www.simplilearn.com/ice9/free_resources_article_thumb/KerasImageClassificationModels/ImageClassification_Models_1.png)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This repository contains the code for an Image Classification Model using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to classify images into one of the 10 categories.

## Features

- **Data Preprocessing**: Normalizes the image data.
- **Model Training**: Implements a convolutional neural network (CNN) for image classification.
- **Model Evaluation**: Evaluates the model's performance using metrics like accuracy and loss.
- **Prediction**: Predicts the class of a given image.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 testing images. The classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/NitinKumar2024/Image-Classification-Model-
   
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

1. **Prepare the Dataset**:
    The CIFAR-10 dataset is available in the `tensorflow` and `keras` libraries. You don't need to download it separately.

2. **Train the Model**:
    ```bash
    python train.py
    ```

    ```


## Model Architecture

The project utilizes a Convolutional Neural Network (CNN) to classify images. The architecture includes:

- Convolutional Layers
- Max Pooling Layers
- Fully Connected Layers
- Dropout Layers
