# Assignment 2: CIFAR-10 Image Classification using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset into 10 distinct classes.

## Files
* `02_cifar10_cnn.py`: Contains the model architecture, training loop, and evaluation metrics.

## Requirements
Ensure you have TensorFlow installed:
```bash
pip install tensorflow
```

## How to Run
Execute the script from the terminal:
```bash
python 02_cifar10_cnn.py
```
The script will automatically download the CIFAR-10 dataset (if not already cached), train the CNN for 10 epochs, and print the final test accuracy and loss.