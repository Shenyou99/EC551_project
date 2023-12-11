# FPGA-Based Handwritten Digit Recognition Calculator

## Introduction

Welcome to our FPGA-Based Handwritten Digit Recognition Calculator project! This innovative calculator is designed to recognize and compute handwritten digits and mathematical operations. The project is divided into two main components: the software for image processing and digit recognition, and the hardware implementation using FPGA (Field-Programmable Gate Array).

## Software Component

### Image Processing

The first step in our software pipeline involves preparing the input images for digit recognition. We begin by resizing the input photographs to a uniform size of 400x300 pixels. This resizing is crucial for standardizing the input for our recognition algorithm. Once resized, the images are converted to black and white to simplify the recognition process.

### Image Segmentation

After preprocessing, we perform image segmentation. This crucial step involves dividing the black and white image into segments. Each segment is expected to contain either a single digit or a mathematical operator. This segmentation is essential for the subsequent recognition process, ensuring that each character is individually analyzed.

### Convolutional Neural Network (CNN) Model

We utilize TensorFlow to construct a Convolutional Neural Network (CNN) model. This model is trained to recognize the segmented images of digits and operators. The training process involves feeding a large dataset of handwritten digits and mathematical symbols into the CNN. After the training, the model is capable of performing convolution operations on the segmented images to predict the digits and operators.

## FPGA Implementation

The final component of our project involves the implementation of the trained CNN model on an FPGA. This hardware integration allows for real-time processing and calculation, making our calculator not only innovative but also efficient and practical for everyday use.

## Conclusion

With both the software and hardware components, our FPGA-Based Handwritten Digit Recognition Calculator represents a cutting-edge approach to digit recognition and computation. We believe this tool will be invaluable for educational and research purposes, and we are excited to see how it can be further developed and utilized in various fields.
