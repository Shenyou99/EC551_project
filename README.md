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

This project represents a significant step forward in the field of handwritten digit recognition and FPGA applications. We hope that it will serve as a valuable tool and a source of inspiration for further innovation in this field.
## Hardware Component

### C Implementation of CNN Model

For the hardware part of our project, we have developed a C version of the CNN model, which is based on our software model. The source code for this C implementation can be found in the `C_version_CNN` directory. This C implementation is crucial for the integration with FPGA.

### Integration with Vivado HLS

The next step involves porting the C version of our CNN model to the Vivado HLS (High-Level Synthesis) tool. In Vivado HLS, we generate an IP (Intellectual Property) core from our C model. This process is key to creating a bridge between our software model and the FPGA hardware.

### Creating Block Design and Generating Bitstream

After generating the IP core, we proceed to create a block design in the Vivado environment. This block design visually represents the interconnections and components of our FPGA implementation. The final step in the hardware implementation is generating a `bitstream` file, which is the binary file that will be loaded onto the FPGA. The `bitstream` file is saved in the `IP` directory.

### Preparing and Inputting Images

To use our calculator, input images should be prepared in a white background JPEG format and placed in the `input_imgs` folder. These images are then processed by our system to recognize the handwritten digits and operators.

### Connecting the FPGA Board

To connect the FPGA board, follow the instructions provided at [PYNQ-Z2 Setup Guide](https://pynq.readthedocs.io/en/latest/getting_started/pynq_z2_setup.html). This guide will help you set up the board and ensure it is ready for use with our system.

### Running the Notebook

Finally, open the Jupyter Notebook and run the `cnn.ipynb` notebook. This notebook will interface with the FPGA board, process the input images, and display the recognition results. The output will show the recognized digits and operators, and perform the necessary calculations based on the handwritten input.

## Conclusion

With both the software and hardware components, our FPGA-Based Handwritten Digit Recognition Calculator represents a cutting-edge approach to digit recognition and computation. We believe this tool will be invaluable for educational and research purposes, and we are excited to see how it can be further developed and utilized in various fields.
