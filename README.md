# Chest X-Ray Image Classification using Densenet

## Introduction

This project involves the classification of chest X-ray images to identify different medical conditions. Using a dataset from Kaggle, we apply deep learning techniques leveraging the DenseNet architecture to perform image classification tasks efficiently. DenseNet, a convolutional neural network model, is reputed for its robustness and high performance in image recognition tasks, as introduced in the research paper titled "Densely Connected Convolutional Networks."

## Dataset

The dataset for this project is sourced from Kaggle, which provides a comprehensive collection of chest X-ray images and their corresponding diagnoses. You can access and download the dataset from [Kaggle's NIH Chest X-rays Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data). It contains labeled images that are utilized to train and evaluate the performance of the image classification model.

## Core Concepts

### Data Preprocessing

Initially, the dataset is divided into training, validation, and test sets to facilitate model evaluation. The images undergo preprocessing steps such as resizing, normalization, and augmentation to improve the model's generalization capabilities.

### Model Architecture

A DenseNet model is employed for this image classification task. DenseNet is designed with densely connected layers, wherein each layer receives input from all preceding layers, thereby enhancing feature propagation and reducing the number of parameters. The model is initialized with pre-trained weights, and the final layer is customized to fit the number of classes in this dataset.

### Training

The model is trained using the PyTorch framework, which supports seamless integration and testing on GPU. We utilize the Adam optimizer and cross-entropy loss function for this multi-class classification problem. The model's performance is periodically validated using the validation set to prevent overfitting.

### Inference

Once trained, the model can predict the medical condition of a patient based on a provided chest X-ray image. An image undergoes the same preprocessing transformations as during training before classification.

## Note :

The dataset is quite large (~42GB) so it is advised to upload the notebook in kaggle and try it out from there

## References

- [Kaggle NIH Chest X-rays Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [DenseNet Paper: Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

In conclusion, this project showcases how advanced deep learning models, specifically DenseNet, can be effectively utilized for medical image classification tasks, aiding in automated diagnosis solutions.
