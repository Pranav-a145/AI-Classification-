# Face and Digit Classification using Naive Bayes and Perceptron

## Overview
This project implements and compares **Naive Bayes** and **Perceptron** classifiers for two supervised learning tasks:

1. **Digit Classification**: Recognizing handwritten digits (0-9) from 28×28 text-based images.
2. **Face Classification**: Identifying whether a 60×70 text-based image contains a face (1) or not (0).

The models were evaluated based on accuracy across different training data fractions (10% to 100%).

## Features
- **Binary Feature Representation**:
  - Each image is converted into a binary feature vector by mapping `#` or `+` to `1`, and all other characters to `0`.
  - Ensures a uniform representation for both classifiers.

- **Training and Evaluation**:
  - Models were trained on varying portions of the dataset (10% to 100% of available data).
  - Accuracy and standard deviation were recorded over **five trials** to analyze performance consistency.

## Algorithms Implemented
### **Naive Bayes Classifier**
- Computes **log-prior** probabilities for each class.
- Uses **Laplace smoothing** to avoid zero probabilities.
- Predicts using **maximum likelihood estimation**.

### **Perceptron Classifier**
- Maintains weight vectors for each c
