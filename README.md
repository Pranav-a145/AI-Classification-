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
- Maintains weight vectors for each class.
- Uses an iterative learning process with **stochastic updates**.
- Updates weights when a misclassification occurs.

## Key Results
- **Digit Classification**:
  - Naive Bayes achieved **77% accuracy**, while Perceptron performed slightly better with **81% accuracy** at full training data.
- **Face Classification**:
  - Naive Bayes outperformed Perceptron, reaching **90.7% accuracy** compared to **86.8%** for Perceptron.
- **More training data improved accuracy**, and Perceptron exhibited higher variance in small data scenarios.

## Lessons Learned
- **Naive Bayes performs well when features are conditionally independent**, as seen in face classification.
- **Perceptron benefits from larger datasets**, surpassing Naive Bayes in multi-class digit recognition.
- **Binary feature extraction** is a simple yet effective preprocessing step.
- **Randomized trials capture model variance**, especially with small datasets.

## Running the Project
Ensure the dataset is in the correct directory structure and run:

```bash
python faceDigitClassification.py
