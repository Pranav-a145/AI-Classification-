# AI-Classification-
Face and Digit Classification using Naive Bayes and Perceptron
Overview
This project focuses on implementing and comparing Naive Bayes and Perceptron classifiers for two supervised learning tasks:

Digit Classification: Recognizing handwritten digits (0-9) from 28×28 text-based images.
Face Classification: Determining whether a 60×70 text-based image contains a face (1) or not (0).
The models were evaluated based on their accuracy under different training data fractions (10% to 100%).

Features
Binary Feature Representation:
Each image is converted into a binary feature vector by mapping # or + to 1 and all other characters to 0.
This transformation ensures a uniform representation for both classifiers.
Training and Evaluation:
Models were trained on varying portions of the dataset (from 10% to 100% of the available data).
Accuracy and standard deviation were recorded over five trials to analyze performance consistency.
Algorithms Implemented
Naive Bayes Classifier
Computes log-prior probabilities for each class.
Uses Laplace smoothing to avoid zero probabilities.
Makes predictions based on maximum likelihood estimation.
Perceptron Classifier
Maintains weight vectors for each class.
Uses an iterative learning process with stochastic updates.
Updates weights whenever a misclassification occurs.
Key Results
Digit Classification:
Naive Bayes achieved 77% accuracy, while Perceptron performed slightly better with 81% accuracy at full training data.
Face Classification:
Naive Bayes outperformed Perceptron, reaching 90.7% accuracy compared to 86.8% for Perceptron.
More training data improved accuracy, and Perceptron exhibited higher variance in small data scenarios.
Lessons Learned
Naive Bayes performs well when features are conditionally independent, as seen in the face classification task.
Perceptron benefits from larger datasets, surpassing Naive Bayes in multi-class digit recognition.
Binary feature extraction is a simple yet effective preprocessing step for classification tasks.
Randomized trials are necessary to capture model variance, especially with small datasets.
Running the Project
To execute the project, ensure the dataset is available in the correct directory structure and run:
python faceDigitClassification.py


Future Improvements
Experiment with feature engineering (e.g., edge detection, pixel intensity histograms).
Optimize Perceptron hyperparameters (learning rate, number of epochs).
Explore deep learning models (CNNs) to enhance classification performance.
