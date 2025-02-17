
import math
import random
import statistics
import os


DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28

FACE_WIDTH = 60
FACE_HEIGHT = 70

NUM_TRIALS = 5

EPOCHS = 5

LAPLACE_K = 1.0

# ----------------------------------------------------------------------
#                         UTILITY FUNCTIONS
# ----------------------------------------------------------------------
def load_data_images_labels(image_path, label_path, width, height):
   
    with open(image_path, 'r') as f:
        image_lines = [line.rstrip('\n') for line in f]

    with open(label_path, 'r') as f:
        label_lines = [line.strip() for line in f]

    data = []
    i = 0
    for _ in range(len(label_lines)):
        img = image_lines[i : i + height]
        i += height
        features = []
        for row in img:
            for ch in row:
                if ch == '#' or ch == '+':
                    features.append(1)
                else:
                    features.append(0)
        data.append(features)

    labels = [int(lbl) for lbl in label_lines]

    return data, labels


def split_random_subset(data, labels, subset_fraction):
    
    combined = list(zip(data, labels))
    random.shuffle(combined)
    cutoff = int(len(combined) * subset_fraction)
    subset = combined[:cutoff]
    rest = combined[cutoff:]
    subset_data, subset_labels = zip(*subset)
    remaining_data, remaining_labels = zip(*rest) if rest else ([], [])
    return list(subset_data), list(subset_labels), list(remaining_data), list(remaining_labels)


def accuracy_score(true_labels, pred_labels):
  
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    return correct / len(true_labels) if true_labels else 0.0


# ----------------------------------------------------------------------
#                         NAIVE BAYES CLASSIFIER
# ----------------------------------------------------------------------
class NaiveBayesClassifier:
    def __init__(self, num_classes, num_features):
        self.num_classes = num_classes
        self.num_features = num_features
        self.log_prior = [0.0]*num_classes
        self.log_prob_feature_on = [[0.0]*num_features for _ in range(num_classes)]
        self.log_prob_feature_off = [[0.0]*num_features for _ in range(num_classes)]

    def train(self, train_data, train_labels):
       
        class_counts = [0]*self.num_classes
        feature_counts_on = [[0]*self.num_features for _ in range(self.num_classes)]
        
        for x, y in zip(train_data, train_labels):
            class_counts[y] += 1
            for i, val in enumerate(x):
                if val == 1:
                    feature_counts_on[y][i] += 1

        total_samples = len(train_data)
        for c in range(self.num_classes):
            if class_counts[c] > 0:
                self.log_prior[c] = math.log(class_counts[c] / total_samples)
            else:
                self.log_prior[c] = float('-inf')  

        for c in range(self.num_classes):
            for i in range(self.num_features):
                numerator_on = feature_counts_on[c][i] + LAPLACE_K
                denominator = class_counts[c] + 2.0 * LAPLACE_K
                prob_on = numerator_on / denominator
                prob_off = 1.0 - prob_on
                self.log_prob_feature_on[c][i] = math.log(prob_on)
                self.log_prob_feature_off[c][i] = math.log(prob_off)

    def predict(self, x):
       
        scores = []
        for c in range(self.num_classes):
            score = self.log_prior[c]
            for i, val in enumerate(x):
                if val == 1:
                    score += self.log_prob_feature_on[c][i]
                else:
                    score += self.log_prob_feature_off[c][i]
            scores.append(score)
        return max(range(self.num_classes), key=lambda idx: scores[idx])

    def classify(self, test_data):
     
        return [self.predict(x) for x in test_data]


# ----------------------------------------------------------------------
#                         PERCEPTRON CLASSIFIER
# ----------------------------------------------------------------------
class PerceptronClassifier:
    def __init__(self, num_classes, num_features):
        self.num_classes = num_classes
        self.num_features = num_features
        self.weights = [[0.0]*num_features for _ in range(num_classes)]

    def train(self, train_data, train_labels, epochs=3):
      
        for _ in range(epochs):
            combined = list(zip(train_data, train_labels))
            random.shuffle(combined)
            for x, y in combined:
                scores = []
                for c in range(self.num_classes):
                    dot = 0.0
                    for i, val in enumerate(x):
                        if val != 0:
                            dot += self.weights[c][i] * val
                    scores.append(dot)
                y_hat = max(range(self.num_classes), key=lambda c: scores[c])
                if y_hat != y:
                    for i, val in enumerate(x):
                        if val != 0:
                            self.weights[y][i]     += val
                            self.weights[y_hat][i] -= val

    def predict(self, x):
        
        scores = []
        for c in range(self.num_classes):
            dot = 0.0
            for i, val in enumerate(x):
                if val != 0:
                    dot += self.weights[c][i] * val
            scores.append(dot)
        return max(range(self.num_classes), key=lambda c: scores[c])

    def classify(self, test_data):
        return [self.predict(x) for x in test_data]


# ----------------------------------------------------------------------
#                            EXPERIMENT LOGIC
# ----------------------------------------------------------------------
def experiment_and_print_results(
    classifier_name, 
    classifier_constructor, 
    train_data, train_labels,
    test_data, test_labels,
    num_classes,
    train_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
):

    print(f"\n====== {classifier_name} ======")
    print("Train Fraction | Mean Accuracy | Std Accuracy")
    print("---------------+---------------+-------------")
    num_features = len(train_data[0]) if train_data else 0

    for frac in train_fractions:
        accuracies = []
        for _ in range(NUM_TRIALS):
            subset_data, subset_labels, _, _ = split_random_subset(train_data, train_labels, frac)
            clf = classifier_constructor(num_classes, num_features)
            if isinstance(clf, PerceptronClassifier):
                clf.train(subset_data, subset_labels, epochs=EPOCHS)
            else:
                clf.train(subset_data, subset_labels)
            preds = clf.classify(test_data)
            acc = accuracy_score(test_labels, preds)
            accuracies.append(acc)
        mean_acc = statistics.mean(accuracies)
        std_acc = statistics.pstdev(accuracies)
        print(f"{int(frac*100):3d}%          | {mean_acc:.4f}       | {std_acc:.4f}")


def main():
    random.seed(42)  

    # ------------------------------------------------------------------
    # Load digit data
    # ------------------------------------------------------------------
    digit_train_data, digit_train_labels = load_data_images_labels(
        os.path.join("digitdata","trainingimages"),
        os.path.join("digitdata","traininglabels"),
        DIGIT_WIDTH, DIGIT_HEIGHT
    )
    digit_test_data, digit_test_labels = load_data_images_labels(
        os.path.join("digitdata","testimages"),
        os.path.join("digitdata","testlabels"),
        DIGIT_WIDTH, DIGIT_HEIGHT
    )

    # ------------------------------------------------------------------
    # Load face data
    # ------------------------------------------------------------------
    face_train_data, face_train_labels = load_data_images_labels(
        os.path.join("facedata","facedatatrain"),
        os.path.join("facedata","facedatatrainlabels"),
        FACE_WIDTH, FACE_HEIGHT
    )
    face_test_data, face_test_labels = load_data_images_labels(
        os.path.join("facedata","facedatatest"),
        os.path.join("facedata","facedatatestlabels"),
        FACE_WIDTH, FACE_HEIGHT
    )

    # ------------------------------------------------------------------
    # Digits => 10 classes
    # ------------------------------------------------------------------
    experiment_and_print_results(
        classifier_name = "Naive Bayes (Digits)",
        classifier_constructor = NaiveBayesClassifier,
        train_data = digit_train_data,
        train_labels = digit_train_labels,
        test_data = digit_test_data,
        test_labels = digit_test_labels,
        num_classes = 10
    )

    experiment_and_print_results(
        classifier_name = "Perceptron (Digits)",
        classifier_constructor = PerceptronClassifier,
        train_data = digit_train_data,
        train_labels = digit_train_labels,
        test_data = digit_test_data,
        test_labels = digit_test_labels,
        num_classes = 10
    )

    # ------------------------------------------------------------------
    # Faces => 2 classes
    # ------------------------------------------------------------------
    experiment_and_print_results(
        classifier_name = "Naive Bayes (Faces)",
        classifier_constructor = NaiveBayesClassifier,
        train_data = face_train_data,
        train_labels = face_train_labels,
        test_data = face_test_data,
        test_labels = face_test_labels,
        num_classes = 2
    )

    experiment_and_print_results(
        classifier_name = "Perceptron (Faces)",
        classifier_constructor = PerceptronClassifier,
        train_data = face_train_data,
        train_labels = face_train_labels,
        test_data = face_test_data,
        test_labels = face_test_labels,
        num_classes = 2
    )


if __name__ == "__main__":
    main()
