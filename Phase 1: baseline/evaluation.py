# ===============================
# Author: Sofia Casadei, Group 2
# Date last modified: 07/06/2021
# Module: CL Team Lab 2021
# Project: Emotion Classification
# ===============================

# Required imports:
import numpy as np
import matplotlib.pyplot as plt


class Evaluation:

    """
    This class evaluates the classifier's predicitons. It uses precision,
    recall and F1-Score. More specifically, it uses macro average,
    meaning that all the classes weigh equally. It creates a confusion matrix
    which is then used to retrieve the values needed for all other calculations.

    Inputs are two lists of labels: the correct ones from the annotated
    dataset and the ones predicted by the classifier.
    """

    def __init__(self, prediction, true):
        self.true = true
        self.prediction = prediction
        self.labels = list(set(true))

        # {'anger': 0, 'fear': 1, 'shame': 2, 'sadness': 3,
        # 'disgust': 4, 'joy': 5, 'guilt': 6}
        self.emotion_int = {}
        for i in range(len(self.labels)):
            self.emotion_int[self.labels[i]] = i

        # Map emotions to integer
        for i in range(len(self.true)):
            self.true[i] = self.emotion_int[self.true[i]]
            self.prediction[i] = self.emotion_int[self.prediction[i]]

        # Indices:
        # self.cm[idx] --> rows
        # self.cm[idx][idx] --> elements in rows
        self.cm = np.zeros((len(self.labels), len(self.labels)))

        # Fill in confusion matrix
        for i in range(len(self.prediction)):
            self.cm[self.prediction[i]][self.true[i]] += 1


    def visualize_cm(self):
        # Visualise the confusion matrix
        plt.imshow(self.cm, cmap=plt.cm.Blues)

        threshold = self.cm.max() / 2
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                plt.text(j, i, int(self.cm[i,j]),
                         color="w" if self.cm[i,j] > threshold else 'black')

        plt.title('Confusion Matrix')
        # define labeling spacing based on number of classes
        tick_marks = np.arange(len(self.labels))
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.colorbar()
        plt.tight_layout()


    def class_precision(self, cm, class_idx):
        # Calculates the precision for one class

        # diagonal
        true_pos = cm[class_idx][class_idx]
        # cm here is the transpose of the original cm -> we need the columns
        precision = true_pos / sum(cm[class_idx])

        return precision


    def macro_avg_precision(self):
        # Calculates macro average precision across all classes

        transposed_cm = np.transpose(self.cm)
        # contains precision score for each class - needed for later?
        p_per_class = []

        for class_idx in range(len(self.cm)):
            precision = self.class_precision(transposed_cm, class_idx)
            p_per_class.append(precision)

        avg_precision = round(sum(p_per_class) / len(p_per_class), 3)

        return avg_precision


    def class_recall(self, class_idx):
        # Calculates recall for one class

        # diagonal
        true_pos = self.cm[class_idx][class_idx]

        # avoid division by zero
        if sum(self.cm[class_idx]) != 0:
            recall =  true_pos / sum(self.cm[class_idx])
        else:
            return 0

        return recall


    def macro_avg_recall(self):
        # Calculates macro average recall across all classes

        r_per_class = []

        for class_idx in range(len(self.cm)):
            recall = self.class_recall(class_idx)
            r_per_class.append(recall)

        avg_recall = round(sum(r_per_class) / len(r_per_class), 3)

        return avg_recall


    def F_score(self):
        # F_score = (2 * Precision * Recall) / (Precision + Recall)

        P = self.macro_avg_precision()
        R = self.macro_avg_recall()
        F_score = round((2 * P * R) / (P + R), 4)

        return F_score

"""
To run this code:

# Assign lists of labels to y_pred and y_true. Here, we evaluate the
predictions of a model that always predicts joy.

y_pred = []
y_true = ['joy', 'disgust', 'disgust', 'anger', 'disgust', 'anger', 'joy', 'fear', 'guilt', 'sadness', 'sadness', 'sadness', 'disgust', 'guilt', 'disgust', 'shame', 'disgust',
          'joy', 'shame', 'guilt', 'fear', 'shame', 'joy', 'shame', 'shame', 'joy', 'disgust', 'joy', 'sadness', 'disgust', 'anger', 'disgust', 'disgust', 'anger', 'disgust',
          'anger', 'anger', 'joy', 'shame', 'sadness', 'guilt', 'guilt', 'sadness', 'anger', 'guilt', 'joy', 'fear', 'anger', 'sadness', 'anger']

for i in range(len(y_true)):
    y_pred.append('joy')

evaluation = Evaluation(y_pred, y_true)

evaluation.F_score()
evaluation.visualize_cm()

Output is the F1-Score (a float) 

"""
