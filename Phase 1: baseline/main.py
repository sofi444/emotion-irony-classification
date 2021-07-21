# ===============================
# Author: Sofia Casadei, Group 2
# Date last modified: 07/06/2021
# Module: CL Team Lab 2021
# Project: Emotion Classification
# ===============================

# Required imports:
import pandas as pd
import re
import math
from collections import defaultdict

from data_cleaning import DataCleaner
from vectoriser import Vectoriser
from perceptron import MultiClassPerceptron
from evaluation import Evaluation

"""
This code import all the relevant classes. It trains the classifier on the
train data and makes predictions on the test data. Finally, it prints the
F1-Score of the classification and displays the confusion matrix, which shows
the distribution of the predictions.
"""

# Assumes a folder 'data' in the same directory as main.py
isear_train = 'data/isear/isear-train.csv'
isear_test = 'data/isear/isear-test.csv'

def clean_data(isear_data):
    # Can take an input any of the isear datasets

    cleaner = DataCleaner(isear_data)
    cleaner.openCSV_organize()
    clean_data = cleaner.clean_text(keep_allcap=False)

    return clean_data

def extract_labels(dataset):
    # Takes as input a clean dataset
    # Extracts the emotion labels and returns a list

    y = []
    for sample in dataset:
        y.append(sample[1])

    return y


# Clean datasets
clean_train_data = clean_data(isear_train)
clean_test_data = clean_data(isear_test)
# Format:
# [[0, 'joy', 'when i understood that i was admitted university'],
#  [1, 'fear', "i broke window of neighbouring house i feared my mother's
#   judgement action on what i had done"], ...]

# Extract labels
y_train = extract_labels(clean_train_data)
y_test = extract_labels(clean_test_data)


# Vectorise training and test data
vectoriser = Vectoriser(clean_train_data, clean_test_data)
x_train = vectoriser.TFIDF_training()
x_test = vectoriser.TFIDF_testing()
# Format:
# [{1315: 0.10007165383810532,
#   1365: 0.08767922491907673,
#   2454: 0.7341554894650688,
#    999: 0.18030338960965467,
#   3552: 0.11758628630364083,
#   3674: 0.6561362008309446,
#    772: 0.47954524855743885},
#  {1365: 0.06189121758993652,
#   7018: 0.2659324180162554, ...}, ...]
# --> index corresponding to word: TF-IDF value

# Training and making predictions
perceptron = MultiClassPerceptron(x_train, y_train, x_test, y_test, epochs=10)
perceptron.train()
predictions = perceptron.predict()


# Evaluation
evaluation = Evaluation(predictions, y_test)

print("F-Score:", evaluation.F_score())

# Visualise confusion matrix
evaluation.visualize_cm()
