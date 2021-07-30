# ==========================================
# Author: Sofia Casadei, Group 2
# Date last modified: 30/07/2021
# Module: CL Team Lab 2021
# Project: Emotion Classification with irony
# ==========================================

# File summary:
# EFFECT OF PRE-PROCESSING OF DATA
# ktrain
# hugging face transformers
# roberta-base
# LR policy: autofit (triangle)

# The following packages need to be installed to be able to run this code:
# ktrain
# contractions
# eli5
# nltk punkt
# tensorflow
# tensorflow_hub

# Required imports:

import nltk

import numpy as np
import pandas as pd

from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import re
import contractions

import ktrain
from ktrain import text


"""Data part"""

# Load train data
# !!! assumes the data file is in a folder 'data' located in the same directory
# as this file
train_path = '/data/corrected_isear-train.csv'

df_train = pd.read_csv(train_path, names=['emotion', 'text', 'NaN'])

classes = df_train.emotion.unique()

# dataset is balanced

# visualize data
sns.displot(df_train, x="emotion", shrink=.8)


# Load validation data
# !!! assumes the data file is in a folder 'data' located in the same directory
# as this file
val_path = '/data/corrected_isear-val.csv'

df_val = pd.read_csv(val_path, names=['emotion', 'text', 'NaN'])


# Load test data
# !!! assumes the data file is in a folder 'data' located in the same directory
# as this file
test_path = '/data/corrected_isear-test.csv'

df_test = pd.read_csv(test_path, names=['emotion', 'text', 'NaN'])


x_train = df_train['text'].to_numpy()
y_train = df_train['emotion'].to_numpy()

x_val = df_val['text'].to_numpy()
y_val = df_val['emotion'].to_numpy()

x_test = df_test['text'].to_numpy()
y_test = df_test['emotion'].to_numpy()



"""Normalise input"""


def normalise_text(text):

    # Expand contractions
    norm_text = contractions.fix(text)

    # Remove stop words
    stop_words = ['and', 'the', 'to', 'a', 'an', 'of', 'or']
    nostops = []
    for word in nltk.word_tokenize(norm_text):
        if word.lower() not in stop_words:
            # lowercase
            nostops.append(word)
    norm_text = ' '.join(nostops)

    # Remove punctuation
    norm_text = re.sub(r"[-()#/@;:<>{}=~\.\?\"\[\]]+", "", norm_text)


    return norm_text

x_train_norm = []
for text in x_train:
    x_train_norm.append(normalise_text(text))

x_val_norm = []
for text in x_val:
    x_val_norm.append(normalise_text(text))

x_test_norm = []
for text in x_test:
    x_test_norm.append(normalise_text(text))


x_train_norm = np.array(x_train_norm)
x_val_norm = np.array(x_val_norm)
x_test_norm = np.array(x_test_norm)

# get length of all the texts in the train set
seq_len = [len(i.split()) for i in x_train_norm]

#visualize lengths
pd.Series(seq_len).hist(bins = 30)



"""Model Set up"""

categories = ['joy', 'fear', 'shame', 'disgust', 'guilt', 'anger', 'sadness']

MODEL_NAME = 'roberta-base'

# Transormer is a wrapper to the Hugging Face transformers library for text classification.
t = text.Transformer(MODEL_NAME, maxlen=150, class_names=categories)

# input data
trn = t.preprocess_train(x_train_norm, y_train)
val = t.preprocess_test(x_val_norm, y_val)

model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)


"""Train"""

best_lr = 5e-5

# Learning policies:
# triangular LR -> learner.autofit(0.0007, 8) (implicit ReduceLROnPlateau and EarlyStopping)
# 1cycle -> learner.fit_onecycle(best_lr, 5)

# saves model checkpoints to 'my_models' folder
learner.autofit(lr=best_lr, checkpoint_folder='/my_models', verbose=1)


"""Evaluate/Inspect model"""

learner.validate(class_names=t.get_classes())

# Set weights to those of the best epoch
#learner.model.load_weights('/my_models/weights-07.hdf5')

# visualize top losses
learner.view_top_losses(n=10, preproc=t)


"""Make predictions on new data"""

predictor = ktrain.get_predictor(learner.model, preproc=t)

test_sent = ('Even though it is raining, it is a nice day and I do not feel sad')

predictor.predict(test_sent)


more_sents = ['Playing with a very cute doggo',
              'Feeling lonely',
              'I rode at the back of a scooter driven by a stranger on the motorway in Vietnam, not knowing where we were going',
              'I told my mum a lie and it caused her being very disappointed afterwards',
              'I shat myself at school',
              'when I see dirty houses full of useless crap',
              'seeing racist and homophobic comments infuriates me']

predictor.predict(more_sents)

# Ask for explanation
#predictor.explain(more_sents[0])

# probability of prediction
#predictor.predict_proba(more_sents[0])


"""Save"""

predictor.save('/my_models/EC_RoBERTa_norm')


"""
To load and continue training
```
# save model and Preprocessor instance after partially training
ktrain.get_predictor(model, preproc).save('/tmp/my_predictor')

# reload Predictor and extract model
model = ktrain.load_predictor('/tmp/my_predictor').model

# re-instantiate Learner and continue training
learner = ktrain.get_learner(model, train_data=trn, val_data=val)
learner.fit_onecycle(2e-5, 1)
```

"""
