# ==========================================
# Author: Sofia Casadei, Group 2
# Date last modified: 30/07/2021
# Module: CL Team Lab 2021
# Project: Emotion Classification with irony
# ==========================================

# File summary:
# Emotion classification on TweetEval data
# REMOVE emojis

# The following packages need to be installed to be able to run this code:
# ktrain
# contractions
# emojis
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

import emojis
from nltk.tokenize import TweetTokenizer


"""Data part"""

# Load train data
# !!! assumes the data file is in a folder 'data' located in the same directory
# as this file
train_path = '/data/tweeteval-train_text.txt'

with open(train_path) as f:
    x_train = f.read().splitlines()


with open('/data/tweeteval-train_labels.txt') as f:
    y_train = f.read().splitlines()


# Convert numbers to emotion labels
y_train_emo = []

for label in y_train:
    if label == '0':
        y_train_emo.append('anger')
    elif label == '1':
        y_train_emo.append('joy')
    elif label == '2':
        y_train_emo.append('optimism')
    elif label == '3':
        y_train_emo.append('sadness')


# Check distribution
sns.displot(y_train_emo, shrink=.8)


# Create new dataset (balanced)
new_x_train = []
new_y_train = []

idx = 0
for label in y_train_emo:
    if new_y_train.count(label) < 320:
        new_y_train.append(label)
        new_x_train.append(x_train[idx])
    idx += 1


# Distribution of modified dataset
sns.displot(new_y_train, shrink=.8)



# Load test data
# !!! assumes the data file is in a folder 'data' located in the same directory
# as this file
test_path = '/data/tweeteval-test_text.txt'

with open(test_path) as f:
    x_test = f.read().splitlines()


with open('/data/tweeteval-test_labels.txt') as f:
    y_test = f.read().splitlines()


# Convert numbers to emotion labels
y_test_emo = []

for label in y_test:
    if label == '0':
        y_test_emo.append('anger')
    elif label == '1':
        y_test_emo.append('joy')
    elif label == '2':
        y_test_emo.append('optimism')
    elif label == '3':
        y_test_emo.append('sadness')


# Check distribution
sns.displot(y_test_emo, shrink=.8)

# Create new dataset (balanced)
new_x_test = []
new_y_test = []

idx = 0
for label in y_test_emo:
    if new_y_test.count(label) < 120:
        new_y_test.append(label)
        new_x_test.append(x_test[idx])
    idx += 1

# Distribution of modified dataset
sns.displot(new_y_test, shrink=.8)

"""Normalisation of input

Normalise:
+ hashtags
+ tagged users
+ emoji (REMOVE ALL - from other experiments we know that they do not help classification)
+ urls

"""


def normalise_tweet(tweet):
    norm_tweet = re.sub("&", "and", tweet)
    norm_tweet = re.sub(r"[<>]", "", norm_tweet)
    norm_tweet = re.sub("http:.*", "url", norm_tweet)
    norm_tweet = re.sub("@", " @", norm_tweet)
    norm_tweet = re.sub("#", " ", norm_tweet)

    norm_tweet = re.sub(r"[-'()/_;:{}=~|,\[\]]", " ", norm_tweet)
    norm_tweet = re.sub(r"\\n", "", norm_tweet)

    norm_tweet = contractions.fix(norm_tweet)

    tokenizer = TweetTokenizer()
    tweet_tokens = tokenizer.tokenize(norm_tweet)
    final_tweet_list = []

    for token in tweet_tokens:
        if token.startswith("@"):
            # then token is a user tag
            tag_token = "tagged_user"
            if final_tweet_list.count(tag_token) < 3:
                final_tweet_list.append(tag_token)
        elif emojis.count(token) == 1:
            # then token is an emoji
            emoji_token = '' # remove all emojis
            #if final_tweet_list.count(emoji_token) < 2:
                #final_tweet_list.append(emoji_token)
        else:
            final_tweet_list.append(token)

    final_tweet = ' '.join(final_tweet_list)

    return final_tweet.strip()


# Create variables for normalised datasets

x_train_norm = []
for tweet in new_x_train:
    x_train_norm.append(normalise_tweet(tweet))

x_test_norm = []
for tweet in new_x_test:
    x_test_norm.append(normalise_tweet(tweet))


"""Model Set up - roberta-base"""

categories = ['anger', 'joy', 'optimism', 'sadness']

MODEL_NAME = 'roberta-base'

# Transormer is a wrapper to the Hugging Face transformers library for text classification.
t = text.Transformer(MODEL_NAME, maxlen=80, class_names=categories)

# input data
trn = t.preprocess_train(x_train_norm, new_y_train)
val = t.preprocess_test(x_test_norm, new_y_test)

model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)

"""Train"""

best_lr = 5e-5

# Learning policies:
# triangular LR -> learner.autofit(0.0007, 8) (implicit ReduceLROnPlateau and EarlyStopping)
# 1cycle -> learner.fit_onecycle(best_lr, 5)


learner.autofit(lr=best_lr, verbose=1)

"""Evaluate/Inspect model"""

learner.validate(class_names=t.get_classes())


# the ones where the loss was highest (the ones that we got very wrong)
learner.view_top_losses(n=5, preproc=t)


"""Make predictions on new data"""

predictor = ktrain.get_predictor(learner.model, preproc=t)

test_sent = ('Even though it is raining, it is a nice day and I do not feel sad')

predictor.predict(test_sent)

# Ask for explanation
#predictor.explain(test_sent)

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


"""Save"""

predictor.save('/my_models/EC_RoBERTa_TweetEval_noemojis')


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
