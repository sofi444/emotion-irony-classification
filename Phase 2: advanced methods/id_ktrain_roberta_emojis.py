# ==========================================
# Author: Sofia Casadei, Group 2
# Date last modified: 30/07/2021
# Module: CL Team Lab 2021
# Project: Emotion Classification with irony
# ==========================================

# File summary:
# Irony detection
# SemEval 2018 data
# keep emojis

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
train_path = '/data/semeval_taskA_corrected.csv'

df_train = pd.read_csv(train_path, header=0, names=['index',
                                                    'irony_label',
                                                    'tweet'])



# Classes are 1 and 0. Tweet can either be ironic or non-ironic -> binary classification
classes = df_train.irony_label.unique()


# => dataset is Balanced

# Load test data
test_path = '/data/semeval_taskA_test.csv'

df_test = pd.read_csv(test_path, sep='\t', header=0, names=['index',
                                                            'irony_label',
                                                            'tweet'])



x_train = df_train['tweet'].to_numpy()
y_train = df_train['irony_label'].to_numpy()

x_test = df_test['tweet'].to_numpy()
y_test = df_test['irony_label'].to_numpy()

"""Normalisation of input

Normalise:
+ hashtags
+ tagged users
+ emoji (keep)
+ urls
"""


def normalise_tweet(tweet):
    norm_tweet = re.sub("&", "and", tweet)
    norm_tweet = re.sub(r"[<>]", "", norm_tweet)
    norm_tweet = re.sub("http:.*", "url", norm_tweet)
    norm_tweet = re.sub("@", " @", norm_tweet)
    norm_tweet = re.sub("#", " ", norm_tweet)

    norm_tweet = re.sub(r"[-()/_;:{}=~|,\[\]]", " ", norm_tweet)

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
            emoji_token = emojis.decode(token)
            if final_tweet_list.count(emoji_token) < 2:
                final_tweet_list.append(emoji_token)
        else:
            final_tweet_list.append(token)

    final_tweet = ' '.join(final_tweet_list)

    return final_tweet.strip()


x_train_norm = []
for tweet in x_train:
    x_train_norm.append(normalise_tweet(tweet))

x_test_norm = []
for tweet in x_test:
    x_test_norm.append(normalise_tweet(tweet))


"""Model"""

categories = [0, 1]

MODEL_NAME = 'roberta-base'

# Transormer is a wrapper to the Hugging Face transformers library for text classification.
t = text.Transformer(MODEL_NAME, maxlen=100, class_names=categories)

# Using normalised input data
trn = t.preprocess_train(x_train_norm, y_train)
val = t.preprocess_test(x_test_norm, y_test)

model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)

"""Train"""

best_lr = 5e-5


learner.autofit(lr=best_lr, verbose=1)


"""Evaluate/Inspect model"""

learner.validate(class_names=t.get_classes())

# the ones that we got most wrong
learner.view_top_losses(n=10, preproc=t)

"""Make predictions on new data"""

predictor = ktrain.get_predictor(learner.model, preproc=t)

test_sent = ('Cool it is raining again')

#predictor.predict(test_sent)

# Ask for explanation
#predictor.explain(test_sent)

more_sents = ['Going to the dentist for a root canal this afternoon. Yay, I can’t wait.',
              'It was so nice of my dad to come to my graduation party. #not',
              'I drank a healthy, homemade fruit smoothie...in a Budweiser glass #irony',
              'Dogs are really cute, one day I want to live in a big house with many dogs',
              'some trees are really tall, others not so much',
              'just came back from dinner at Nandos with my mates']

predictor.predict(more_sents)


"""Save"""

predictor.save('/my_models/ID_RoBERTa_emojis')


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
