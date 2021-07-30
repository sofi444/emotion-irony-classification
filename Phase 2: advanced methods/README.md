## The Interplay between Emotions and Irony in Text Classification

### Abstract

Irony is an intricate phenomenon that has the
potential to undermine Natural Language Processing
(NLP) systems for the classification of
emotions, as it can mask the true sentiment
conveyed by a text. This study analyses the
effects of giving classifiers information regarding
the presence of irony in Twitter data, revealing
that this is not sufficient to improve
performance. Similarly, irony detectors do not
seem to benefit from the addition of information
about the emotion of a tweet. Moreover,
a feature analysis revealed that there may be a
few soft linguistic indicators of irony. Finally,
co-occurrence patterns between irony and both
positive and negative emotions were found,
providing additional evidence for the complexity
of this phenomenon.

### Classifiers:
1) Emotion classifier: RoBERTa, with preprocessing
2) Emotion classifier: RoBERTa, without preprocessing
3) Emotion classifier: RoBERTa, with emojis
4) Emotion classifier: RoBERTa, without emojis
5) Irony Detector: RoBERTa, with emojis
6) Irony Detector: RoBERTa, without emojis

The classifiers are in the form of .py files

### Experiments:
1) Experiment 1: do emotion labels improve irony detection performance?
2) Experiment 2: do irony labels improve emotion classification performance?
3) Experiment 3: are there features that can be considered indicators of irony?
4) Experiment 4: what emotions does irony appear most often with?

The experiments can be found in this repository (notebooks), however, I suggest opening them directly by clicking on this link. It directs to a folder with the original Google Colab notebooks, with table of contents that are useful to browse through the files.

==> https://drive.google.com/drive/folders/1gbBdu2qsnkkPnGvcTT4-kYsM72SF1sdL?usp=sharing
