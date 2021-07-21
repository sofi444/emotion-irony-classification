# CL-Team-Lab

### Emotion Classification
### Group 2
### Sofia Casadei, Sakshi Shukla


The code in the repository is an emotion classification baseline.
The data we have used is the ISEAR dataset, which is single labeled and self reported.
The machine learning algorithm used here is a multiclass perceptron.

#### By running main.py, all the other files and relative classes are imported and the output is the F-Score obtained on the test data.

More specifically:

#### data_cleaning.py
contains all the functions needed to handle the raw dataset, 
cleaning the text and transforming the data to a suitable format which will be fed to the Vectoriser class.
It performs the following pre-processing steps: 
punctuation removal, regularisation of case, regularisation of contractions, handling of negation.

#### vectoriser.py
contains a class that vectorises the data.
This class takes an input the cleaned data and vectorises the text
samples. To do so, it uses TF-IDF. On the training data, TF-IDF is
calculated from scratch, meaning that both Term Frequency (TF - sample
specific) and Inverse Document Frequency (IDF - corpus specific) are
computed on the training data and them multiplied to obtain the final
TF-IDF score for each word. On the other hand, when vectorising the
test data, TF is calculated on the incoming samples but the IDF values
used are the ones calculated on the training data.

Each word in the training set is mapped to an integer, which is then
used in the final representation. The words that do not appear are not
recorded, as they would anyway have a score of 0. This also helps to
make the code more space and time efficient.

Output:     A list of dictionaries. Every dictionary represents a text
            sample. keys are integers that represent words and values
            are TF-IDF scores.
           
#### perceptron.py
contains our ML model.
This model is trained on labeled data and is then able to make a prediction
on unseen text samples. It predicts an emotion for a given text sample.

The key mechanism consists of a multiclass decision rule and an update rule:
For each class (there are 7 classes i.e., 7 emotions labels) we have a
weight vector. We calculate the dot product between a given text sample and
each weight vector, thus obtaining an 'activation'. The class that yields
the highest activation will be chosen by the perceptron as the predicted
class for that text sample.
Then, during training, the prediction is compared with the correct label.
If the prediction is correct, nothing happens. However, if the predcition
is wrong, the weight vector corresponding to the correct class is boosted
by adding to it the text sample's vector, and the weight vector
corresponding to the predicted class is weakened by subtracting from it
the text sample's vector. This procedure is repeated for every text sample
in the training data. This is how we train the model's parameters.

At test time, the model calcualtes the dot product between the input text's
vector and the weight vectors of each class. The class corresponding to the
highest activation is the model's prediction.

#### evaluation.py
contains a class 'Evaluation'.
This class evaluates the classifier's predicitons. It uses precision,
recall and F1-Score. More specifically, it uses macro average,
meaning that all the classes weigh equally. It creates a confusion matrix
which is then used to retrieve the values needed for all other calculations.

Inputs are two lists of labels: the correct ones from the annotated
dataset and the ones predicted by the classifier.
