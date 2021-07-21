# ===============================
# Author: Sofia Casadei, Group 2
# Date last modified: 07/06/2021
# Module: CL Team Lab 2021
# Project: Emotion Classification
# ===============================

# Required imports:
# None


class MultiClassPerceptron():
    """

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

    """

    def __init__(self, x_train, y_train, x_test, y_test, epochs=10):

        # A list of dictionaries. One dict per sample: {word: TF-IDF value, ..}
        # Dictionaries have different lengths as they only have entries for
        # words that have a non-zero TF-IDF value (only words that occurred)
        self.x_train = x_train

        # Add bias term (1) to each sample at index 0
        for sample in self.x_train:
            sample[0] = 1

        # A list of labels
        self.y_train = y_train

        # A list of dictionaries
        self.x_test = x_test
        # A list of labels
        self.y_test = y_test

        # 7 emotions -> 7 classes
        self.classes = list(set(self.y_train))

        #Unique words
        self.features = set()
        for sample in x_train:
            for word in sample.keys():
                self.features.add(word)

        self.epochs = epochs

        # As many weight vectors as classes
        # Each weight has as many elements as features + extra bias term
        # Initialised to zeros
        # len(self.weight_vectors) == 7
        # len(self.weight_vectors['joy']) == number of unique words + 1
        # {emotion1: [zeros], emotion2: [zeros], ..}
        self.weight_vectors = {c: [0 for i in range(len(self.features) + 1)]
                               for c in self.classes}


    def train(self):
        for i in range(self.epochs):

            # Keeps track of iterations through samples
            tally = 0

            # 'sample' is a dictionary {word's index: tf-idf value, ..}
            for sample in self.x_train:

                # Initialize arg_max value and predicted class randomly.
                arg_max = 0
                predicted_class = self.classes[0]

                # Multi-Class Decision Rule:
                for c in self.classes:

                    # Dot product
                    current_activation = 0.0
                    for idx_of_word, value in sample.items():
                        prod = value * self.weight_vectors[c][idx_of_word]
                        current_activation += prod

                    if current_activation >= arg_max:
                        arg_max, predicted_class = current_activation, c


                # Update rule
                correct_class = self.y_train[tally]

                if not (correct_class == predicted_class):

                    for idx_of_word, value in sample.items():

                        self.weight_vectors[correct_class][idx_of_word] += value
                        self.weight_vectors[predicted_class][idx_of_word] -= value

                tally += 1


    def predict(self):

        # A list containing predictions for all samples in the test data
        predictions = []

        for sample in self.x_test:
            arg_max = 0
            predicted_class = self.classes[0]

            # Multi-Class Decision Rule:
            for c in self.classes:

                # Dot product
                current_activation = 0.0
                # test data: {index: tf-idf value}
                for idx_of_word, value in sample.items():
                    prod = value * self.weight_vectors[c][idx_of_word]
                    current_activation += prod

                if current_activation >= arg_max:
                    arg_max = current_activation
                    predicted_class = c

            predictions.append(predicted_class)

        return predictions


"""
To run this code:

# Just some toy data
clean_train_data = [[0, 'joy', 'when i understood that i was admitted university'],
 [1,'fear',"i broke window of neighbouring house i feared my mother's judgement action on what i had done"],
 [2, 'joy', 'got big fish in fishing'],
 [3,'fear','whenever i am alone in dark room walk alone on street sleep alone in room at night or see something which is only partly visible this emotion was very strong when as 8 year old child i saw something horrible'],
 [4,'shame','i bought possible answer homework problem which was completely inapplicable question due my having read about subject matter'],
 [5,'disgust','i read about murderer who brutalized his victims by cutting open their stomaches taking out their bowels'],
 [6,'joy','day that my boyfriend appeared at home with pair of rings for our wedding'],
 [7,'guilt','i went pub with group of friends very close i was with one girl most of time while other girls in group wanted be with me they stopped talking girl i was with'],
 [8, 'anger', 'had insulting letter from my father'],
 [9, 'sadness', 'no response'],
 [10, 'fear', 'i was be given audition get role i had competitress i wasnot well prepared because i was ill']]

clean_test_data = [65,'anger','unjust accusations directed at me my way of acting by someone close me'],
 [66,'anger','i was angry at cafeteria when cook scolded said many bad things about me without reason he thought i was among girls who did him wrong'],
 [67,'fear','my mother hadnot come home at midnight she had forgotten tell me about it i was very young all alone at home'],
 [68,'fear','when our school was raided by pupils of boys secondary school who beat us up'],
 [69, 'sadness', 'very close friend left me'],
 [70, 'joy', 'saw in TV that china had most gold medals in asian olympic'],
 [71,'fear','my episode of fright happened when i came study i had stand in front of my still unknown fellow students talk about myself my hands shook i flushed became tonguetied']]


# Extract training labels
y_train = []
for sample in clean_train_data:
    y_train.append(sample[1])

# Extract test labels
y_test = []
for sample in clean_test_data:
    y_test.append(sample[1])


# Vectorise training and test data
from vectoriser import Vectoriser

vectoriser = Vectoriser(clean_train_data, clean_test_data)

x_train = vectoriser.TFIDF_training()
x_test = vectoriser.TFIDF_testing()


# Training and making predictions
perceptron = MultiClassPerceptron(x_train, y_train, x_test, y_test, epochs=10)
perceptron.train()

predictions = perceptron_v.predict()


Example of output:
['joy',
 'shame',
 'sadness',
 'joy',
 'guilt',
 'sadness',
 'sadness',
 'shame',
 'sadness',
 'fear',
 'shame',
 'fear',
 'sadness',
 'sadness',
 'shame', ...]

"""
