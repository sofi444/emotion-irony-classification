# ===============================
# Author: Sofia Casadei, Group 2
# Date last modified: 07/06/2021
# Module: CL Team Lab 2021
# Project: Emotion Classification
# ===============================

# Required imports:
import math
from collections import defaultdict


class Vectoriser:

    def __init__(self, train_data, test_data):

        """
            train_data:   a list of lists. Output of DataCleaner class.
            One inner list for each text sample:
            data[some number][0] = ID
                             [1] = emotion
                             [2] = text/sentence
        test_data:    as above

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
        """

        self.train_data = train_data

        # Unique words in training data
        self.unique_words = set()
        for sample in self.train_data:
            for word in sample[2].split():
                self.unique_words.add(word)

        # Map unique words (in training data) to integers
        # Index starts from 1 because bias will be added at index 0
        # {word: index}
        self.word_to_idx = {}
        index = 1
        for word in self.unique_words:
            self.word_to_idx[word] = index
            index += 1

        # This will contain the IDF values from training
        self.idf_dict = None
        self.test_data = test_data


    def TFIDF_training(self):
        # A list of dictionaries {words's index:count within that text sample}
        # len(count_words_in_sent) == number of samples in the corpus
        # for every sample --> a dictionary of len(unique_words).
        count_words_in_sent = []

        for sample in self.train_data:
            # [0] = ID
            # [1] = emotion
            # [2] = sentence
            sent = sample[2]
            count_words_in_sent.append(self.get_count(sent))

        # A list of dictionaries containing word's index: TF score of that word
        # within that text
        tf_list = []
        for i in range(len(self.train_data)):
            tf_list.append(self.calculate_tf(count_words_in_sent[i], self.train_data[i][2]))

        # Create global variable so that it can be used to vectorise test data
        # A dictionary {unique word: idf score}
        self.idf_dict = self.calculate_idf(count_words_in_sent)

        # A list of dictionaries
        # len == number of samples(sentences). One dictionary per sentence.
        # dictionaries of form:
        # {unique word's index: tfidf score for that word in that sentence}
        tfidf_list = []
        for tf in tf_list:
            tfidf_list.append(self.calculate_tfidf(tf, self.idf_dict))

        return tfidf_list


    def TFIDF_testing(self):

        count_words_in_sent = []

        for sample in self.test_data:
            # [0] = index
            # [1] = emotion
            # [2] = sentence
            sent = sample[2]

            count = defaultdict(int)
            for word in sent.split():

                # Words that appear in the test data, but not in the training
                # data will not have a corresponding index.
                # They can be disregarded because their IDF value would be 0
                # (and therefore their TF-IDF value would be zero)
                # At test time, we use the IDF values from training.
                if word in self.word_to_idx:

                    # Get index for this word and search/store that in 'count'
                    count[self.word_to_idx[word]] += 1

            count_words_in_sent.append(count)


        # A list of dictionaries containing word's index: TF score of that word
        # within that text
        tf_list = []

        for i in range(len(self.test_data)):
            tf_list.append(self.calculate_tf(count_words_in_sent[i],
                                             self.test_data[i][2]))

        # A list of dictionaries, len == number of samples(sentences).
        # One dictionary per sentence.
        # {unique word's index: tfidf score for that word in that sentence}
        tfidf_list = []
        for tf in tf_list:
            # Use idf_dict from training {unique word: idf score}
            tfidf_list.append(self.calculate_tfidf(tf, self.idf_dict))

        return tfidf_list


    ### TF-IDF HELPER FUNCTIONS: ###

    def get_count(self, sentence):
        '''
        Input is a string (one text sample)
        Returns a dictionary --> {word's index : its frequency within the doc}
        '''
        count = defaultdict(int)

        for word in sentence.split():
            # get index for this word and search/store that in the 'count' dict
            count[self.word_to_idx[word]] += 1

        return count


    def calculate_tf(self, words_dict, sentence):
        '''
        words_dict:   output of get_count (a dict)
            {unique word's index: its frequency in the given text}
        sentence:     a string, one text sample

        Term frequency = (Number of times term t appears in a document) /
                         (Total number of terms in the document)
        '''
        tf_dict = {}
        num_words = len(sentence.split())
        for word, count in words_dict.items():
            tf_dict[word] = count / num_words

        return tf_dict


    def calculate_idf(self, texts):
        '''
        Input is a list of dictionaries.
        There is one dictionary per sentence in the corpus and it contains
        the raw count of words within that sentence.

        IDF(t) = log_e(Total number of documents /
                        Number of documents with term t in it).
        '''
        # number of texts in the corpus
        N = len(texts)

        idf_dict = defaultdict(float)

        # Iterate through the list of dictionaries
        # 'sent' is a dict
        for sent in texts:
            for word, value in sent.items():
                # If word appeared in that sentence
                if value > 0:
                    # Increase of +1 when a sentence contains the relevant word
                    idf_dict[word] += 1

        #idf_dict now: {word1: num of sentences it appeared in, word2: ...}

        for word, value in idf_dict.items():
            idf_dict[word] = math.log(N / float(value))

        return idf_dict


    def calculate_tfidf(self, tf_scores_dict, idf_scores_dict):
        '''
        Takes as input one dictionary with tf scores at a time and
        the dictionary containing the idf scores (one per corpus).
        Returns a dictionary containing the TF-IDF scores for ONE text sample
        '''
        tfidf_dict = {}
        for word, value in tf_scores_dict.items():
            tfidf_dict[word] = value * idf_scores_dict[word]

        return tfidf_dict

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


vec = Vectoriser(clean_train_data, clean_test_data)

vec.TFIDF_training()
vec.TFIDF_testing()


Example of output:
[{117: 0.21309351152980316,
  43: 0.1129962809357643,
  118: 0.29973690909979633,
  14: 0.21309351152980316,
  45: 0.09855717004553378,
  33: 0.29973690909979633,
  25: 0.29973690909979633},
 {43: 0.07976208066053951,
  10: 0.1410526631057865,
  26: 0.1410526631057865,
  113: 0.07642841083119181,
  90: 0.1410526631057865,
  107: 0.1410526631057865,
  39: 0.1410526631057865,
  34: 0.05950593598108705,
  91: 0.1410526631057865,
  46: 0.1410526631057865,
  81: 0.1410526631057865,
  71: 0.10027929954343678,
  53: 0.1410526631057865,
  60: 0.07642841083119181,
  87: 0.1410526631057865},...]

"""
