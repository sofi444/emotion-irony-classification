# ===============================
# Author: Sofia Casadei, Group 2
# Date last modified: 07/06/2021
# Module: CL Team Lab 2021
# Project: Emotion Classification
# ===============================

# Required imports:
import pandas as pd
import re


class DataCleaner:
    """
    This class contains all the functions needed to handle the raw dataset,
    cleaning the text and transforming the data to a suitable format which
    will be fed to the Vectoriser class.

    Raw data:   CSV file, two columns: emotion label and text
    Output:     List of lists. Each inner list represents a text sample and
                contains the following elements:
                [0] = ID
                [1] = emotion label
                [2] = clean text sample
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def openCSV_organize(self):
        # Open CSV file and name columns of dataframe
        data_raw = pd.read_csv(self.file_path, error_bad_lines=False,
                               names=['Emotion', 'Sentence'])

        # Remove bad lines
        data_raw = data_raw[data_raw['Emotion'].str.len().lt(10)]

        # Reset index. line above completely removes indices
        # (avoid errors later in for loops)
        data = data_raw.reset_index(drop=True)

        # Organize data in a list of lists
        # One inner list for each text sample
        # idx_emotion_sent[some number][0] = ID
                                     # [1] = emotion
                                     # [2] = text/sentence

        self.idx_emotion_sent = []

        for i in range(len(data)):
            self.idx_emotion_sent.append([i, data['Emotion'][i],
                                         data['Sentence'][i]])

    def clean_text(self, keep_allcap=False):
        """
        This function cleans the text: punctuation removal, regularisation of
        case, regularisation of contractions, handling of negation. It calls
        the helper functions merge_negation() and regularise_contractions().

        keep_allcap:    If set to True, all words will be lowercased except
                        the ones that consist only of uppercase characters.
                        (boolean. Default is False)

        # NOTE: currently it removes only a few stop words as some (e.g., 'I')
        were found to be significant for emotion classification. Below are the
        stop words included in the NLTK english stop words collections, as
        additional ones might be removed in the future.

        ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now"]
        """

        for datapoint in self.idx_emotion_sent:
            # Remove punctuation
            datapoint[2] = re.sub(r"[-()\"#/@;:<>{}=~|.?,\[\]]", "",
                                  datapoint[2])

            # Regularise case
            text = ''
            for word in datapoint[2].split():
                if word.isupper() and len(word) > 1 and keep_allcap==True:
                    # Lowercase everything except words that are made up
                    # exclusively of uppercase characters.
                    text += word + " "
                else:
                    # Lowercase everything
                    word = word.lower()
                    text += word + " "
            datapoint[2] = text

            # Remove stop words
            stop_words = ['and', 'the', 'an', 'a', 'to', 'of', 'or']
            nostops = [word for word in datapoint[2].split() if not word
                       in stop_words]
            datapoint[2] = ' '.join(nostops)

            # Regularise contractions
            # (e.g., "we'll" will be tokenised as "we", "will")
            datapoint[2] = self.regularise_contractions(datapoint[2])

            # Merge negation
            # modals followed by not become a unique token
            # e.g., 'doesnot'
            datapoint[2] = self.merge_negation(datapoint[2])


        return self.idx_emotion_sent

    ### HELPER FUNCTIONS ###

    def merge_negation(self, text):
        """
        Input (text) is a string
        """
        modals = ['do', 'did', 'is', 'are', 'was', 'were','can',
                  'could', 'would', 'shall', 'have', 'had', 'has']

        tokens = text.split()
        # Add NULL token to avoid Error: index out of range later
        tokens.append('NULL')

        new_text = []
        for i in range(len(tokens)):

            if tokens[i] in modals and tokens[i+1]=='not':
                merge = tokens[i]+tokens[i+1]
                new_text.append(merge)
            else:
                if tokens[i] != 'not' and tokens[i] != 'NULL':
                    new_text.append(tokens[i])

        new_text = ' '.join(new_text)

        return new_text

    def regularise_contractions(self, text):
        text = re.sub(r"\'ll", "will", text)
        text = re.sub(r"\'ve", "have", text)
        text = re.sub(r"\'re", "are", text)
        text = re.sub(r"\'d", "would", text)
        text = re.sub(r"i[ ]*'[ ]*m", "i am", text)
        text = re.sub(r"he[ ]*'[ ]*s", "he is", text)
        text = re.sub(r"she[ ]*'[ ]*s", "she is", text)
        text = re.sub(r"that[ ]*'[ ]*s", "that is", text)
        text = re.sub(r"what[ ]*'[ ]*s", "what is", text)
        text = re.sub(r"where[ ]*'[ ]*s", "where is", text)
        text = re.sub(r"there[ ]*'[ ]*s", "there is", text)
        text = re.sub(r"won[ ]*'[ ]*t", "will not", text)
        text = re.sub(r"can[ ]*'[ ]*t", "can not", text)
        text = re.sub(r"isn[ ]*'[ ]*t", "is not", text)
        text = re.sub(r"aren[ ]*'[ ]*t", "are not", text)
        text = re.sub(r"couldn[ ]*'[ ]*t", "could not", text)
        text = re.sub(r"wasn[ ]*'[ ]*t", "was not", text)
        text = re.sub(r"weren[ ]*'[ ]*t", "were not", text)
        text = re.sub(r"wouldn[ ]*'[ ]*t", "would not", text)
        text = re.sub(r"hadn[ ]*'[ ]*t", "had not", text)
        text = re.sub(r"hasn[ ]*'[ ]*t", "has not", text)
        text = re.sub(r"can[ ]*'[ ]*t", "can not", text)
        text = re.sub(r"don[ ]*'[ ]*t", "do not", text)
        text = re.sub(r"didn[ ]*'[ ]*t", "did not", text)

        return text

"""
To run this code:

clean = DataCleaner('### insert file (CSV) path here ###')
clean.openCSV_organize()
clean_text = clean.clean_text()

Example of output:
[[0,
  'joy',
  'when i understood that i was admitted university'],
 [1,
  'fear',
  "i broke window of neighbouring house i feared my mother's judgement action
  on what i had done"],
 [2, 'joy', 'got big fish in fishing'],
 [3,
  'fear',
  'whenever i am alone in dark room walk alone on street sleep alone in room
  at night or see something which is only partly visible this emotion was very
  strong when as 8 year old child i saw something horrible'],
 [4,
  'shame',
  'i bought possible answer homework problem which was completely inapplicable
  question due my having read about subject matter'],
 [5,
  'disgust',
  'i read about murderer who brutalized his victims by cutting open their
  stomaches taking out their bowels'], ...]
"""
