import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor(object):

    def __init__(self, texts):
        self.texts = texts
        self.processed_text = []
        self.freqs = {}

    def process_text(self, text):
        """
            process text function
            The preprocessing involves following steps -:
                1. Remove all hyperlinks, hashtags
                2. Tokenizing the string
                3. Lowercasing
                4. Removing Stop words and punctuation
                5. Stemming
            input: text
            output: a list containing processed words.
        """
        stopwords_english = stopwords.words('english')
        stemmer = PorterStemmer()
        # Remove hyperlinks
        text = re.sub(r'^https|http?:\/\/.*[\r\n]*', '', text)
        # remove stock market tickers like $GE
        text = re.sub(r'\$\w*', '', text)
        # remove old style retweet text "RT"
        text = re.sub(r'^RT[\s]+', '', text)
        # remove hash tags
        text = re.sub(r'#', '', text)
        # tokenizing the string
        words = word_tokenize(text)
        # Lowercasing
        words = [word.lower() for word in words if word.strip()]
        # Removing stop word and punctuation
        words = [word for word in words if word not in string.punctuation and word not in stopwords_english]
        # Stemming
        words = [stemmer.stem(word) for word in words]
        self.processed_text.append(words)
        return words

    def process_texts(self):
        return [self.process_text(text) for text in self.texts]

    def build_freqs(self, sentiments):
        """
        Build frequencies.
            Input:
                texts: a list of texts
                sentiments: an m x 1 array with the sentiment label of each tweet
                    (either 0 or 1)
            Output:
                freqs: a dictionary mapping each (word, sentiment) pair to its
                frequency
        """
        sentiments = np.squeeze(sentiments).tolist()
        freqs = {}
        for sentiment, text in zip(sentiments, self.texts):
            for word in self.process_text(text):
                pair = (word, sentiment)
                freqs[pair]  = freqs.get(pair, 0) + 1
        self.freqs = freqs








    