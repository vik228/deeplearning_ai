import numpy as np
from algorithms.text.preprocessing import TextPreprocessor


class NaiveBayes(object):
    def __init__(self, texts, sentiments):
        self.texts = texts
        self.sentiments = sentiments
        self.text_preprocessor = TextPreprocessor(texts)
        self.text_preprocessor.build_freqs(sentiments)
        self.freqs = self.text_preprocessor.freqs
        self.log_prior = 0
        self.log_likelihood = {}
        self.positives = [
            word for (word, sentiment) in self.freqs.keys() if sentiment == 1
        ]
        self.negatives = [
            word for (word, sentiment) in self.freqs.keys() if sentiment == 0
        ]
        self.all_words = list(set((self.positives + self.negatives)))

    def prior(self):
        D_positives = 0
        D_negatives = 0
        for sentiment in self.sentiments:
            if sentiment == 0:
                D_negatives = D_negatives + 1
            if sentiment == 1:
                D_positives = D_positives + 1
        self.log_prior = np.log(D_positives) - np.log(D_negatives)

    def likelihood(self):
        for word in self.all_words:
            p_w_pos = (self.freqs.get((word, 1), 0) + 1) / (
                len(self.positives) + len(self.all_words)
            )
            p_w_neg = (self.freqs.get((word, 0), 1) + 1) / (
                len(self.negatives) + len(self.all_words)
            )
            self.log_likelihood[word] = np.log(p_w_pos / p_w_neg)

    def fit(self):
        self.prior()
        self.likelihood()

    def predict(self, text):
        words = self.text_preprocessor.process_text(text)
        p = self.log_prior
        for word in words:
            if word in self.log_likelihood:
                p += self.log_likelihood[word]
        return p
    def accuracy(self, test_x, test_y):
        y_hats = []
        for sentance in test_x:
            if self.predict(sentance) > 0:
                y_hats.append(1)
            else:
                y_hats.append(0)
        error = np.mean(np.absolute(y_hats - test_y))
        accuracy = 1 - error
        return accuracy
