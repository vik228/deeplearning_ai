import re
import nltk
import emoji
import numpy as np
from nltk.tokenize import word_tokenize

nltk.download("punkt")


class TextPreprator(object):
    def __init__(self, corpus) -> None:
        self.corpus = corpus
        self.word2ind = {}
        self.ind2word = {}
        self.vocabulary_size = 0
        self.tokenized_corpus = []
        self.tokenize()
        self.build_dict()

    def tokenize(self):
        corpus = re.sub(r"[,!?;-]", ".", self.corpus)
        self.tokenized_corpus = word_tokenize(corpus)
        self.tokenized_corpus = [
            ch.lower()
            for ch in self.tokenized_corpus
            if ch.isalpha() or ch == "." or emoji.is_emoji(ch)
        ]

    def get_context_and_center_words_for_window(self, context_half_size):
        i = context_half_size
        while i + context_half_size < len(self.tokenized_corpus):
            context_words = [
                *self.tokenized_corpus[(i - context_half_size) : i],
                *self.tokenized_corpus[i + 1 : (i + context_half_size + 1)],
            ]
            yield context_words, self.tokenized_corpus[i]
            i += 1

    def build_dict(self):
        sorted_words = sorted(list(set(self.tokenized_corpus)))
        ind2word = {}
        word2ind = {}
        for idx, word in enumerate(sorted_words):
            ind2word[word] = idx
            word2ind[idx] = word
        self.word2ind = word2ind
        self.ind2word = ind2word
        self.vocabulary_size = len(word2ind)

    def word_to_one_hot_vector(self, word):
        one_hot_vector = np.zeros(self.vocabulary_size)
        idx = self.ind2word[word]
        one_hot_vector[idx] = 1
        return one_hot_vector

    def context_words_to_one_hot_vector(self, context_words):
        context_words_vector = [
            self.word_to_one_hot_vector(context_word) for context_word in context_words
        ]
        context_words_vector = np.mean(context_words_vector, axis=0)
        return context_words_vector

    def build_training_data(self, context_half_size):
        for context_words, center_word in self.get_context_and_center_words_for_window(
            context_half_size
        ):
            yield self.context_words_to_one_hot_vector(
                context_words
            ), self.word_to_one_hot_vector(center_word)

    def get_batches(self, context_half_size, batch_size):
        batch_x = []
        batch_y = []
        for x, y in self.build_training_data(context_half_size):
            if len(batch_x) < batch_size:
                batch_x.append(x)
                batch_y.append(y)
            else:
                yield np.array(batch_x).T, np.array(batch_y).T
                batch_x = []
                batch_y = []
