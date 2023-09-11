from __future__ import annotations

import math
from collections import Counter

import nltk
import numpy as np
from nltk.util import ngrams

nltk.download("punkt")


class PerformanceMetrics:
    """
    Defines methods to implement the model

    parameters
    ----------
    y_actual : array-like, shape = [n_samples]
        Observed values from the training samples

    y_predicted : array-like, shape = [n_samples]
        Predicted value from the model
    """

    def __init__(self, y_actual, y_predicted):
        self.y_actual = y_actual
        self.y_predicted = y_predicted

    def mean_squared_error(self):
        """Compute the root mean squared error
        Returns
        ------
        rmse : mean squared error
        """
        m = self.y_actual.shape[0]
        mse = self.sum_of_squares_of_residuals()
        return mse / m

    def mean_absolute_error(self):
        """Compute the mean absolute error
        Returns
        ------
        mae : mean_absolute_error
        """
        n = self.y_actual.shape[0]
        mae = np.sum(np.abs(self.y_predicted - self.y_actual))
        return mae / n

    def root_mean_squared_error(self):
        """Compute the root mean squared error
        Returns
        ------
        rmse : root mean squared error
        """
        return np.sqrt(self.mean_squared_error())

    def r2_score(self):
        """Compute the r-squared score
        Returns
        ------
        r2_score : r-squared score

        RSS -> Sum of squares of residuals
        TSS -> Total sum of squares
        """

        RSS = self.sum_of_squares_of_residuals()
        TSS = self.total_sum_of_squares()
        return 1 - (RSS / TSS)

    def total_sum_of_squares(self):
        return np.sum((self.y_actual - np.mean(self.y_actual))**2)

    def sum_of_squares_of_residuals(self):
        return np.sum((self.y_actual - self.y_predicted)**2)

    def brevity_penalty(self, candidate, reference):
        ref_len = len(reference)
        can_len = len(candidate)
        if ref_len < can_len:
            return 1
        return np.exp(1 - (ref_len / can_len))

    def clipped_precision(self, candidate, reference):
        clipped_precision_scores = []
        for n_gram in range(1, 5):
            candiate_ngram = Counter(ngrams(candidate, n_gram))
            reference_ngram = Counter(ngrams(reference, n_gram))
            total_candidate_ngrams = sum(candiate_ngram.values())
            for c_ngram in candiate_ngram:
                candiate_ngram[c_ngram] = min(candiate_ngram[c_ngram],
                                              reference_ngram.get(c_ngram, 0))
            precision = sum(reference_ngram.values()) / total_candidate_ngrams
            clipped_precision_scores.append(precision)
        wieghts = [0.25] * 4
        s = (w_i * math.log(p_i)
             for w_i, p_i in zip(wieghts, clipped_precision_scores))
        return math.exp(math.fsum(s))

    def bleu_score(self, candidate, reference):
        bp = self.brevity_penalty(candidate, reference)
        precision = self.clipped_precision(candidate, reference)
        return precision * bp
