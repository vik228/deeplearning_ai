import pandas as pd
import numpy as np


class LDA(object):

    def __init__(self, X, Y, **kwargs):
        self.X = pd.DataFrame(X)
        self.Y = pd.DataFrame(Y)
        self.__dict__.update(kwargs)
        self.overall_mean = None
        self.class_mean = None
        self.num_classes = None
        self.m_features = X.shape[1]

    @staticmethod
    def get_mean(data):
        columns = list(data.columns)
        means = []
        for column in columns:
            if column != 'label':
                means.append(data[column].mean())
        return np.array(means)

    def find_overall_and_class_mean(self, dfs, data):
        self.overall_mean = LDA.get_mean(data).reshape((self.m_features, 1))
        self.class_mean = {}
        for label, df in dfs.items():
            self.class_mean[label] = LDA.get_mean(df).reshape((self.m_features, 1))

    def calculate_eigen_values(self, sb, sw):
        mat = np.dot(np.linalg.pinv(sb), sw)
        eigvals, eigvec = np.linalg.eig(mat)
        eigvalslist = [(eigvals[i], eigvec[:, i]) for i in range(len(eigvals))]

    def fit(self):
        data = self.X.join(self.Y)
        labels = data['label'].unique()
        self.num_classes = len(labels)
        dfs = {}
        for label in labels:
            dfs[label] = data[data['label'] == label]
            dfs[label].drop(['label'], axis=1, inplace=True)
        self.find_overall_and_class_mean(dfs, data)
        sb = np.zeros((self.m_features, self.m_features))
        for label, class_mean in self.class_mean.items():
            nk = dfs[label].shape[0]
            diff = class_mean - self.overall_mean
            sb += nk*np.dot(diff, diff.T)
        sw = np.zeros((self.m_features, self.m_features))
        for label in labels:
            label_data = dfs[label].to_numpy()
            label_data = label_data.T
            label_mean = self.class_mean[label]
            diff = label_data - label_mean
            sw += np.dot(diff, diff.T)
        self.calculate_eigen_values(sb, sw)

