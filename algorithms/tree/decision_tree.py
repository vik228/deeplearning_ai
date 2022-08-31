from cProfile import label
import numpy as np
import pandas as pd

class DecisionTree(object):

    def __init__(self, data, **kwargs):
        if isinstance(data, pd.DataFrame):
            self.data = data
        self.left = None
        self.right = None
    
    def gini_impurity(self):
        label = self.data[:, -1]
        return 1 - (np.sum(np.square(np.divide(label, np.sum(label)))))
    
    def is_pure_split(self):
        label_data = self.data[:, -1]
        return len(np.unique(label_data)) == 1
    
    def classify_data(self):
        label_column = self.data[:, -1]
        unique_values, counts = np.unique(label_column, return_counts=True)
        return unique_values[counts.argmax()]
    
    def entropy(self):
        class_values = self.data[:, -1]
        class_values = class_values[class_values > 0]
        total = np.sum(class_values)
        class_probablities = np.divide(class_values, total)
        log_probablities = np.log2(class_probablities)
        return -np.sum(class_probablities*log_probablities)
    
    def get_potential_splits_for_contineous_data(self):
        potential_splits = {}
        _, cols = self.X.shape
        for col_ndex in range(cols - 1):
            potential_splits[col_ndex] = []
            all_values = self.X[:, col_ndex]
            unique_values = np.unique(all_values)
            for index in range(1, len(unique_values)):
                potential_splits[col_ndex].append(((unique_values[index - 1] + unique_values[index])/2))
        return potential_splits
    
    def split_contineous_data(self, split_column, split_value):
        split_column_value = self.X[:, split_column]
        data_below = self.X[split_column_value <= split_value]
        data_above = self.X[split_column_value > split_value]
        return data_below, data_above
    
    def calculate_overall_entropy(self, split_column, split_value):
        data_below, data_above = self.split_contineous_data(split_column, split_value)
        total_data_points = len(data_below) + len(data_above)
        p_data_below = len(data_below) / total_data_points
        p_data_above = len(data_above) / total_data_points
        self._overall_entropy = (p_data_below*DecisionTree.entropy(data_below) + p_data_above*DecisionTree.entropy(data_above))
        return self._overall_entropy
    
    def determine_best_split(self):
        best_split_column = None
        best_split_value = None
        best_entropy = float('inf')
        potential_splits = self.get_potential_splits_for_contineous_data(self.X)
        for column_index, values in potential_splits.items():
            for value in values:
                overall_entropy = self.calculate_overall_entropy(column_index, value)
                if overall_entropy < best_entropy:
                    best_entropy = overall_entropy
                    best_split_column = column_index
                    best_split_value = value
        self._best_split_col = best_split_column
        self._best_split_value = best_split_value
        return best_split_column, best_split_value
    
    def fit(self):
        if self.is_pure_split():
            return
        else:
            best_split_col, best_split_value = self.determine_best_split()
            data_below, data_above = self.split_contineous_data(best_split_col, best_split_value)
            self.left = DecisionTree(data_below)
            self.right = DecisionTree(data_above)
            self.left.fit()
            self.right.fit()



