import numpy as np


class DecisionTreeClassifier(object):

    def __init__(self, height, **kwargs):
        self.impurity = None
        self.max_height = height
        self.__dict__.update(kwargs)
        self.trees = {}
        self.height = 0

    @staticmethod
    def get_entropy_of_a_single_nodes(y_data):
        n_classes = len(np.unique(y_data))
        class_array = np.zeros((1, n_classes))
        for i in range(0, len(y_data)):
            class_array[0][y_data[i] - 1] += 1
        total = np.sum(class_array)
        individual_entropies = np.divide(class_array, total)
        log_entropies = np.log2(individual_entropies) * -1
        return np.sum(np.dot(individual_entropies, log_entropies.T))

    @staticmethod
    def get_entropy_for_multiple_nodes(y_real, y_predict, total):
        total_array = np.zeros((1, 2))
        node_entropies = np.zeros((1, 2))
        if len(y_real) != len(y_predict):
            return None
        i = 0
        nodes = {
            'node_0': y_real[y_predict],
            'node_1': y_predict[~y_predict]
        }
        for _node, y_node_data in nodes.items():
            total_array[0][i] = len(y_node_data) / total
            node_entropies[0][i] = DecisionTreeClassifier.get_entropy_of_a_single_nodes(y_node_data)
            i += 1
        return np.sum(np.dot(node_entropies, total_array.T))

    @staticmethod
    def find_best_split(col, y):
        min_entropy = 10
        total = len(y)
        cutoff = None
        for value in set(col):
            y_predict = col < value
            entropy = DecisionTreeClassifier.get_entropy_for_multiple_nodes(
                y, y_predict, total)
            if entropy < min_entropy:
                min_entropy = entropy
                cutoff = value
        return min_entropy, cutoff

    @staticmethod
    def find_best_split_of_all(x, y):
        min_entropy = 0
        final_col = None
        final_cutoff = None
        for i, col in enumerate(x.T):
            entropy, cutoff = DecisionTreeClassifier.find_best_split(col, y)
            if entropy == 0:
                return col, cutoff, min_entropy
            elif entropy < min_entropy:
                final_cutoff = cutoff
                final_col = col
                min_entropy = entropy
        return final_col, final_cutoff, min_entropy

    @staticmethod
    def all_same(y):
        return len(list(set(y))) == 1

    def fit(self, x, y, node={}, height=0):
        if node is None:
            return None
        elif len(y) == 0:
            return None
        elif self.max_height <= height:
            return None
        elif DecisionTreeClassifier.all_same(y):
            return None
        else:
            col, cutoff, entropy = DecisionTreeClassifier.find_best_split_of_all(x, y)
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]
            node = {'col': col, 'cutoff': cutoff, 'val': np.round(np.mean()),
                    'left': self.fit(x[x[:, col] < cutoff], y_left, {}, height + 1),
                    'right': self.fit(x[x[:, col] >= cutoff], y_right, {}, height + 1)}
            self.height += 1
            self.trees = node
        return node
