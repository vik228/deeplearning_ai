from algorithms.tree.decision_tree import DecisionTree
import numpy as np


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, X, y):
        DecisionTree.__init__(self, X, y)
