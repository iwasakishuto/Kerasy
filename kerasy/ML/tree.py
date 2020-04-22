# coding: utf-8
# Ref: http://darden.hatenablog.com/entry/2016/12/15/222447
import os
import numpy as np
from ..utils import DecisionTreeDOTexporter
from ..utils import DOTexporterHandler

def split_data(data, cond):
    return (data[cond], data[~cond])

class Node():
    """ Node for Tree structure.
    @params depth       : (int)   Depth (Root: depth=0)
    @params left        : (Node)  Left side node.
    @params right       : (Node)  Right side node.
    @params feature     : (int)   Index of the features.
    @params threshold   : (float) Threshold for dividing
    @params label       : (int)   Class label for this node. It is used when this node is the last one.
    @params impurity    : (float) Node impurity.
    @params info_gain   : (float) How much information was obtained by dividing at this node.
    @params num_samples : (int)   The number of samples which flowed into the node
    @params num_classes : (list)  The number of samples which belongs to each class.
    """
    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        self.criterion    = criterion
        self.max_depth    = max_depth
        self.random_state = random_state
        self.depth        = None
        self.left         = None
        self.right        = None
        self.feature      = None
        self.threshold    = None
        self.label        = None
        self.impurity     = None
        self.info_gain    = None
        self.num_samples  = None
        self.num_classes  = None

    def split_node(self, x_train, y_train, depth, ini_classes):
        self.depth = depth
        self.num_samples, num_features = x_train.shape
        self.num_classes = [np.count_nonzero([y_train==k]) for k in ini_classes]

        unique_classes = np.unique(y_train)
        if len(unique_classes) == 1: # We don't have to divide!!
            self.label = unique_classes[0]
            self.impurity = self.criterion_func(y_train)
            return

        class_count = {cls: np.count_nonzero(y_train==cls) for cls in np.unique(y_train)}
        self.label  = max(class_count.items(), key=lambda x:x[1])[0]
        self.impurity = self.criterion_func(y_train)

        self.info_gain = 0.0
        # The order of looking at features. (If Information gain is equal, the fastest one is given priority.)
        f_order = np.random.RandomState(self.random_state).permutation(num_features).tolist()
        for f in f_order:
            uniq_feature = np.unique(x_train[:, f]) # NOTE: "uniq_feature" are sorted!!
            split_points = (uniq_feature[:-1] + uniq_feature[1:]) / 2.0

            for threshold in split_points:
                y_train_l, y_train_r = split_data(data=y_train, cond=x_train[:,f]<=threshold)
                val = self.calc_info_gain(y_train, y_train_l, y_train_r)
                if self.info_gain < val:
                    self.info_gain = val
                    self.feature   = f
                    self.threshold = threshold

        if self.info_gain == 0.0: return
        if depth == self.max_depth: return

        #=== Recursion ===
        x_train_l, x_train_r = split_data(data=x_train, cond=x_train[:,self.feature]<=self.threshold)
        y_train_l, y_train_r = split_data(data=y_train, cond=x_train[:,self.feature]<=self.threshold)
        # Left Node
        self.left  = Node(self.criterion, self.max_depth)
        self.left.split_node(x_train_l, y_train_l, depth+1, ini_classes)
        # Left Node
        self.right = Node(self.criterion, self.max_depth, ini_classes)
        self.right.split_node(x_train_r, y_train_r, depth+1, ini_classes)

    def criterion_func(self, y_train):
        num_classes = np.unique(y_train)
        num_data = len(y_train)

        if self.criterion == "gini":
            """ IG(t) = 1 - sum_{i=1}^c p(i|t)^2  """
            val = 1 - np.sum([(np.count_nonzero(y_train==cls)/num_data)**2 for cls in num_classes])

        elif self.criterion == "entropy":
            """ IH(t) = - sum_{i=1}^c p(i|t)log p(i|t) """
            val = 0
            for c in classes:
                p = np.count_nonzero(y_train==c)/num_data
                if p != 0.0: val -= p * np.log2(p)
        return val

    def calc_info_gain(self, y_train_all, y_train_l, y_train_r):
        """ Information gain = I_before - (wL・I_after_L + wR・I_after_R) """
        I_before = self.criterion_func(y_train_all); Nall = len(y_train_all)
        I_afterL = self.criterion_func(y_train_l); NL = len(y_train_l)
        I_afterR = self.criterion_func(y_train_r); NR = len(y_train_r)
        InformationGain = I_before - (NL/Nall*I_afterL + NR/Nall*I_afterR)
        return InformationGain

    def predict(self, x_train):
        if self.feature == None or self.depth == self.max_depth:
            return self.label
        else:
            if x_train[self.feature] <= self.threshold: return self.left.predict(x_train)
            else: return self.right.predict(x_train)

class TreeAnalysis():
    """ Calcurate the feature importances. """
    def __init__(self):
        self.num_features = None
        self.importances  = None

    def compute_feature_importances(self, node):
        if node is None or node.feature is None: return
        self.importances[node.feature] += node.info_gain*node.num_samples
        self.compute_feature_importances(node.left)
        self.compute_feature_importances(node.right)

    def get_feature_importances(self, node, num_features, normalize=True):
        self.importances  = np.zeros(num_features)
        self.compute_feature_importances(node)
        self.importances /= node.num_samples

        if normalize:
            normalizer = np.sum(self.importances)
            if normalizer > 0.0: self.importances /= normalizer # Avoid dividing by zero (e.g., when root is pure)
        return self.importances

class DecisionTreeClassifier():
    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        self.root          = None
        self.criterion     = criterion
        self.max_depth     = max_depth
        self.random_state  = random_state
        self.root_analysis = TreeAnalysis()

    def fit(self, x_train, y_train):
        num_samples, num_features = x_train.shape
        ini_classes = np.unique(y_train)
        self.root = Node(criterion=self.criterion, max_depth=self.max_depth, random_state=self.random_state)
        self.root.split_node(x_train=x_train, y_train=y_train, depth=0, ini_classes=ini_classes)
        self.feature_importances_ = self.root_analysis.get_feature_importances(node=self.root, num_features=num_features)
        self.num_features = num_features
        self.ini_classes = ini_classes

    def predict(self, x_train):
        predictions = np.asarray([self.root.predict(x) for x in x_train])
        return predictions

    def score(self, x_train, y_train):
        return sum(self.predict(x_train) == y_train)/float(len(y_train))

    def export_graphviz(self, out_file=None, feature_names=None,
                        class_names=None, cmap="jet", filled=True,
                        rounded=True, precision=3):
        if feature_names is None:
            feature_names = [f"x{i+1}" for i in range(self.num_features)]
        if class_names is None:
            class_names = [f"cls{k+1}" for k in self.ini_classes]
        exporter = DecisionTreeDOTexporter(
            cmap=cmap, feature_names=feature_names, class_names=class_names,
            filled=filled, rounded=rounded, precision=precision
        )
        return DOTexporterHandler(exporter, root=self.root, out_file=out_file)
