# coding: utf-8
# Ref: http://darden.hatenablog.com/entry/2016/12/15/222447
import numpy as np
from ..utils import mk_color_dict
from ..utils import chooseTextColor
from ..utils import rgb2hex

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
        self.tree          = None
        self.criterion     = criterion
        self.max_depth     = max_depth
        self.random_state  = random_state
        self.tree_analysis = TreeAnalysis()

    def fit(self, x_train, y_train):
        num_samples, num_features = x_train.shape
        ini_classes = np.unique(y_train)
        self.tree = Node(criterion=self.criterion, max_depth=self.max_depth, random_state=self.random_state)
        self.tree.split_node(x_train=x_train, y_train=y_train, depth=0, ini_classes=ini_classes)
        self.feature_importances_ = self.tree_analysis.get_feature_importances(node=self.tree, num_features=num_features)
        self.num_features = num_features
        self.ini_classes = ini_classes

    def predict(self, x_train):
        predictions = np.asarray([self.tree.predict(x) for x in x_train])
        return predictions

    def score(self, x_train, y_train):
        return sum(self.predict(x_train) == y_train)/float(len(y_train))

    def export_graphviz(self, cmap="jet", feature_names=None, class_names=None):
        if feature_names is None:
            feature_names = [f"x{i+1}" for i in range(self.num_features)]
        if class_names is None:
            class_names = [f"cls{k+1}" for k in self.ini_classes]
        exporter = _DOTTreeExporter(cmap=cmap, feature_names=feature_names, class_names=class_names)
        return exporter.export(self.tree)

# Tree2Graphviz
class _DOTTreeExporter(object):
    def __init__(self, feature_names, class_names, cmap="jet"):
        self.num_node = None
        self.dot_data = None
        self.feature_names = feature_names
        self.class_names = class_names
        fill_color_dict = mk_color_dict(class_names, cmap=cmap, ctype="rgb")
        self.color_dict = {
            cls: {
                "fillcolor": rgb2hex(rgb_bg),
                "fontcolor":rgb2hex(chooseTextColor(rgb_bg))
            } for cls,rgb_bg in fill_color_dict.items()
        }

    def export(self, node):
        self.num_node = 0
        self.dot_data = ""
        self.head()
        self.recurse(node, 0)
        self.tail()
        return self.dot_data

    def head(self):
        self.dot_data = """digraph Tree {
        node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;
        edge [fontname=helvetica] ;
        """

    def tail(self):
        self.dot_data += "}"

    def recurse(self, node, parent_node_num):
        node.my_node_num = self.num_node
        node.parent_node_num = parent_node_num

        tree_str = ""
        if node.feature == None or node.depth == node.max_depth:
            tree_str += str(self.num_node) + " [label=<" + node.criterion + " = " + "%.4f" % (node.impurity) + "<br/>" \
                                           + "samples = " + str(node.num_samples) + "<br/>" \
                                           + "value = " + str(node.num_classes) + "<br/>" \
                                           + "class = " + self.class_names[node.label] + ">, fillcolor=\""+ self.color_dict.get(self.class_names[node.label]).get("fillcolor") + "\", fontcolor=\""+ self.color_dict.get(self.class_names[node.label]).get("fontcolor") +"\"] ;\n"
            if node.my_node_num!=node.parent_node_num:
                tree_str += str(node.parent_node_num) + " -> "
                tree_str += str(node.my_node_num)
                if node.parent_node_num==0 and node.my_node_num==1:
                    tree_str += " [labeldistance=2.5, labelangle=45, headlabel=\"True\"] ;\n"
                elif node.parent_node_num==0:
                    tree_str += " [labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ;\n"
                else:
                    tree_str += " ;\n"
            self.dot_data += tree_str
        else:
            tree_str += str(self.num_node) + " [label=<" + self.feature_names[node.feature] + " &le; " + str(node.threshold) + "<br/>" \
                                           + node.criterion + " = " + "%.4f" % (node.impurity) + "<br/>" \
                                           + "samples = " + str(node.num_samples) + "<br/>" \
                                           + "value = " + str(node.num_classes) + "<br/>" \
                                           + "class = " + self.class_names[node.label] + ">, fillcolor=\""+ self.color_dict.get(self.class_names[node.label]).get("fillcolor") + "\", fontcolor=\""+ self.color_dict.get(self.class_names[node.label]).get("fontcolor") +"\"] ;\n"
            if node.my_node_num!=node.parent_node_num:
                tree_str += str(node.parent_node_num) + " -> "
                tree_str += str(node.my_node_num)
                if node.parent_node_num==0 and node.my_node_num==1:
                    tree_str += " [labeldistance=2.5, labelangle=45, headlabel=\"True\"] ;\n"
                elif node.parent_node_num==0:
                    tree_str += " [labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ;\n"
                else:
                    tree_str += " ;\n"
            self.dot_data += tree_str

            self.num_node+=1
            self.recurse(node.left, node.my_node_num)
            self.num_node+=1
            self.recurse(node.right, node.my_node_num)
