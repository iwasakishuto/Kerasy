# coding: utf-8
from .vis_utils import mk_color_dict
from .vis_utils import chooseTextColor
from .vis_utils import rgb2hex

class _DOTTreeExporter(object):
    def __init__(self, feature_names, class_names, cmap="jet"):
        self.node_id = None
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
        self.node_id = 0
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

    def recurse(self, node, par_node_id):
        my_node_num = self.node_id

        tree_str = ""
        if node.feature == None or node.depth == node.max_depth:
            tree_str += str(self.node_id) + " [label=<" + node.criterion + " = " + "%.4f" % (node.impurity) + "<br/>" \
                                           + "samples = " + str(node.num_samples) + "<br/>" \
                                           + "value = " + str(node.num_classes) + "<br/>" \
                                           + "class = " + self.class_names[node.label] + ">, fillcolor=\""+ self.color_dict.get(self.class_names[node.label]).get("fillcolor") + "\", fontcolor=\""+ self.color_dict.get(self.class_names[node.label]).get("fontcolor") +"\"] ;\n"
            if my_node_num!=par_node_id:
                tree_str += str(par_node_id) + " -> "
                tree_str += str(my_node_num)
                if par_node_id==0 and my_node_num==1:
                    tree_str += " [labeldistance=2.5, labelangle=45, headlabel=\"True\"] ;\n"
                elif par_node_id==0:
                    tree_str += " [labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ;\n"
                else:
                    tree_str += " ;\n"
            self.dot_data += tree_str
        else:
            tree_str += str(self.node_id) + " [label=<" + self.feature_names[node.feature] + " &le; " + str(node.threshold) + "<br/>" \
                                           + node.criterion + " = " + "%.4f" % (node.impurity) + "<br/>" \
                                           + "samples = " + str(node.num_samples) + "<br/>" \
                                           + "value = " + str(node.num_classes) + "<br/>" \
                                           + "class = " + self.class_names[node.label] + ">, fillcolor=\""+ self.color_dict.get(self.class_names[node.label]).get("fillcolor") + "\", fontcolor=\""+ self.color_dict.get(self.class_names[node.label]).get("fontcolor") +"\"] ;\n"
            if my_node_num!=par_node_id:
                tree_str += str(par_node_id) + " -> "
                tree_str += str(my_node_num)
                if par_node_id==0 and my_node_num==1:
                    tree_str += " [labeldistance=2.5, labelangle=45, headlabel=\"True\"] ;\n"
                elif par_node_id==0:
                    tree_str += " [labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ;\n"
                else:
                    tree_str += " ;\n"
            self.dot_data += tree_str

            self.node_id+=1
            self.recurse(node.left, my_node_num)
            self.node_id+=1
            self.recurse(node.right, my_node_num)
