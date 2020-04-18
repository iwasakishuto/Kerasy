# coding: utf-8
import os

from .generic_utils import handleKeyError
from .generic_utils import handleTypeError
from .vis_utils import mk_color_dict
from .vis_utils import chooseTextColor
from .vis_utils import rgb2hex

BLACK_hex = "#000000"
WHITE_hex = "#FFFFFF"

TREE_STRUCTURES = ["lr", "children"]

def DOTexporterHandler(exporter, root, out_file=None):
    if out_file is not None:
        ext = os.path.splitext(os.path.basename(out_file))[-1]
        if ext==".png":
            return exporter.write_png(root, path=out_file)
        elif ext==".dot":
            return exporter.export(root, out_file=out_file)
    else:
        return exporter.export(root, out_file=None)


class BaseTreeDOTexporter():
    def __init__(self, feature_names=None, class_names=None, cmap="jet",
                 filled=True, rounded=True, precision=3, tree_structure="lr"):
        self.node_id = None
        self.dot_text = None
        self.feature_names = feature_names
        self.class_names = class_names
        self.filled = filled
        self.rounded = rounded
        self.precision = precision
        handleKeyError(lst=TREE_STRUCTURES, tree_structure=tree_structure)
        self.tree_structure = tree_structure
        # Coloring according to `class names`.
        if class_names is not None:
            self._mk_color_dict(class_names=class_names, cmap=cmap)

    def _mk_color_dict(self, class_names, cmap):
        bg_color_dict = mk_color_dict(class_names, cmap=cmap, ctype="rgb")
        fw_color_dict = {
            cls: rgb2hex(chooseTextColor(rgb_bg)) for cls,rgb_bg in bg_color_dict.items()
        }
        bg_color_dict = {cls : rgb2hex(rgb_bg) for cls,rgb_bg in bg_color_dict.items()}
        if not self.filled:
            fw_color_dict = {cls : BLACK_hex for cls,rgb_bg in fw_color_dict.items()}
        self.fill_color_dict = bg_color_dict
        self.font_color_dict = fw_color_dict

    def _add_head(self):
        head_info = "digraph Tree {\nnode [shape=box"
        # Specify node aesthetics (`round`, and `filled`).
        rounded_filled = []
        rounded_filled.extend(["filled"]  if self.filled  else [])
        rounded_filled.extend(["rounded"] if self.rounded else [])
        if len(rounded_filled) > 0:
            head_info += f", style=\"{', '.join(rounded_filled)}\", color=\"black\""
        if self.rounded:
            head_info += ", fontname=helvetica"
        head_info += "] ;\n"
        # Specify graph & edge aesthetics
        if self.rounded:
            head_info += "\tedge [fontname=helvetica] ;\n"
        self.dot_text += head_info

    def _add_node_info(self, node_id, label_="", fillcolor_=WHITE_hex, fontcolor_=BLACK_hex):
        self.dot_text += f'\t{node_id} [label=<{label_}>, fontcolor="{fontcolor_}", fillcolor="{fillcolor_}"] ;\n'

    def _add_par_chil_info(self, son_id, par_id, arrow_info_=""):
        self.dot_text += f'\t{par_id} -> {son_id} {arrow_info_} ;\n'

    def _add_tail(self):
        self.dot_text += "}"

    def _initialize(self):
        self.node_id = 0
        self.dot_text = ""

    def recurse(self, node, par_node_id=None):
        {
            "lr"   : self._recurse_lr,
            "children" : self._recurse_children,
        }[self.tree_structure](node=node, par_node_id=par_node_id)

    def _recurse_lr(self, node, par_node_id=None):
        my_node_id = self.node_id
        self.node_id += 1
        self._add_node_info(node=node, node_id=my_node_id)
        if par_node_id is not None:
            self._add_par_chil_info(son_id=my_node_id, par_id=par_node_id)
        if node.left is not None:
            self._recurse_lr(node.left, my_node_id)
        if node.right is not None:
            self._recurse_lr(node.right, my_node_id)

    def _recurse_children(self, node, par_node_id=None):
        my_node_id = self.node_id
        self.node_id += 1
        self._add_node_info(node=node, node_id=my_node_id)
        if par_node_id is not None:
            self._add_par_chil_info(son_id=my_node_id, par_id=par_node_id)

        children = node.children
        if isinstance(children, list):
            for child in children:
                self._recurse_children(child, my_node_id)
        elif isinstance(children, dict):
            for child in children.values():
                self._recurse_children(child, my_node_id)
        else:
            handleTypeError(
                types=[list, dict], children=children,
                msg_="Please check the type of `nodel.children`"
            )


    def export(self, node, out_file=None):
        self._initialize()
        self._add_head()
        self.recurse(node, par_node_id=None)
        self._add_tail()
        if out_file is not None:
            with open(out_file, mode="w", encoding="utf-8") as f:
                f.write(self.dot_text)
            print(f"{out_file} was created.")
        else:
            return self.dot_text

    def write_png(self, node, path):
        try:
            import pydotplus
        except:
            raise ImportError("You Need to run the following command\
                              \n`$ pip install -U pydotplus`")
        if path[-4:] != ".png":
            path += ".png"
        dot_data = self.export(node, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        return graph.write_png(path, f='png', prog='dot')

class DecisionTreeDOTexporter(BaseTreeDOTexporter):
    def __init__(self, feature_names, class_names, cmap="jet",
                 filled=True, rounded=True, precision=3):
        super().__init__(feature_names=feature_names, class_names=class_names,
                         cmap=cmap, filled=filled, rounded=rounded,
                         precision=precision, tree_structure="lr")
        self.headlabeled = False

    def _initialize(self):
        self.headlabeled = False
        super()._initialize()

    def _add_node_info(self, node, node_id):
        precision = self.precision
        class_names = self.class_names
        # label info.
        label = ""
        if not (node.feature == None or node.depth == node.max_depth):
            label += f"{self.feature_names[node.feature]} &le; {round(node.threshold, precision)}<br/>"
        label += f"{node.criterion} = {round(node.impurity, precision)}<br/>" \
               + f"N = {node.num_samples}<br/>" \
               + f"classes = {node.num_classes}<br/>" \
               + f"class = {class_names[node.label]}"
        # fill color info.
        fillcolor = self.fill_color_dict.get(class_names[node.label], WHITE_hex)
        # font color info.
        fontcolor = self.font_color_dict.get(class_names[node.label], BLACK_hex)
        super()._add_node_info(node_id=node_id, label_=label, fillcolor_=fillcolor, fontcolor_=fontcolor)

    def _add_par_chil_info(self, son_id, par_id):
        arrow_info_ = ""
        if par_id==0:
            deg = "" if self.headlabeled else "-"
            arrow_info_ += f'[labeldistance=2.5, labelangle={deg}45, headlabel="{not self.headlabeled}"]'
            self.headlabeled = not self.headlabeled
        super()._add_par_chil_info(son_id=son_id, par_id=par_id, arrow_info_=arrow_info_)

class NaiveTrieDOTexporter(BaseTreeDOTexporter):
    def __init__(self, filled=True, rounded=True, precision=3):
        super().__init__(feature_names=None, class_names=None,
                         filled=filled, rounded=rounded, precision=precision,
                         tree_structure="children")

    def _add_node_info(self, node, node_id):
        label = node.value
        super()._add_node_info(node_id=node_id, label_=label)

class ItemsetTreeDOTexporter(BaseTreeDOTexporter):
    def __init__(self, class_names, cmap="jet",
                 filled=True, rounded=True, precision=3):
        super().__init__(feature_names=None, class_names=class_names,
                         cmap=cmap, filled=filled, rounded=rounded,
                         precision=precision, tree_structure="children")

    def _add_node_info(self, node, node_id):
        # label info.
        label = ""
        label += f"frequency = {node.freq}<br/>" \
               + f"itemset = {[self.class_names[idx] for idx in node.itemset]}<br/>"
        super()._add_node_info(node_id=node_id, label_=label)

    # def _add_par_chil_info(self, son_id, par_id):
    #     arrow_info_ = ""
    #     if par_id==0:
    #         deg = "" if self.headlabeled else "-"
    #         arrow_info_ += f'[labeldistance=2.5, labelangle={deg}45, headlabel="{not self.headlabeled}"]'
    #         self.headlabeled = not self.headlabeled
    #     super()._add_par_chil_info(son_id=son_id, par_id=par_id, arrow_info_=arrow_info_)
