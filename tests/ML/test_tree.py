# coding: utf-8
import os
from kerasy.ML.tree import DecisionTreeClassifier
from kerasy.utils import cluster_accuracy
from kerasy.utils import generateWholeCakes

num_samples = 200
num_classes = 5
max_depths = [1,2,4,8]

def get_test_data():
    x_train, y_train = generateWholeCakes(num_classes=num_classes,
                                          num_samples=num_samples,
                                          r_low=1,
                                          r_high=5,
                                          same=False,
                                          seed=0)
    return x_train, y_train

def test_decision_tree(target=0.75, path="decision_tree.png"):
    x_train, y_train = get_test_data()

    for i,max_depth in enumerate(sorted(max_depths)):
        model = DecisionTreeClassifier(criterion="gini",
                                       max_depth=max_depth,
                                       random_state=0)

        model.fit(x_train,y_train)
        y_pred = model.predict(x_train)
        score = cluster_accuracy(y_pred, y_train)

        assert i==0 or prev_score <= score
        prev_score = score

        assert model.export_graphviz(out_file=path)
        os.remove(path)

    assert score >= target
