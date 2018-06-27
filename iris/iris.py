import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# load iris dataset
iris = load_iris()

# print out the entire dataset
for i in range(len(iris.target)):
    print("Example %d: label: %s features %s" % (i, iris.target[i], iris.data[i]))

# print features and labels from dataset
print("\nFeatures\n", iris.feature_names)
print("\nLabels\n", iris.target_names)

# remove one example from each type of flower from testing
# first setosa is at 0, first versicolor is at 50, and so on
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# 0 = setosa, 1 = versicolor, 2 = virginica
print("\nActual values", test_target)
print("Predicted values", clf.predict(test_data))

# visualise the decision tree
import graphviz

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)

graph = graphviz.Source(dot_data) 
graph.render("iris")