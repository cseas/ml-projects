#!/usr/bin/python
# Created by Abhijeet Singh on 16-June-18
""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

# slice datasets to 1%
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

from sklearn.svm import SVC 
clf = SVC(C=10000.0, kernel='rbf')

t0 = time()
clf.fit(features_train, labels_train)
print("Training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("Prediction time:", round(time()-t0, 3), "s")

print("Accuracy", clf.score(features_test, labels_test))

# Number of Chris predictions
count = 0
for i in pred:
    if i == 1:
        count += 1
print("Chris (1) predictions:", count)
#########################################################


