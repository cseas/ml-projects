# This is a scrappy version of the K Nearest Neighbors classifier, 
# one of the simplest classifiers around.
# Created by Abhijeet Singh

# custom classifier
from scipy.spatial import distance

# find euclidian distance between two points
def euc(a, b):
    # a is a point from our training data
    # b is a point from our testing data
    return distance.euclidean(a, b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        # distance from closest training point
        best_dist = euc(row, self.X_train[0])
        # index of closest training point
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])

            if dist < best_dist:
                best_dist = dist
                best_index = i
                
        return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))