import numpy as np
import matplotlib.pyplot as plt
# IPython magic function to store 
# plot outputs within the notebook
# %matplotlib inline
import tensorflow as tf
learn = tf.contrib.learn

tf.logging.set_verbosity(tf.logging.ERROR)

# Import the dataset
mnist = learn.datasets.load_dataset('mnist')

# 55,000 training images
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)

# 10,000 test images
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Use the following code to limit the samples to experiment faster
# max_examples = 10000
# data = data[:max_examples]
# labels = labels[:max_examples]

# Display some images with their labels
def display(i):
    img = test_data[i]
    plt.title('Example %d, Label %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)
    plt.show()

# display(0)

print("Length of one image:", len(data[0]))

# Fit a linear classifier
feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(
    feature_columns=feature_columns, 
    n_classes=10)

classifier.fit(data, labels, batch_size=100, steps=1000)

# Evaluate accuracy
classifier.evaluate(test_data, test_labels)
print("Classifier Accuracy:",
    classifier.evaluate(test_data, test_labels)["accuracy"])

# Classify a few examples
# We can make predictions on individual images using predict method.
# Here's one it gets right
prediction = classifier.predict(np.array([test_data[0]], dtype=float), as_iterable=False)
print("Predicted %d, Label: %d" % (prediction, test_labels[0]))
display(0)

# And one it gets wrong
prediction = classifier.predict(np.array([test_data[8]], dtype=float), as_iterable=False)
print("Predicted %d, Label: %d" % (prediction, test_labels[8]))
display(8)