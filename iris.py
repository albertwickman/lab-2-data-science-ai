import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

iris = load_iris()
print(iris)

data = iris.data
target = iris.target
classifications = iris.target_names

# Splits the data into a training set and a validation set, is later used in the classifier
dataTrain, dataTest, targetTrain, targetTest = train_test_split(data, target, random_state=0)

# Model that is too regularized (C too low) to see the impact on the results
classifier = svm.SVC(kernel="linear", C=0.01).fit(dataTrain, targetTrain)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix", None)
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        dataTest,
        targetTest,
        display_labels=classifications,
        cmap=plt.cm.Blues,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()