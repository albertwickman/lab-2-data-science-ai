import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score



dataset = load_iris()
data = dataset.data
target = dataset.target
classifications = dataset.target_names

# Splits the data into a training set and a validation set, is later used in the classifier
dataTrain, dataTest, targetTrain, targetTest = train_test_split(data, target, random_state=0)

# Model that is too regularized (C too low) to see the impact on the results
classifier = svm.SVC(kernel="linear", C=100).fit(dataTrain, targetTrain)
np.set_printoptions(precision=2)

# Plot non-normalized and normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Confusion matrix, with normalization", "true"),
]

for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        dataTest,
        targetTest,
        display_labels=classifications,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
#plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Testing k neighbors from 1 to 100
for i in range(1, 100):

    kNeighbors = KNeighborsClassifier(n_neighbors=i)

    #Train the model using the training sets
    kNeighbors.fit(dataTrain, targetTrain)

    #Predict the response for test dataset
    targetPred = kNeighbors.predict(dataTest)
    print("Accuracy in percentage, given", i, "neighbors:", metrics.accuracy_score(targetTest, targetPred) * 100, "%")