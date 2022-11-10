import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

cv_scores = []
neighbors = list(np.arange(3, 50, 2))
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n, algorithm='brute')

    cross_val = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(cross_val.mean())

error = [1 - x for x in cv_scores]
optimal_n = neighbors[error.index(min(error))]
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_n, algorithm='brute')
knn_optimal.fit(x_train, y_train)
pred = knn_optimal.predict(x_test)
acc = accuracy_score(y_test, pred) * 100
print("The accuracy for optimal k = {0} using brute is {1}".format(optimal_n, acc))