from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay

dataset = load_iris()
target = dataset.target
classifications = dataset.target_names

n_neighbors = 71

data = dataset.data[:, :2]
data_target = dataset.target

cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

for weights in ["uniform", "distance"]:
    classification = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    classification.fit(data, data_target)

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        classification,
        data,
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel=dataset.feature_names[0],
        ylabel=dataset.feature_names[1],
        shading="auto",
    )

    sns.scatterplot(
        x=data[:, 0],
        y=data[:, 1],
        hue=dataset.target_names[data_target],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.title(
        "Classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )


plt.show()