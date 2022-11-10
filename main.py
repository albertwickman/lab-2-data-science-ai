import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd



# 1. a)
df = pd.read_csv('data_assignment2.csv')
area = 'Living_area'
price = 'Selling_price'

filterOutliers = df[df[price] > 2400000]

# Variables without filtering
# x = df[area]
# y = df[price]

x = filterOutliers[area]
y = filterOutliers[price]

linReg = LinearRegression(fit_intercept=True)
linReg.fit(x[:, np.newaxis], y)

xFit = np.linspace(50, 250, 200)
yFit = linReg.predict(xFit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xFit, yFit)
plt.show()

# 1. b)
slope = linReg.coef_
intercept = linReg.intercept_

print('Slope coefficient: ', slope)
print('Intercept: ', intercept)


# 1. c)
slope = linReg.coef_
intercept = linReg.intercept_

# y = kx + m, given a value x
def predictionFunc(x):
    return slope * x + intercept

print('Predicted price for 100 m^2: ', predictionFunc(100))
print('Predicted price for 150 m^2: ', predictionFunc(150))
print('Predicted price for 200 m^2: ', predictionFunc(200))

# 1. d)
import seaborn as sea

sea.residplot(x=x, y=y, data=filterOutliers)
plt.show()