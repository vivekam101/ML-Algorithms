import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

path = os.getcwd() + '\data\ex1data1.txt'  
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# normalising the fields
#data = (data - data.mean())/data.std()
#print (data.head())

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

X_train = X[20:]
X_test = X[:20]

y_train = y[20:]
y_test = y[:20]


# using sklearn library
model = linear_model.LinearRegression()  
model.fit(X_train, y_train)
price_pred = model.predict(X_test)

# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, price_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, price_pred))


# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, price_pred, color='blue', linewidth=1)

plt.xticks(())
plt.yticks(())

plt.show()



