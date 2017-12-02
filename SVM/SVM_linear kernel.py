import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.io import loadmat  

raw_data = loadmat('data/ex6data1.mat')  

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])  
data['y'] = raw_data['y']

positive = data[data['y'].isin([1])]  
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')  
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')  
ax.legend()  
plt.show()

from sklearn import svm  
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(data[['X1', 'X2']], data['y'])  
print(svc.score(data[['X1', 'X2']], data['y']))

# Regularized C=100

from sklearn import svm  
svc = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc.fit(data[['X1', 'X2']], data['y'])  
print(svc.score(data[['X1', 'X2']], data['y']))

