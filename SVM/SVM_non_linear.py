import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.io import loadmat
from sklearn import svm

raw_data = loadmat('data/ex6data3.mat')

X = raw_data['X']  
Xval = raw_data['Xval']  
y = raw_data['y'].ravel()  
yval = raw_data['yval'].ravel()

# grid search
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]  
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0  
best_params = {'C': None, 'gamma': None}

for C in C_values:  
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)

        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

print(best_score, best_params)

# Spam Classification

spam_train = loadmat('data/spamTrain.mat')  
spam_test = loadmat('data/spamTest.mat')
X = spam_train['X']  
Xtest = spam_test['Xtest']  
y = spam_train['y'].ravel()  
ytest = spam_test['ytest'].ravel()
svc = svm.SVC()  
svc.fit(X, y)  
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))  
