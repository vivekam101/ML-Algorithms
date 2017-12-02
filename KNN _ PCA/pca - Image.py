import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.io import loadmat


faces = loadmat('data/ex7faces.mat')  
X = faces['X'] 

face = np.reshape(X[3,:], (32, 32))  
plt.imshow(face)
plt.show()

def pca(X):  
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V

U, S, V = pca(X)  

def project_data(X, U, k):  
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)

def recover_data(Z, U, k):  
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

U, S, V = pca(X)  
Z = project_data(X, U, 100)  

X_recovered = recover_data(Z, U, 100)  
face = np.reshape(X_recovered[3,:], (32, 32))  
plt.imshow(face)
plt.show()
