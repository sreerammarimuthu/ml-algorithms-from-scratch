import math
import numpy as np

# Linear Regression based upon gradient descent
def compute_Phi(x,p):
    n = len(x)
    Phi = np.ones((n, 1))
    for i in range(1, p):
        Phi = np.column_stack((Phi, x ** i))
    return Phi 

def compute_yhat(Phi, w):
    yhat = np.dot(Phi, w)
    return yhat

def compute_L(yhat,y):
    n = len(y)
    L = 1/(2 * n) * np.sum((y - yhat)**2)
    return L 

def compute_dL_dw(y, yhat, Phi):
    n = len(y)
    dL_dw = -1/n * np.dot(Phi.T, (y - yhat))
    return dL_dw

def update_w(w, dL_dw, alpha = 0.001):
    w = w - alpha * dL_dw
    return w

def train(X, Y, alpha=0.001, n_epoch=100):
    w = np.array(np.zeros(X.shape[1])).T

    for _ in range(n_epoch):
        Y_pred = np.dot(X, w)
        gradient = -2 * np.dot(X.T, (Y - Y_pred)) / len(Y)
        
        w = w - alpha * gradient
    return w
