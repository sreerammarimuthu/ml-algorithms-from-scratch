import math
import numpy as np

# Forward Pass 
def compute_z(x,w,b):
    z = np.dot(w.flatten(), x) + b
    return z 

def compute_a(z):  
    e = np.exp(-np.clip(z, -700, 700))
    a = float(1 / (1 + e))
    return a

def compute_L(a,y):
    if a==y:
        L = 0.0
    elif a==0 or a==1:
        L=1e15
    else:     
        L = -(y*np.log(a)+(1 - y)*np.log(1 - a))
    return float(L) 


def forward(x,y,w,b):
    z = compute_z(x,w,b)
    a = compute_a(z)
    L = compute_L(a,y)
    return z, a, L 


# Compute Local Gradients
def compute_dL_da(a, y):
    ep = 1e-15 
    dL_da = -(y/(a + ep)) + ((1 - y)/(1- a + ep))
    return dL_da 

def compute_da_dz(a):
    if np.isclose(a, 0):
        return 1e-15
    da_dz = a*(1 - a)
    return da_dz 

def compute_dz_dw(x):
    dz_dw = x
    return dz_dw

def compute_dz_db():
    dz_db = 1.0    
    return dz_db


# Back Propagation 
def backward(x,y,a):
    dL_da = compute_dL_da(a, y)
    da_dz = compute_da_dz(a)
    dz_dw = compute_dz_dw(x)
    dz_db = compute_dz_db()
    return dL_da, da_dz, dz_dw, dz_db 

def compute_dL_dw(dL_da, da_dz, dz_dw):
    dL_dw = dL_da * da_dz * dz_dw
    return dL_dw

def compute_dL_db(dL_da, da_dz, dz_db):
    dL_db = dL_da * da_dz * dz_db
    return dL_db 


# gradient descent 
def update_w(w, dL_dw, alpha=0.001):
    w = w - alpha*dL_dw
    return w

def update_b(b, dL_db, alpha=0.001):
    b = b - alpha*dL_db
    return  b 

def train(X, Y, alpha=0.001, n_epoch=100):
    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(n_epoch):
        for x,y in zip(X,Y):
            x = x.T
            z, a, L = forward(x, y, w, b)

            dL_da, da_dz, dz_dw, dz_db = backward(x, y, a)
            dL_dw = compute_dL_dw(dL_da, da_dz, dz_dw)
            dL_db = compute_dL_db(dL_da, da_dz, dz_db)

            w = update_w(w, dL_dw, alpha)
            b = update_b(b, dL_db, alpha)

    return w, b

def predict(Xtest, w, b):
    n = Xtest.shape[0]
    Y = np.zeros(n)
    P = np.zeros((n, 1))

    for i, x in enumerate(Xtest):
        x = x.T
        z, a, L = forward(x, 0, w, b)
        if a >= 0.5:
            Y[i] = 1
        P[i] = a       
    return Y, P

# gradient checking 
def check_dL_da(a, y, delta=1e-10):
    dL_da = (compute_L(a+delta,y) - compute_L(a,y)) / delta
    return dL_da 

def check_da_dz(z, delta= 1e-7):
    da_dz = (compute_a(z+delta) - compute_a(z)) / delta
    return da_dz 

def check_dz_dw(x,w, b, delta=1e-7):
    p = x.shape[0] 
    dz_dw = np.zeros(p).reshape(-1, 1)
    for i in range(p):
        d = np.zeros(p).reshape(-1, 1)
        d[i] = delta
        dz_dw[i] = (compute_z(x,w+d, b) - compute_z(x, w, b)) / delta
    return dz_dw

def check_dz_db(x,w, b, delta=1e-7):
    dz_db = (compute_z(x, w, b+delta) - compute_z(x, w, b)) / delta
    return  dz_db

def check_dL_dw(x,y,w,b, delta=1e-7):
    p = x.shape[0]
    dL_dw = np.zeros(p).reshape(-1, 1)
    for i in range(p):
        d = np.zeros(p).reshape(-1, 1)
        d[i] = delta
        dL_dw[i] = (forward(x,y,w+d,b)[-1] - forward(x,y,w,b)[-1]) / delta
    return dL_dw

def check_dL_db(x,y,w,b, delta=1e-7):
    dL_db = (forward(x,y,w,b+delta)[-1] - forward(x,y,w,b)[-1]) / delta
    return dL_db
