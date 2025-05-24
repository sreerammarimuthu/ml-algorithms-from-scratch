import math
import numpy as np
from collections import Counter

# Decision Tree (with Discrete Attributes) using ID3 (Iterative Dichotomiser 3) algorithm. 
class Node:

    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

class Tree(object):

    @staticmethod
    def entropy(Y):
        outcomes = Counter(Y)
        numy = len(Y)
        e = -sum((i/numy) * np.log2(i/numy)for i in outcomes.values())
        return e 
    
    @staticmethod
    def conditional_entropy(Y,X):
        P_xy=Counter(zip(X,Y))
        P_x=Counter(X)
        ce=0
        
        for (i,j), k in P_xy.items():
            Pxy= k/len(X)
            Px=P_x[i]/len(X)
            ce-=Pxy*math.log2(Pxy/Px)
        return ce

    @staticmethod
    def information_gain(Y,X):       
        ent=Tree.entropy(Y)
        cond_ent=Tree.conditional_entropy(Y,X)       
        g=ent-cond_ent   
        return g

    @staticmethod
    def best_attribute(X,Y):
        information_gains = [Tree.information_gain(Y, X[n, :]) for n in range(X.shape[0])]
        i = np.argmax(information_gains)        
        return i

    @staticmethod
    def split(X,Y,i):
        C={}
        values=np.unique(X[i,:])
        
        for a in values:
            temp=X[i,:]==a
            Xc=X[:,temp]
            Yc=Y[temp]
            
            cn=Node(Xc,Yc)
            C[a]=cn
        return C

    @staticmethod
    def stop1(Y):
        s = len(set(Y)) == 1
        return s
    
    @staticmethod
    def stop2(X):
        if (X == X[:, [0]]).all():
            s=True
        else:
            s=False        
        return s
    
    @staticmethod
    def most_common(Y):
        if len(Y) == 0:
            return 'unknown' 
        c=Counter(Y)
        y=max(c,key=c.get)        
        return y
    
    @staticmethod
    def build_tree(t):
        if Tree.stop1(t.Y):
            t.isleaf = True
            t.p = Tree.most_common(t.Y)
        elif Tree.stop2(t.X):
            t.isleaf = True
            t.p = Tree.most_common(t.Y)
        else:
            i = Tree.best_attribute(t.X, t.Y)
            t.i = i
            t.C = Tree.split(t.X, t.Y, i)
            
            for a, cn in t.C.items():
                Tree.build_tree(cn)
        
        if not t.isleaf and t.p is None:
            t.p = Tree.most_common(t.Y)
        
    @staticmethod
    def train(X, Y):
        t = Node(X, Y)
        Tree.build_tree(t)
        return t
    
    @staticmethod
    def inference(t,x):
        if t.isleaf:
            return np.array([t.p])
        
        if t.i is None or x[t.i] not in t.C:
            return np.array([t.p])
        
        cn = t.C[x[t.i]]
        return Tree.inference(cn, x)

    @staticmethod
    def predict(t,X):
        n = X.shape[1]
        Y = np.empty(n, dtype=object)

        for i in range(n):
            xi = X[:, i]
            yi = Tree.inference(t, xi)
            Y[i] = yi[0]
        return Y
    
    @staticmethod
    def load_dataset(filename = 'data1.csv'):       
        X = []
        Y = []
        with open(filename, 'r') as file:
            lines = file.readlines()

            header = lines[0].strip().split(',')
            num_attributes = len(header)

            for line in lines[1:]:
                values = line.strip().split(',')

                label = values[0]
                Y.append(label)

                features = values[1:]
                X.append(features)
        
        X = np.array(X).T
        Y = np.array(Y)        

        return X,Y
