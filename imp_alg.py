# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:23:38 2020

@author: Chathura
"""

import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg import inv,det
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import entropy

# Array shaping functions
def ar(X,r=1,c=1):
    return np.array(X).reshape(r,c)

def poly(X,order):
    conv=PolynomialFeatures(order)
    return conv.fit_transform(X)

#Regression functions
def over(X):
    return inv(X.T @ X) @ X.T
    
def under(X):
    return X.T @ inv(X @ X.T)

def ridge(X,lam):
    return inv((X.T @ X) + lam*np.eye(np.shape(X)[1])) @ X.T

def pred(X,Y,X_hat,p=0,lam=0):
    if p>0:
        X=poly(X,p)
        X_hat=poly(X_hat,p)
    
    if lam!=0:
        w=ridge(X,lam) @ Y
        return X_hat @ w
    
    shape=np.shape(X)
    if shape[0]>=shape[1]:
        w=over(X) @ Y
    else:
        w=under(X) @ Y
    return X_hat @ w

def regr(X,Y,p=0,lam=0):
    if p>0:
        X=poly(X,p)
    
    if lam!=0:
        w=ridge(X,lam) @ Y
        return w
    
    shape=np.shape(X)
    if shape[0]>shape[1]:
        w=over(X) @ Y
    else:
        w=under(X) @ Y
    return w

#Impurity calculation functions
def entr(X):
    return entropy(X,base=2)
def gini(X):
    return 1-sum(map(lambda x: x**2,X))
def mis(X):
    return 1-max(X)
def imp(X_train,func):
    return sum([sum(X)*func(X) for X in X_train])

#Bias variance
def bi(X,X_hat,w_real,lam):
    return -1*lam*X_hat @ inv((X.T @ X) + lam*np.eye(np.shape(X)[1])) @ w_real
def va(X,X_hat,w_real,noice_var,lam=0):
    block=inv((X.T @ X) + lam*np.eye(np.shape(X)[1]))
    return X_hat @ (block - lam* (block**2)) @ X_hat.T * noice_var 

def mse1(Y):
    mean=sum(Y)/len(Y)
    mseo=0
    for y in Y:
        mseo+=(y-mean)**2
    return mseo/len(Y)


X=ar([1,2,3,4,0,6,1,1,0,0,1,2],4,3)
Y=ar([1,0,0,1,0,0,0,1,0,0,0,1],4,3)
X_hat=ar([1,0,1],1,3)
yh=pred(X,Y,X_hat,2)


