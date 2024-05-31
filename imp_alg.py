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

# =============================================================================
# X=ar([1,2,0,6,1,0,0,5,1,7],5,2)
# Y=ar([1,2,3,4,5],5,1)
# X_hat=ar([1,3],c=2)
# =============================================================================

# =============================================================================
# A=ar([1,2,5,6,3,5,4,3,2,1,1,1,3,2,1,3,3,3,2,1,6,7,9,8,4],5,5)
# =============================================================================

# =============================================================================
# def chksum(A,B,C):
#     return (7+(4*A)+(3*B)+(2*C))%9
# A=ar([1,2,3,2,1,3],2,3)
# =============================================================================
# =============================================================================
# X=ar([-1,1,0,0,0,2,1,1],4,2)
# Y=ar([0,1,1,0,1,0,0,1],4,2)
# P=poly(X,2)
# =============================================================================

def mse1(Y):
    mean=sum(Y)/len(Y)
    mseo=0
    for y in Y:
        mseo+=(y-mean)**2
    return mseo/len(Y)

# =============================================================================
# y1=[2.7,4.9,2.4,6.8]
# y2=[7.4,9.4,9.9,12.6,14.2,16.8]
# 
# def sig(x1,x2):
#     a=-30+20*x1+20*x2
#     return 1/(1+np.exp(-a))
# 
# =============================================================================

# =============================================================================
# yh=[9, 11, 23, 6, 8, 12, 10, 4, 13, 7]
# =============================================================================

X=ar([1,2,3,4,0,6,1,1,0,0,1,2],4,3)
Y=ar([1,0,0,1,0,0,0,1,0,0,0,1],4,3)
X_hat=ar([1,0,1],1,3)
yh=pred(X,Y,X_hat,2)

# =============================================================================
# def dev(w):
#     return 2*np.sin(w)*np.cos(w)*np.exp(w)
# 
# def euc(Y,c):
#     mseo=[]
#     for y in Y:
#         mseo.append((y-c)**2)
#     return mseo
# Y=[50,60,66,70,72,76,82,90,98]
# y1=[50,60,66,70,72,76]
# y2=[82,90,98]
# =============================================================================

# =============================================================================
# def mse2(Y,y):
#     mseo=0
#     for yh in Y:
#         mseo+=(yh-y)**2
#     return mseo
# 
# Y=[6, 8, 9, 5, 10, 5, 4, 8, 9, 3]
# =============================================================================
