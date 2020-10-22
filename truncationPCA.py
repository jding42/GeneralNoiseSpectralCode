#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:21:11 2020

@author: jingqiu
"""
import numpy as np
import numpy.random as random
import numpy.linalg as lin
import scipy.linalg as sciLin
import scipy.special as special
import itertools
import numpy.polynomial.polynomial as polynomial
from scipy.sparse import csr_matrix

def truncationDeaverageAlgorithm(tensor,tau):
    
    '''thresholding at tau then centerize, finally transfer to naive algorithm'''
    shapes=np.shape(tensor)
    order=len(shapes)
    n=shapes[0]
    if(order==2):
        tensor=np.minimum(tensor,np.ones(shapes)*tau);
        tensor=np.maximum(tensor,-np.ones(shapes)*tau);
        tensor-=np.mean(tensor)
        #w,v=sciLin.eigh((tensor+tensor.T)/2)
        #print(w/np.sqrt(n))
        #input()
        return naiveSpectralAlgorithm(tensor);

def truncationSpectralAlgorithm(tensor,tau):
    ''' only threshold but not centerize'''
    
    shapes=np.shape(tensor);
    order=len(shapes);
    n=shapes[0];
    if(order==2):
        tensor=np.minimum(tensor,np.ones(shapes)*tau);
        tensor=np.maximum(tensor,-np.ones(shapes)*tau);
        #print(tensor)
        #print(naiveSpectralAlgorithm(tensor))
        return naiveSpectralAlgorithm(tensor);
        

def naiveSpectralAlgorithm(tensor):
    '''simply take leading eigenvector from the symmetrized matrix'''
    shapes=np.shape(tensor);
    order=len(shapes);
    n=shapes[0];
    if(order==2):
        w,v=sciLin.eigh((tensor+tensor.T)/2,eigvals=(n-1,n-1));
        return v[:,0];
    elif (order==3):
        return 'not implemented yet'