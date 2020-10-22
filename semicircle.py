#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:30:34 2020

@author: jingqiu
"""

import algorithms as algo
import model
import numpy as np
import numpy.linalg as lin
from scipy.sparse import rand

size=2000;
for parameter in [1,5,10]:
    
    M=model.RandomTensor('ErdosRenyi', size,2,parameter)
    #for i in range(size):
    #    for j in range(size):
    #        if(i>j):
    #            M[j,i]=M[i,j]
    
    print('norm')
    x=np.random.normal(0,1,size);
    lamda=2/np.sqrt(size);
    u,s,v=lin.svd(M)
    u,s2,v=lin.svd(M+lamda*np.outer(x,x))
    #s,v=lin.eigh((M+M.T)/np.sqrt(2));
    #s2,v=lin.eigh((M+M.T)/np.sqrt(2)+lamda*np.outer(x,x));
    #print(x)
    print(np.inner(u[:,0],x)/lin.norm(x))
    #print(np.inner(u[:,-1],x)/lin.norm(x))
    input()
    #s,w=lin.eigh((M+M.T)/np.sqrt(2))
    print(s[:10]/np.sqrt(size))
    print(s2[:10]/np.sqrt(size))
    #print(s[1]/np.sqrt(size))
    #print(s[-1]/np.sqrt(size))
    input()