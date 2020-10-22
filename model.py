#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:27:18 2020

@author: jingqiu
"""
import numpy as np
import numpy.random as nrandom


def symm(Y):
    n=np.shape(Y)[0]
    for i in range(n):
        for j in range(i):
            Y[i,j]=Y[j,i];
    return Y;

def RandomTensor(distribution, size,order,parameter=1.0):
    size_tuple=tuple([size]*order)
    if(distribution=='Gaussian'):
        return symm(nrandom.normal(0,1,size_tuple))
    if(distribution=='Sparse'):
        parameter=0.4;
        mu=parameter/size_tuple[0];
        return symm(nrandom.choice([0.0,np.sqrt(1.0/mu),-np.sqrt(1.0/mu)],size_tuple,p=[1.0-mu,mu/2,mu/2]))
    
    if(distribution=='ErdosRenyi'):
        #discrete, biased and singular distribution
        #parameter=0.05;
        values=[-parameter/size_tuple[0],1-parameter/size_tuple[0]]
        prob=[1-parameter/size_tuple[0],parameter/size_tuple[0]]

        scaling=np.sqrt((-values[0]*values[1]));
        values/=scaling;
        print(values)
        Y=nrandom.choice(values,size_tuple,p=prob)
        return symm(Y);

    if(distribution=='MinusErdosRenyi'):
        #for failing algorithm truncating but not deaveraging        
        return -RandomTensor('ErdosRenyi',size,order,parameter);
    
    if(distribution=='mixed'):
        order=2
        random1=RandomTensor('ErdosRenyi',size,order,parameter);
        random2=RandomTensor('Gaussian',size,order);
        for i in range(size):
            for j in range(i%2,size,2):
                random1[i,j]=random2[i,j]
        return symm(random1);
    
    if(distribution=='NonIdentical'):
        order=2
        random1=RandomTensor('MinusErdosRenyi',size,order,parameter);
        random2=RandomTensor('ErdosRenyi',size,order,parameter);       
        for i in range(size):
            for j in range((i+1)%2,size,2):
                random1[i,j]=random2[i,j]   
        return symm(random1);

def spikedTensorModel(lamda,prior,noise,size,order,parameter=1.0):
    if(prior=='Rademacher'):
        x=2*nrandom.binomial(1, 0.5,size)-1
    if(prior=='Gaussian'):
        x=np.random.normal(0,1,size);
    if(prior=='Sparse'):
        x=np.random.choice([0,-size**0.1,size**0.1],size,p=[1-1/(size**0.2),0.5/(size**0.2),0.5/(size**0.2)]);
    Y=1
    for i in range(order):
        Y=np.tensordot(Y,x,axes=0)
    return x,lamda*Y+RandomTensor(noise,size,order,parameter);


