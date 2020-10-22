#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:18:22 2020

@author: jingqiu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:18:09 2020

@author: jingqiu
"""

import algorithms as algo
import model
import numpy as np
import numpy.linalg as lin


def exp4():
    lamda=1.5
    noise='mixed'
    prior='Gaussian';
    size=200;
    parameters=[0.1,0.3,0.5,2,5,8]
    order=2;#we only consider matrix here
    iterations=10
    for parameter in parameters:
        correlation1,correlation2,correlation3,correlation4=exp4single(iterations,prior,size,lamda/np.sqrt(size),noise,parameter)
        np.savetxt(noise+str(parameter)+'naive'+str(lamda)+'.'+str(size)+'lout', correlation1, delimiter=',')   # X is an array
        np.savetxt(noise+str(parameter)+'TruncateDeaverage'+str(lamda)+'.'+str(size)+'lout',correlation2,delimiter=',')
        np.savetxt(noise+str(parameter)+'NonBackTrackingAlgorithm'+str(lamda)+'.'+str(size)+'lout',correlation3,delimiter=',')
        np.savetxt(noise+str(parameter)+'SelfAvoiding'+str(lamda)+'.'+str(size)+'out',correlation4,delimiter=',')
    
    

def exp4single(iterations,prior,size,lamda,noise,parameter):
    correlation1=[];
    correlation2=[];
    correlation3=[];
    correlation4=[];
    for iteration in range(iterations):
        x,Y=model.spikedTensorModel(lamda,prior,noise,size,2,parameter)
        hatx1=algo.naiveSpectralAlgorithm(Y);
        hatx2=algo.truncationDeaverageAlgorithm(Y,5);#threshold at 10
        hatx3=hatx2
        hatx3=algo.NonBackTrackingAlgorithm(Y,int(2*np.log(size)),1);
        
        hatx4=algo.GeneralSpectralAlgorithm(Y,int(1.8*np.log(size)))
        #hatx4=hatx3;
        correlation1.append(hatx1.dot(x)/lin.norm(x));
        correlation2.append(hatx2.dot(x)/lin.norm(x));
        correlation3.append(hatx3.dot(x)/lin.norm(x))
        correlation4.append(hatx4.dot(x)/lin.norm(x))
        print(correlation1)
        print(correlation2)
        print(correlation3)
        print(correlation4)
        print(np.sqrt(1-(1/(size*(lamda**2)))))
    return correlation1,correlation2,correlation3,correlation4