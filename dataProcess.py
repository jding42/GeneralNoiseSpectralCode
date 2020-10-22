#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:52:32 2020

@author: jingqiu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from fig1plot import fig1plot 
from fig2plot import fig2plot
from fig3plot import fig3plot
from fig4plot import fig4plot

#parameters=[0.05,0.5,3.5]
rcParams.update({'font.size': 13})
correlations1=[]
correlations2=[]
correlations3=[]
correlations4=[]
lamda=1.1

# for large matrix, the performance of non-backtracking walk in the very sparse case
size=400
noise='MinusErdosRenyi'
parameters=[0.05,0.1,0.5,0.8,2.0,5.0]
#correlation1=np.loadtxt(noise+str(parameter)+'naive'+str(lamda)+'.'+str(size)+'lout', delimiter=',')   # X is an array
# correlation2=np.loadtxt(noise+str(parameter)+'TruncateDeaverage'+str(lamda)+'.'+str(size)+'lout',delimiter=',')
# correlation3=np.loadtxt(noise+'NonBackTrackingAlgorithm'+str(lamda)+'.'+str(size)+'400lout',delimiter=',')
# print(np.mean(correlation1**2))
# print(np.mean(correlation2**2))
# print(np.mean(correlation3**2))

for parameter in parameters:
    correlations1.append(np.mean(np.loadtxt(noise+str(parameter)+'naive'+str(lamda)+'.'+str(size)+'lout', delimiter=',')**2))  # X is an array
    correlations2.append(np.mean(np.loadtxt(noise+str(parameter)+'TruncateDeaverage'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
    correlations3.append(np.mean(np.loadtxt(noise+str(parameter)+'NonBackTrackingAlgorithm'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
    #correlations4.append(np.mean(np.loadtxt(noise+str(parameter)+'SelfAvoiding'+str(lamda)+'.'+str(size)+'out',delimiter=',')**2)) 

fig1plot(lamda,parameters, correlations1,correlations2,correlations3)


size=2000
lamdas=[1.2,1.4,1.5,1.7,2.1,2.5];
correlations1=[]
correlations2=[]
correlations3=[]
correlations4=[]
noise='nonIdentical'
parameter=30.0
for lamda in lamdas:
    correlations1.append(np.mean(np.loadtxt(noise+'naive'+str(lamda)+'.'+str(size)+'lout', delimiter=',')**2))  # X is an array
    correlations2.append(np.mean(np.loadtxt(noise+'TruncateDeaverage'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
    #correlations3.append(np.mean(np.loadtxt(noise+str(parameter)+'NonBackTrackingAlgorithm'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
    #correlations4.append(np.mean(np.loadtxt(noise+str(parameter)+'SelfAvoiding'+str(lamda)+'.'+str(size)+'out',delimiter=',')**2)) 

fig2plot(lamdas,parameter, correlations1,correlations2)


size=200
lamda=1.5
parameters=[0.1,0.3,0.5,2,5,8]
correlations1=[]
correlations2=[]
correlations3=[]
correlations4=[]
noise='NonIdentical'
for parameter in parameters:
    correlations1.append(np.mean(np.loadtxt(noise+str(parameter)+'naive'+str(lamda)+'.'+str(size)+'lout', delimiter=',')**2))  # X is an array
    #correlations2.append(np.mean(np.loadtxt(noise+str(parameter)+'TruncateDeaverage'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
    correlations2.append(np.mean(np.loadtxt(noise+str(parameter)+'NonBackTrackingAlgorithm'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
    correlations3.append(np.mean(np.loadtxt(noise+str(parameter)+'SelfAvoiding'+str(lamda)+'.'+str(size)+'out',delimiter=',')**2))
    correlations4.append(np.mean(np.loadtxt(noise+str(parameter)+'TruncateDeaverage'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
fig3plot(lamda,parameters,correlations1,correlations2,correlations3,correlations4);


size=200
lamda=1.5
parameters=[0.1,0.3,0.5,2,5,8]
correlations1=[]
correlations2=[]
correlations3=[]
correlations4=[]
noise='mixed'
for parameter in parameters:
    correlations1.append(np.mean(np.loadtxt(noise+str(parameter)+'naive'+str(lamda)+'.'+str(size)+'lout', delimiter=',')**2))  # X is an array
    #correlations2.append(np.mean(np.loadtxt(noise+str(parameter)+'TruncateDeaverage'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
    correlations2.append(np.mean(np.loadtxt(noise+str(parameter)+'NonBackTrackingAlgorithm'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
    correlations3.append(np.mean(np.loadtxt(noise+str(parameter)+'SelfAvoiding'+str(lamda)+'.'+str(size)+'out',delimiter=',')**2))
    correlations4.append(np.mean(np.loadtxt(noise+str(parameter)+'TruncateDeaverage'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
fig4plot(lamda,parameters,correlations1,correlations2,correlations3,correlations4);



#load data
#noise='NonIdentical'
# noise='mixed'
# size=200
# parameters=[0.1,0.3,0.5,2,5,8]
# #lamdas=[1.2,1.3,1.4,1.5,1.6]
# lamda=1.5;
# for parameter in parameters:
#     correlations1.append(np.mean(np.loadtxt(noise+str(parameter)+'naive'+str(lamda)+'.'+str(size)+'lout', delimiter=',')**2))  # X is an array
#     correlations2.append(np.mean(np.loadtxt(noise+str(parameter)+'TruncateDeaverage'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
#     correlations3.append(np.mean(np.loadtxt(noise+str(parameter)+'NonBackTrackingAlgorithm'+str(lamda)+'.'+str(size)+'lout',delimiter=',')**2))
#     correlations4.append(np.mean(np.loadtxt(noise+str(parameter)+'SelfAvoiding'+str(lamda)+'.'+str(size)+'out',delimiter=',')**2))
# print(correlations1)
# print(correlations2)  
# #print(correlations3)


     
# plot correlations from 4 different algorithms   
# naive,=plt.semilogx(parameters,correlations1,'-*',label='naive',linewidth=2.0)
# truncate,= plt.plot
# backtracking,=plt.plot(parameters,correlations3,'-D',label='backtracking')
# saw,=plt.plot(parameters,correlations4,'-ro',label='SAW')
# worst,=plt.plot(parameters, [1-1/lamda**2 for parameter in parameters],'-',label='worst',color='grey')
# #worst,=plt.plot(lamdas,[1-1/lamda**2 for lamda in lamdas],label='worst',color='grey')
# plt.legend(handles=[naive,backtracking,saw,worst],loc='best',frameon=False);
# plt.xlabel(r'signal strength $\lambda$')
# plt.xlabel(r'SNR $\lambda$')
# plt.xlabel(r'parameter $d$')
# plt.ylabel('Squared correlation')
# #plt.savefig('MixedExperiment200parameters2.eps')
# plt.savefig('MinusERExperiment400lamda11.eps')




