#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 07:41:00 2020

@author: jingqiu
"""

import matplotlib.pyplot as plt

def fig4plot(lamda, parameters, correlations1,correlations2,correlations3,correlations4):
    naive,=plt.semilogx(parameters,correlations1,'r-*',label='naive',linewidth=2.0)
    #saw,= plt.plot(parameters,correlations3,'-^',label='truncate')
    backtracking,=plt.plot(parameters,correlations2,'b-D',label='NBW')
    saw,=plt.plot(parameters,correlations3,'y-o',label='SAW')
    worst,=plt.plot(parameters, [1-1/lamda**2 for parameter in parameters],'-',label='worst',color='grey')
    #worst,=plt.plot(lamdas,[1-1/lamda**2 for lamda in lamdas],label='worst',color='grey')
    truncate,= plt.plot(parameters,correlations4,'g-^',label='truncate')
    plt.legend(handles=[naive,truncate,backtracking,saw,worst],loc='best',frameon=False);
    plt.xlabel(r'signal strength $\lambda$')
    plt.xlabel(r'SNR $\lambda$')
    plt.xlabel(r'parameter $d$')
    plt.ylabel('Squared correlation')
    #plt.savefig('MixedExperiment200parameters2.eps')
    plt.savefig('mixedExperiment200lamda15.eps')
    plt.close()