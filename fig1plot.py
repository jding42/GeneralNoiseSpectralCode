#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 07:40:18 2020

@author: jingqiu
"""
import matplotlib.pyplot as plt

def fig1plot(lamda, parameters, correlations1,correlations2,correlations3):
    naive,=plt.semilogx(parameters,correlations1,'r-*',label='naive',linewidth=2.0)
    truncate,= plt.plot(parameters,correlations2,'g-^',label='truncate')
    backtracking,=plt.plot(parameters,correlations3,'b-D',label='NBW')
    #saw,=plt.plot(parameters,correlations4,'-ro',label='SAW')
    worst,=plt.plot(parameters, [1-1/lamda**2 for parameter in parameters],label='worst',color='grey')
    #worst,=plt.plot(lamdas,[1-1/lamda**2 for lamda in lamdas],label='worst',color='grey')
    plt.legend(handles=[naive,truncate,backtracking,worst],loc='best',frameon=False);
    plt.xlabel(r'signal strength $\lambda$')
    plt.xlabel(r'SNR $\lambda$')
    plt.xlabel(r'parameter $d$')
    plt.ylabel('Squared correlation')
    #plt.savefig('MixedExperiment200parameters2.eps')
    plt.savefig('MinusERExperiment400lamda11.eps')
    plt.close()