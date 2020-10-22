#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 07:40:33 2020

@author: jingqiu
"""
import matplotlib.pyplot as plt

def fig2plot(lamdas,parameter, correlations1,correlations2):
    naive,=plt.plot(lamdas,correlations1,'r-*',label='naive',linewidth=2.0)
    truncate,= plt.plot(lamdas,correlations2,'g-^',label='truncate')
    #saw,=plt.plot(parameters,correlations4,'-ro',label='SAW')
    worst,=plt.plot(lamdas, [1-1/lamda**2 for lamda in lamdas],'-',label='worst',color='grey')
    #worst,=plt.plot(lamdas,[1-1/lamda**2 for lamda in lamdas],label='worst',color='grey')
    plt.legend(handles=[naive,truncate,worst],loc='best',frameon=False);
    plt.xlabel(r'SNR $\lambda$')
    plt.ylabel('Squared correlation')
    #plt.savefig('MixedExperiment200parameters2.eps')
    plt.savefig('NonIdenticalExperiment2000d30.eps')
    plt.close()