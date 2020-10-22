#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:20:22 2020

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

def GeneralSpectralAlgorithm(tensor,path_length):
    '''algorithm based on self-avoiding walk'''
    num_vertices=path_length+1
    shapes=np.shape(tensor);
    order=len(shapes);
    n=shapes[0];
    #size_color_matrix=n*np.exp(num_vertices);
    colors=num_vertices
    matrix1=[]
    matrix2=[]
    matrix3=[]
    # self-avoiding walk matrix can be represented as a sum of matrices
    # each matrix is a product of end_matrix1---k chain matrix--- end_matrix2
    # matrix1,2,3 record these matrices
    for c in range(2*colors):
        random_color=np.random.choice(colors,n);
        matrix1.append(chainMatrix(tensor,random_color,num_vertices));
        matrix2.append(endMatrix1(tensor,random_color,num_vertices));
        matrix3.append(endMatrix2(tensor,random_color,num_vertices));     
    
    # use power method for extracting the leading eigenvector
    # of symmetrized self-avoiding walk matrix
    hatx=2.0*random.binomial(1, 0.5,n)-1.0;
    hatx/=lin.norm(hatx)
    for iteration in range(2*int(np.log(n))):
        new_hatx=np.zeros(n);
        for c in range(2*colors):
            temp=matrix3[c]@hatx;
            temp2=matrix2[c].T@hatx;
            for t in range(path_length-2):
                temp=matrix1[c]@temp;
                temp2=matrix1[c].T@temp2;
            new_hatx+=matrix2[c]@temp;
            new_hatx+=matrix3[c].T@temp;
        hatx=new_hatx/lin.norm(new_hatx);
    return hatx;

def setMapIndex(colorSet,vertice):
    #given set of used colors and vertice, return the corresponding index of row/column in matrix
    return int(colorSet,2)+vertice*np.power(2,len(colorSet))
        
def chainMatrix(tensor,random_color,num_vertices):
    #print(num_vertices)
    '''matrix indexed by (i,S) and (j,V) used in chaining product construction of self-avoiding walk matrix''' 
    n=np.shape(tensor)[0];
    #result=np.zeros((np.power(2,num_vertices)*n,np.power(2,num_vertices)*n));
    rows=[]
    cols=[]
    entries=[]
    for i in range(np.power(2,num_vertices)):
        S='0'*(num_vertices-len(bin(i)[2:]))+bin(i)[2:];
        #print(S);
        for vertice in range(n):
            if S[random_color[vertice]]=='0':
                T=S[:random_color[vertice]]+'1'+S[random_color[vertice]+1:]
                for vertice2 in range(n):
                    if(T[random_color[vertice2]]=='0'):  
                        rows.append(setMapIndex(S,vertice));
                        cols.append(setMapIndex(T,vertice2));
                        entries.append(tensor[vertice,vertice2]);
                        #result[setMapIndex(S,vertice),setMapIndex(T,vertice2)]=tensor[vertice,vertice2]
    result=csr_matrix((np.array(entries),(np.array(rows),np.array(cols))),shape=(np.power(2,num_vertices)*n,np.power(2,num_vertices)*n));
    return result;

def endMatrix1(tensor,random_color,num_vertices):
    '''matrix used in chaining product corresponding to starting from color of vertice i'''
    n=np.shape(tensor)[0];
    rows=[]
    cols=[]
    entries=[]    
    #result=np.zeros((n,np.power(2,num_vertices)*n));
    for vertice in range(n):  
        T=np.power(2,num_vertices-random_color[vertice]-1)          #translate single vertice to index
        for vertice2 in range(n):
            if(random_color[vertice2]!=random_color[vertice]):
                rows.append(vertice)
                cols.append(T+vertice2*np.power(2,num_vertices))
                entries.append(tensor[vertice,vertice2])
                #result[vertice,T+vertice2*np.power(2,num_vertices)]=tensor[vertice,vertice2]
    return csr_matrix((np.array(entries),(np.array(rows),np.array(cols))),shape=(n,np.power(2,num_vertices)*n));
               
def endMatrix2(tensor,random_color,num_vertices):
    ''' matrix used in chaining product corresponding to ending at color of vertice j'''
    n=np.shape(tensor)[0]
    result=np.zeros((np.power(2,num_vertices)*n,n));
    rows=[]
    cols=[]
    entries=[]
    for vertice2 in range(n):
        T=('1'*num_vertices)[:random_color[vertice2]]+'0'+('1'*num_vertices)[random_color[vertice2]+1:]
        #T=np.power(2,num_vertices)-1-np.power(2,num_vertices-random_color[vertice2]-1) 
        for vertice in range(n):
            if(T[random_color[vertice]]=='1'):
                S=T[:random_color[vertice]]+'0'+T[random_color[vertice]+1:]
                #print([S,T,vertice,num_vertices])
                rows.append(setMapIndex(S,vertice))
                cols.append(vertice2)
                entries.append(tensor[vertice,vertice2])
    return csr_matrix((np.array(entries),(np.array(rows),np.array(cols))),shape=(np.power(2,num_vertices)*n,n));