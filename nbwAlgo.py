#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:48:08 2020

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

#things to change:first change the matrix format to sparse
#second:change the chain matrix. respectively process the first few products to save the space and time

def NonBackTrackingAlgorithm(tensor,path_length,steps=1):
    '''estimator based on Nonbacktracking walk
     steps=1 corresponds to naive non-backtracking estimator(without direct backtrack)'''
    num_vertices=path_length+1
    shapes=np.shape(tensor);
    order=len(shapes);
    n=shapes[0];
    #size_color_matrix=n*np.exp(num_vertices);
    colors=num_vertices
    matrix1=[]
    matrix2=[]
    matrix3=[]
    for c in range(colors):
        random_color=np.random.choice(colors,n)+1;
        matrix1.append(chainMatrixNonBacktracking(tensor,random_color,num_vertices,steps));
        matrix2.append(endMatrixNonBackTracking1(tensor,random_color,num_vertices,steps));
        matrix3.append(endMatrixNonBackTracking2(tensor,random_color,num_vertices,steps));     
    hatx=2.0*random.binomial(1, 0.5,n)-1.0;
    hatx/=lin.norm(hatx)
    for iteration in range(2*int(np.log(n))):
        new_hatx=np.zeros(n);
        for c in range(colors):
            temp=matrix3[c]@hatx;
            temp2=(matrix2[c].T)@hatx;
            for t in range(path_length-1):
                temp=matrix1[c]@temp;
                temp2=(matrix1[c].T)@temp2;
            new_hatx+=matrix2[c]@temp;
            new_hatx+=(matrix3[c].T)@temp2;
        hatx=new_hatx/lin.norm(new_hatx);
    return hatx;   

def fieldSetMapIndex(colorSet,colors):
    
    '''map color set to index for non-backtracking estimator'''
    '''colors:number of colors used'''
    
    '''S is an element of F_{q}^size'''
    '''here q is just number of colors'''
    '''size is number of steps'''
    #print(polynomial.polyval(colors,colorSet[::-1])+vertice*np.power(colors,size))
    if(np.size(colorSet)==1):
        return colorSet[0]-1;
    return int(round(polynomial.polyval(colors,np.array(colorSet[::-1])-1)))

def fieldIndexMapSet(index,colors,step):
    '''map index to color set'''
    
    '''here index is the colorset index and step is length of coding'''
    Set=[]
    while step!=0:
        Set.append((index%colors)+1)
        index//=colors;
        step-=1;
    return Set[::-1]
    

            
def chainMatrixNonBacktracking(tensor,random_color,num_vertices,steps):
    '''num_vertices represent the number of colors and number of 
    vertices in the non-backtracking walk here, random coloring is given by [num_vertices]'''
    
    non_zero_elements=0
    n=np.shape(tensor)[0];
    color_size=[0]
    for step in range(steps+1):
        color_size+=[np.power(num_vertices,step)+color_size[-1]];
    '''next we find a bijection between the index and color set'''
    '''First strategy is to use sparse matrix'''
    '''Second strategy is to introduce 0 for incomplete colors and use (num_vertices+1)-ary code'''
    #result=np.zeros((color_size[-1]*n,color_size[-1]*n));
    rows=[];
    cols=[];
    entries=[];
    #binary_coding=[0]*num_vertices;
    for step in range(steps+1):  
        #choose size step 
        for index in range(color_size[step],color_size[step+1]):
            zero_paddings=steps-step
            record_colors=fieldIndexMapSet(index-color_size[step],num_vertices,step)
            S=[0]*zero_paddings+record_colors;
            #print(S);
            binary_coding=[0]*num_vertices
            for i in record_colors:
                binary_coding[i-1]=1;
            for vertice in range(n):
                if binary_coding[random_color[vertice]-1]==0:
                    if(step==steps):
                        T=[random_color[vertice]]+S[:-1];
                    else:
                        T=[0]*(zero_paddings-1)+[random_color[vertice]]+S[zero_paddings:]
                    #print([step,S,T,vertice,random_color[vertice]])
                    #input()
                    for vertice2 in range(n):
                        if binary_coding[random_color[vertice2]-1]==0 and random_color[vertice2]!=random_color[vertice]:
                            #print([T,num_vertices,steps,vertice2])
                            step2=min(step+1,steps);
                            non_zero_elements+=1
                            '''sparse'''
                            if(non_zero_elements%100000==0):
                                print(non_zero_elements)
                            rows.append(index+vertice*color_size[-1])
                            cols.append(fieldSetMapIndex(T,num_vertices)+color_size[step2]+vertice2*color_size[-1])
                            entries.append(tensor[vertice,vertice2]);
                            
                            #result[index+vertice*color_size[-1],fieldSetMapIndex(T,num_vertices)+color_size[step2]+vertice2*color_size[-1]]=tensor[vertice,vertice2]
    result=csr_matrix((np.array(entries),(np.array(rows),np.array(cols))),shape=(color_size[-1]*n,color_size[-1]*n));
    
    #print(non_zero_elements)
    #print(np.shape(result))
    return result;

def endMatrixNonBackTracking1(tensor,random_color,num_vertices,steps):
    '''matrix corresponds to the start of non-backtracking walk'''
    
    n=np.shape(tensor)[0];
    color_size=[0]
    for step in range(steps+1):
        color_size+=[np.power(num_vertices,step)+color_size[-1]];
    colors=color_size[-1];
    result=np.zeros((n,colors*n));
    for vertice in range(n):     #translate single vertice to index
        result[vertice,vertice*colors]=1
    return result;    

def endMatrixNonBackTracking2(tensor,random_color,num_vertices,steps):
    '''matrix corresponds to the end of non-backtracking walk'''
    n=np.shape(tensor)[0];
    color_size=[0]
    for step in range(steps+1):
        color_size+=[np.power(num_vertices,step)+color_size[-1]];
    '''next we find a bijection between the index and color set'''
    '''First strategy is to use sparse matrix'''
    '''Second strategy is to introduce 0 for incomplete colors and use (num_vertices+1)-ary code'''
    result=np.zeros((color_size[-1]*n,n));
    #for step in range(steps+1):  
    step=steps;
        #choose size step 
    for index in range(color_size[step],color_size[step+1]):
        zero_paddings=steps-step
        S=[0]*zero_paddings+fieldIndexMapSet(index-color_size[step],num_vertices,step);
        #print(S);
        binary_coding=[0]*num_vertices
        for i in fieldIndexMapSet(index-color_size[step],num_vertices,step):
            binary_coding[i-1]=1;
        for vertice in range(n): 
            if binary_coding[random_color[vertice]-1]==0:                  
                for vertice2 in range(n):
                    if binary_coding[random_color[vertice2]-1]==0 and random_color[vertice2]!=random_color[vertice]:
                        #print([T,num_vertices,steps,vertice2])
                       result[index+vertice*color_size[-1],vertice2]=tensor[vertice,vertice2]
    return result;