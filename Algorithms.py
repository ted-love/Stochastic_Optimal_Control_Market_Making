# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 23:03:29 2024

@author: TedLove
"""
import numpy as np

def Thomas_Algorithm(a,b,c,d):
    n = len(d)
    c_star = np.zeros(n-1)
    d_star = np.zeros(n)
    x = np.zeros(n)
    
    c_star[0] = c[0]/b[0]
    d_star[0] = d[0]/b[0]

    for i in range(1,n-1):
        c_star[i] = c[i]/(b[i] - a[i-1]*c_star[i-1])
        
    for i in range(1,n):
        d_star[i] = (d[i] - a[i-1]*d_star[i-1])/(b[i] - a[i-1]*c_star[i-1])
        
    x[n-1] = d_star[n-1]
    
    for i in range(n-1,0,-1):
        x[i-1] = d_star[i-1] - c_star[i-1]*x[i]
        
    return x

def Thomas_Algorithm_Shermann(a,b,c,d):
    n = len(d)

    alpha = 0.
    beta = c[n-1]
    gamma = -b[0]
    
    x = np.array([1]+[0]*(n-2)+[alpha])
    
    cmod,u = np.empty(n),np.empty(n)

    cmod[0] = alpha / (b[0] - gamma)
    u[0] = gamma / (b[0] - gamma)
    x[0] = x[0] / (b[0] - gamma)
    for i in range(n-1):
        m = 1. / (b[i] - a[i]*cmod[i-1])
        cmod[i] = m * c[i]
        u[i] = m * (-a[i]*u[i-1])
        x[i] = m * (x[i] - a[i]*u[i-1])
        
    for i in range(n-2,-1,-1):
        u[i] = u[i] - cmod[i]*u[i+1]
        x[i] = x[i] - cmod[i]*x[i+1]
        
    factor =  (x[0] + x[n - 1] * beta / gamma) / (1.0 + u[0] + u[n - 1] * beta / gamma)
    for i in range(0,n):
        x[i] = x[i] - factor * u[i]
        
    return x
