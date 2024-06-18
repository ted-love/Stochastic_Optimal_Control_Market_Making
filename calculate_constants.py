# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 22:58:57 2024

@author: TedLove
"""

import numpy as np
from scipy import stats
import pandas as pd

def _calculate_arrivIntensity_kappa(Lambda):
    log_Lambda = np.log(Lambda)
    res = stats.linregress(log_Lambda.index, log_Lambda.values)
    
    kappa = -res.slope
    A = np.exp(res.intercept)
    return kappa, A
    
def _calulate_Lambda(execution_LO, Delta, time, n):
    
    null_data = np.zeros(np.size(Delta))
    
    Lambda = pd.Series(data=null_data,index=Delta)
    
    for t in range(1,np.size(time)):
        temp_df = execution_LO.loc[time[t-1]:time[t]]
        x = temp_df['X'].value_counts()
        Lambda = Lambda.add(x,fill_value=0)
        
    
    Lambda = Lambda / (n-1)  
    Lambda = Lambda[Lambda != 0]
    return Lambda


def calculate_trading_intensities(order_book, n):
        
    execution_LO_bid = order_book[(order_book['direction']==1)& (order_book['X'] > 0)]
    execution_LO_ask = order_book[(order_book['direction']==-1) & (order_book['X'] > 0)]
    
    
    
    time = np.linspace(0,n-1,n)*60  # seconds in minute intervals
    
    
    Delta = np.unique(np.concatenate([execution_LO_ask['X'].values,execution_LO_bid['X'].values]))
    
    
    Lambda_bid = _calulate_Lambda(execution_LO_bid, Delta, time, n)
    Lambda_ask = _calulate_Lambda(execution_LO_ask, Delta, time, n)
    
    kappa_ask, A_ask = _calculate_arrivIntensity_kappa(Lambda_ask)
    kappa_bid, A_bid = _calculate_arrivIntensity_kappa(Lambda_bid)
    
 
    return kappa_ask, kappa_bid, A_ask, A_bid


def stochastic_volatility_params(LOB_truncated, variance, n, dt):

    term1 = np.sum(np.sqrt(variance[1:]*variance[:-1]))
    
    term2 = 0
    for i in range(1,n-1):
     
        term2 += np.sqrt(variance[i]/variance[i-1]) * np.sum(variance[:-1])
    
    numerator = 1/n *term1 - 1/n**2 * term2
    
    denominator = dt/ 2 - dt /2 /n**2 * np.sum(1/variance[:-1]) * np.sum(variance[:-1])
    
    P = numerator / denominator
    
        
    
    kappa = 2/dt * (1 + P*dt / 2/n * np.sum(1/variance[:-1]) - 1/n * np.sum(np.sqrt(variance[1:]/variance[:-1])))
    
    eta = np.sqrt(4 / dt/n * np.sum((np.sqrt(variance[1:]) - np.sqrt(variance[:-1]) \
                      - dt / 2 / np.sqrt(variance[:-1]) * (P - kappa*np.sqrt(variance[:-1])))**2))
    
    v_bar = (P + eta**2/4) / kappa
    
    dW = (LOB_truncated['midPrice'].values[1:] - LOB_truncated['midPrice'].values[:-1]) / np.sqrt(variance[:-1])
    
    dB = (np.diff(variance) - kappa * (v_bar - variance[:-1])  * dt )/ (eta * np.sqrt(variance[:-1]))
        
    rho = (1/n/dt) * np.sum(dW * dB)
    
    return v_bar, kappa, eta, rho

   