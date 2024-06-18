# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 07:48:33 2024

@author: TedLove
"""

import numpy as np
import pandas as pd
from scipy import interpolate
import local_plotting
from LOB_clean_up import clean_up_order_book, clean_up_LOB
from calculate_constants import calculate_trading_intensities, stochastic_volatility_params
import Algorithms

def subsample_volatility(LOB, K):
    """
    This function subsamples the high-frequency data points to remove the noise and get the Two-Scaled Realized Variance (TSRV)

    Parameters
    ----------
    LOB : DataFrame
        Limit order book with levels.
    K : Int
        Length of the subsample.

    Returns
    -------
    variance : NumPy Array
        TSRV with len H .
    LOB_truncated : DataFrame
        The truncated LOB book with len H.
    H : float
        length of the chosen subsample.
    time_max : float
        max time (seconds).

    """

    n = len(LOB)
    n_bar = (n-K+1) / K
    
    
    H = int(390 *3 / 2)
    TSRV = np.empty(H)
    
    time_max = LOB.index.values.max()    # seconds
    
    time_index = np.linspace(0,n,H+1,dtype=int)
    time_index = time_index[1:]
    time_vec = np.linspace(0,time_max,1000)
    time_vec = time_vec[1:]
    
    idx=0
    
    for j in time_index:
        n = np.size(LOB.iloc[K:j]['midPrice'])
        n_bar = (n-K+1) / K
        
        TSRV_sub_ave = np.sum((LOB.iloc[K:j]['midPrice'].values - LOB.iloc[:j-K]['midPrice'].values)**2)
        
        TSRV_noisy = np.sum((LOB.iloc[1:j]['midPrice'].values - LOB.iloc[:j-1]['midPrice'].values)**2)
        
        TSRV[idx] = 1/K * TSRV_sub_ave - n_bar / n * TSRV_noisy
        
        if TSRV[idx]<0:
            break

        idx+=1
        
    
    variance = np.diff(TSRV,n=1) / (time_index[1]-time_index[0])
    variance = np.insert(variance,0,variance[0])

    time_increments = LOB.index.max() / H
    LOB_truncated = pd.DataFrame(columns = LOB.columns)
    
    for i in range(1,H+1):
        
        end_data = LOB[LOB.index.values<i * time_increments].tail(1)
        LOB_truncated = pd.concat([LOB_truncated if not LOB_truncated.empty else None, end_data])
       
    LOB_truncated = LOB_truncated.reset_index(drop=True)
    LOB_truncated = LOB_truncated.shift(periods=1)
    LOB_truncated.loc[0] = LOB.loc[0]

    return variance, LOB_truncated, H, time_max


def solve_optimal_bid_ask(A_ask,A_bid,kappa_ask,kappa_bid,alpha,gamma,q_min,q_max,v_max,T,M,N,v_bar,kappa,rho,eta):
    """
    

    Parameters
    ----------
    A_ask : float
        Arrival intensity parameter of buyers.
    A_bid : float
        Arrival intensity parameter of sellers.
    kappa_ask : float
        Arrival intensity parameter of buyers.
    kappa_bid : float
        Arrival intensity parameter of sellers.
    alpha : float
        Inventory punishment factor.
    gamma : float
        risk aversion parameter.
    q_min : int
        minimum inventory.
    q_max : int
        maximum inventory.
    v_max : float
        maximum variance.
    T : float
        max time.
    M : int
        number of spatial steps (variance).
    N : int
        number of time steps.
    v_bar : float
        long-term variance.
    kappa : float
        rate of mean-reversion.
    rho : float
        correlation between variance and stock price.
    eta : float
        volatility of volatility.

    Returns
    -------
    U_dict : Dict
        A dictionary of the solutions of U. The keys are the inventory and the values are MxN NumPy arrays.
    d_bid_dict : Dict
        A dictionary of the optimal bid prices. The keys are the inventory and the values are MxN NumPy arrays.
    d_ask_dict : Dict
        A dictionary of the optimal ask prices. The keys are the inventory and the values are MxN NumPy array.

    """
    

    dt = T/M
    v_min = 1e-10

    v_vec = np.linspace(v_min,v_max,N+1)
    
    dv = v_max/N
        
    U = np.zeros((N, M+1))
    d_bid_dict={}
    d_ask_dict={}

    U_dict = {}
    for q in range(q_min,q_max+1):
        U[:,-1] = np.exp(-gamma*alpha*np.abs(q))
        U_dict[q] = U.copy()
        
    for q in range(q_min+1,q_max+1):  
        
        p = (1/gamma) * np.log(U_dict[q-1][:,-1]/U_dict[q][:,-1]) 
        mat = np.zeros((N,M+1))
        d_ask_dict[q] = mat
        d_ask_dict[q][:,-1] = 1/gamma * np.log(1+gamma/kappa_ask) + p #+ alpha*(1-2*q)
    
    for q in range(q_min,q_max):
        p = (1/gamma) * np.log(U_dict[q+1][:,-1]/U_dict[q][:,-1])
    
        mat = np.zeros((N,M+1))
        d_bid_dict[q] = mat
        
        d_bid_dict[q][:,-1] =  1/gamma * np.log(1+gamma/kappa_bid) + p #+ alpha*(1+2*q)
    
    for t in range(M-1,-1,-1):
        print('t: ',t)
        for q in range(q_min+1,q_max):
        
            F = A_ask * np.exp( -kappa_ask * d_ask_dict[q][:,t+1]) * ( U_dict[q-1][:,t+1] * np.exp(-gamma * d_ask_dict[q][:,t+1]) - U_dict[q][:,t+1]) \
              + A_bid * np.exp( -kappa_bid * d_bid_dict[q][:,t+1]) * ( U_dict[q+1][:,t+1] * np.exp(-gamma * d_bid_dict[q][:,t+1]) - U_dict[q][:,t+1]) \
        
            d_vec = - U_dict[q][:,t+1] / dt - F
            a_vec = 0.5 * v_vec * eta**2 / dv**2 - 0.5 * ( kappa * (v_bar - v_vec) - rho*eta*v_vec*gamma*q**2 / dv)
            b_vec = - v_vec * eta**2 / dv**2 - 1/dt + 0.5 * v_vec * (gamma**2) * q**2
            c_vec = 0.5 * v_vec * eta**2 / dv**2 + 0.5 * ( kappa * (v_bar - v_vec) - rho*eta*v_vec*gamma*q**2 / dv)
            
            U = Algorithms.Thomas_Algorithm(a_vec, b_vec, c_vec, d_vec)
            U_dict[q][:,t] = U
            
        U_dict[q_min] = U_dict[q_min+1]
        U_dict[q_max] = U_dict[q_max-1]
        
        for q in range(q_min+1,q_max+1):  
            
            p = (1/gamma) * np.log(U_dict[q-1][:,t]/U_dict[q][:,t]) 
            d_ask_dict[q][:,t] = 1/gamma * np.log(1+gamma/kappa_ask) + p 
        
        for q in range(q_min,q_max):
            
            p = (1/gamma) * np.log(U_dict[q+1][:,t]/U_dict[q][:,t])
            d_bid_dict[q][:,t] = 1/gamma * np.log(1+gamma/kappa_bid) + p 
    
    return U_dict,d_bid_dict,d_ask_dict



order_book = pd.read_csv('Data/AMZN_2012-06-21_34200000_57600000_message_10.csv',
                              names=['time','type','ID','size','price','direction'],index_col=['time'])

LOB_spread = pd.read_csv('Data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv',header=None)



LOB = clean_up_LOB(LOB_spread)
order_book, LOB = clean_up_order_book(order_book, LOB)


n=391
kappa_ask, kappa_bid, A_ask, A_bid = calculate_trading_intensities(order_book, n)

K = 30 # Length of the subsample
variance, LOB_truncated, H, time_max = subsample_volatility(LOB, K)


n = H
dt = time_max / n
variance = variance *100
v_bar, kappa, eta, rho = stochastic_volatility_params(LOB_truncated, variance, n, dt)

T=1

q_max = 25      # max inventory
q_min = -25     # min inventory

M=300     # number of spatial steps for the variance 
N=1_000   # number of time steps

alpha = 0.1 # invetory punishment factor
gamma=0.1   # risk-aversion parameter

vol_max = 200. # vol as a %

v_max = (vol_max / np.sqrt(252) / 100) ** 2

v_bar = LOB_truncated['midPrice'].pct_change().var()

kappa_ask = 0.5
kappa_bid = 0.5


U_dict,d_bid_dict,d_ask_dict = solve_optimal_bid_ask(A_ask, A_bid, kappa_ask, kappa_bid, gamma, alpha,
                                                     q_min, q_max, v_max, T, M, N, v_bar, kappa, rho, eta)


local_plotting.bid_ask_quotes(T,M,N,v_max,d_ask_dict,d_bid_dict)


#%%


S_0 = LOB_spread.iloc[0]['midPrice']

k = 100
S = S_0

S_list = [S]
sale=0
buy=0

t_vec = np.linspace(0,T,M+1)
    
v_min = 1e-10
v_vec = np.linspace(v_min,v_max,N)

dv = v_max/N
    
d_ask_dict_interp = {}
d_bid_dict_interp = {}

d_bid_dict_interp[-25] = interpolate.RegularGridInterpolator((v_vec,t_vec), d_bid_dict[-25])
d_ask_dict_interp[25] = interpolate.RegularGridInterpolator((v_vec,t_vec), d_ask_dict[25])

for q in range(q_min+1,q_max):
    
    d_ask_dict_interp[q] = interpolate.RegularGridInterpolator((v_vec,t_vec), d_ask_dict[q])
    d_bid_dict_interp[q] = interpolate.RegularGridInterpolator((v_vec,t_vec), d_bid_dict[q])


S = S_0

X = 25000
k = 10
kk = 58
dt = 1 / k / kk
eta = 0.8/252

portfolio_values = []
utility_values = []
dt = 1 / k / kk



f1 = lambda x : np.maximum(x,0)
f2 = lambda x : np.maximum(x,0)

Nsims = 100
for m in range(Nsims):
        
    q=0
    
    v_0 = v_max/1.05
    v = v_0
    
    dt = 1/585
    
    i=0
    S = S_0
    X = 25_000
    
    sale=0
    buy=0

    v_vec = np.zeros(k * kk+1)
    v_vec[0] = v_0
    v_tilde = v_vec
    U_0 = X + q*S - q*S**2
    S_vec = [S_0]
    
    pa = []
    pb = []
    
    while i < k:
        
        point = np.array([v, i/585])
        
        p_a = S + d_ask_dict_interp[q](point).item()
        p_b = S - d_bid_dict_interp[q](point).item()

        ask_hit = False
        bid_hit = False
        
        for j in range(1,kk+1):
            
            if S >= p_a and ask_hit==False and q > -24:
                q = q - 1
                X = X + p_a
                ask_hit=True
                sale+=1
        
            if S<= p_b and bid_hit == False and q < 24:
                q = q + 1
                X = X - p_b     
                bid_hit=True
                buy+=1
            
            Z_v = np.random.randn(1).item()
            Z_2 = np.random.randn(1).item()
            
            Z_s = rho * Z_v + np.sqrt(1-rho**2)*np.random.randn(1).item()
            
            v_tilde[58*i + j] = f1(v_tilde[58*i + j-1]) + kappa*dt * (v_bar - f2(v_tilde[58*i + j-1])) \
                                 + eta *np.sqrt(f2(v_tilde[58*i + j -1])*dt) * Z_v \
                                 +  0.25 * (eta**2) * (Z_v**2 - 1)*dt   # Milstein
                
            v_vec[58*i + j] = f2(v_tilde[58*i+j])
            
            S *= np.exp( -0.5 * v_vec[58*i+j] *dt + np.sqrt(v_vec[58*i+j]*dt)*Z_s)
            
            S_vec.append(S)
            pa.append(p_a)
            pb.append(p_b)
  
            if ask_hit==True and bid_hit == True:
                break
        i+=1
         
    V = X + q*S
    U = X + q*S - alpha*q**2
    
    portfolio_values.append(V)
    utility_values.append(U)
    
    if m % 10==0:
        print(m)

    
print("Average Portfolio PnL: ", np.array(portfolio_values).mean() - 25_000)
print("Average Utility: ", np.array(utility_values).mean() - U_0)

local_plotting.backtest_algorithm(S_vec, pa, pb)



