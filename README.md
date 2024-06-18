# Stochastic_Optimal_Control_Market_Making

The limit order book (LOB) for Amazon over a 1-day period was used to calibrate a market making algorithm. The optimal bid-ask spread was calculated using optimal control where the diffusion of the stock price was stochastic volatility with 0 drift. That is:
$$dS_t = \sqrt{v_t} dW_t$$
$$dv_t = \kappa(\bar{v} - v_t)dt + \eta dB_t$$
where $W_t$ and $B_t$ are brownian motions with correlation $\rho$.
The cash process used:
$$dX_t = (S_t + \delta^a)dN_t^a - (S_t - \delta^b)dN_t^b$$
where $N_t$ is a poisson random variable given by the arrival of the market orders (MO).

The utility function used was:
$$\mathbb{E}\left[-e^{-\gamma\left(X_T+q_T S_T-lq_T^2\right)}\right]$$


The optimal bid and ask spreads are given by

![newplot (14)](https://github.com/ted-love/Stochastic_Optimal_Control_Market_Making/assets/46618315/902390e6-c98a-4a7a-b053-a6a466a88bf1)
![newplot (15)](https://github.com/ted-love/Stochastic_Optimal_Control_Market_Making/assets/46618315/e6370f95-5225-4acb-8d3b-d3654c60364e)

Then using these optimal quotes, backtests were ran to test the efficacy

![newplot (16)](https://github.com/ted-love/Stochastic_Optimal_Control_Market_Making/assets/46618315/75c14dde-770c-4a1b-b50f-f19916458ee9)


Sources: 
Amit Zubier Arfan, (2021), On the Topic of Market Making Models: Applying and Calibrating with Stochastic Volatility and Limit Order Book Data.

Dependencies:
```
pip install plotly
```
