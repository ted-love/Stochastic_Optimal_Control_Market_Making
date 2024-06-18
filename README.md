# Stochastic_Optimal_Control_Market_Making

The limit order book (LOB) containing high-frequency data for Amazon over a 1-day period was used to calibrate a market making algorithm. To remove the noise of the high-frequency data and get an accurate measure of the running volatility, the two-scaled realized volatility (TSRV) subsampling technique was used. 

The optimal bid-ask spread was calculated using Hamiltonian-Jacobian-Bellman (HJB) equation where the diffusion of the stock price was stochastic volatility with 0 drift. That is:
$$dS_t = \sqrt{\nu_t} dW_t$$
$$d\nu_t = \kappa(\bar{\nu} - \nu_t)dt + \eta \sqrt{\nu_t}dB_t$$
where $W_t$ and $B_t$ are brownian motions with correlation $\rho$.
The cash process used:
$$dX_t = (S_t + \delta^a)dN_t^a - (S_t - \delta^b)dN_t^b$$
$$dq_t = dN^b_t - dN^a_t$$
where $N_t$ is a poisson random variable given by the arrival of the market orders (MO), and $q_t$ is the inventory.

The utility function used was:
$$\mathbb{E}\left[-e^{-\gamma\left(X_T+q_T S_T-\kappa q_T^2\right)}\right]$$

That is, we are punished for holding inventory at the end of the trading day.

Then using Bellman's principle of optimality:
$$U(t, X) = \sup_{\delta_t \in \mathcal{A}(t, X)} \{ \mathbb{E}^{t, X} \left[ \int_t^\theta L(s, X_s, \delta_s) \, ds + U(\theta, X_\theta) \right] \}$$

And applying Ito's lemma, we get the resultant HJB:
<div align="center">
<img src="https://github.com/ted-love/Stochastic_Optimal_Control_Market_Making/assets/46618315/9e677ddf-383f-4da1-b138-cad19f3e941a" width="600" height="auto">
</div>

with terminal condition, $V(T, q, \nu, X, S)=-e^{-\gamma(X+q S-\kappa q^2)}$

To reduce the dimensions, we use the ansatz solution:
$$V(t, q, \nu, X, S)=-e^{-\gamma(X+q S)} U(t, q, \nu)$$
which gives the resultant HJB:
<div align="center">
<img src="https://github.com/ted-love/Stochastic_Optimal_Control_Market_Making/assets/46618315/f62b2022-cce2-4fe9-b7f1-ba0c400c41a8" width="600" height="auto">
</div>

This ansatz solution is known to converge to the viscosity solution and is unique

To solve the HJB PDE, the Thomas Algorithm was implemented, and the optimal bid and ask spreads were calibrated to be:

![newplot (14)](https://github.com/ted-love/Stochastic_Optimal_Control_Market_Making/assets/46618315/902390e6-c98a-4a7a-b053-a6a466a88bf1)
![newplot (15)](https://github.com/ted-love/Stochastic_Optimal_Control_Market_Making/assets/46618315/e6370f95-5225-4acb-8d3b-d3654c60364e)

Then using these optimal quotes, backtests were ran to test the efficacy. At time $t$ set bid and ask quotes given current volatility $\nu_t$ and inventory $q$. Then at time $t+1$, the stock price $S_t$ and volatility $\nu_t$, evolve according to their SDEs. If the price hits the bid or ask, a 1 unit trade is executed (we do not requote until a pre-defined period, it's why in the figure below was have flat lines for the quotes). 

![newplot (16)](https://github.com/ted-love/Stochastic_Optimal_Control_Market_Making/assets/46618315/75c14dde-770c-4a1b-b50f-f19916458ee9)


Sources: 
Amit Zubier Arfan, (2021), On the Topic of Market Making Models: Applying and Calibrating with Stochastic Volatility and Limit Order Book Data.

Dependencies:
```
pip install plotly
```
