# Stochastic_Optimal_Control_Market_Making

The limit order book (LOB) for Amazon over a 1-day period was used to calibrate a market making algorithm. The optimal bid-ask spread was calculated using Hamiltonian-Jacobian-Bellman (HJB) equation where the diffusion of the stock price was stochastic volatility with 0 drift. That is:
$$dS_t = \sqrt{v_t} dW_t$$
$$dv_t = \kappa(\bar{v} - v_t)dt + \eta \sqrt{v_t}dB_t$$
where $W_t$ and $B_t$ are brownian motions with correlation $\rho$.
The cash process used:
$$dX_t = (S_t + \delta^a)dN_t^a - (S_t - \delta^b)dN_t^b$$
$$dq_t = dN^b_t - dN^a_t$$
where $N_t$ is a poisson random variable given by the arrival of the market orders (MO), and $q_t$ is the inventory.

The utility function used was:
$$\mathbb{E}\left[-e^{-\gamma\left(X_T+q_T S_T-lq_T^2\right)}\right]$$

That is, we are punished for holding inventory at the end of the trading day.

Then using Bellman's principle of optimality:
$$U(t, X)=\sup _{\delta_t \in \mathcal{A}(t, X)}\left\{\mathbb{E}^{t, X}\left[\int_t^\theta L\left(s, X_s, \delta_s\right) d s+U\left(\theta,X_\theta\right)\right]\right\}$$
And applying Ito's lemma, we get the resultant HJB:

$$
\begin{align*}
0= & V_t(t, q, \nu, X, S)+\frac{1}{2} \nu V_{S S}(t, q, \nu, X, S)+\theta(\alpha-\nu) V_\nu(t, q, \nu, X, S) \\
& +\frac{1}{2} \xi^2 \nu V_{\nu \nu}(t, q, \nu, X, S)+\rho \xi \nu V_{S \nu}(t, q, \nu, X, S) \\
& +\mathbf{1}_{q<Q} \sup _{\delta_t^b}\left[\left[V\left(t, q+1, \nu+d \nu, X-S+\delta^b, S\right)-V(t, q, \nu, X, S)\right] \Lambda^b\left(\delta_t^b\right)\right], \\
& +\mathbf{1}_{q>-Q} \sup _{\delta_t^a}\left[\left[V\left(t, q-1, \nu+d \nu, X+S+\delta^a, S\right)-V(t, q, \nu, X, S)\right] \Lambda^a\left(\delta_t^a\right)\right],
\end{align}
$$

with terminal condition,
$$V(T, q, \nu, X, S)=-\exp (-\gamma(X+q S-l(|q|))) \text {. }$$

The optimal bid and ask spreads was calibrated to be:

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
