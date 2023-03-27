import numpy as np
import math
import scipy.stats as st
from typing import List,Tuple
from scipy.optimize import fsolve

def nnodes(nperiods):
    return (nperiods + 2) * (nperiods + 1) // 2


def node_idx(i, j):
    return nnodes(j - 1) + i


def cal_price_chg_prob(u, d, dt, carry_cost):
    pu = (np.exp(carry_cost * dt) - d) / (u - d)
    return pu, 1 - pu


def cal_price_chg(ivol, dt):
    u = np.exp(ivol * np.sqrt(dt))
    return u, 1 / u


def bt_american_continuous_div(is_call, price, strike, ttm, r_f, div_rate, ivol, nperiods):
    is_call = 1 if is_call else -1
    dt = ttm / nperiods
    u, d = cal_price_chg(ivol, dt)
    pu, pd = cal_price_chg_prob(u, d, dt, r_f - div_rate)
    
    option_values = np.empty(nnodes(nperiods), dtype=float)
    for j in range(nperiods, -1, -1):
        for i in range(j, -1, -1):
            idx = node_idx(i, j)
            curr_price = price * u ** i * d ** (j - i)
            option_values[idx] = max(0, is_call * (curr_price - strike))
            if j < nperiods:
                value_no_exercise = np.exp(-r_f * dt) * (pu * option_values[node_idx(i + 1, j + 1)]
                                                         + pd * option_values[node_idx(i, j + 1)])
                option_values[idx] = max(option_values[idx], value_no_exercise)
    
    return option_values[0]


def bt_american(is_call: bool, price, strike, ttm, r_f, div_rate,
                ivol, nperiods, dividends: List[Tuple] = None):
    """
    Caculate the value of american options by binomial tree.
    """
    
    if not dividends or dividends[0][0] > nperiods:
        return bt_american_continuous_div(is_call, price, strike, ttm, r_f, div_rate, ivol, nperiods)
    
    is_call = 1 if is_call else -1
    dt = ttm / nperiods
    u, d = cal_price_chg(ivol, dt)
    pu, pd = cal_price_chg_prob(u, d, dt, r_f - div_rate)
    
    div_time, div_amount = dividends[0]  # do not use dividends.pop(0),
    # because it will modify the dividends argument of the function on the previous stack
    option_values = np.empty(nnodes(div_time), dtype=float)
    
    # update the parameters for the next recursion
    new_dividends = [(time - div_time, amount) for time, amount in dividends[1:]]
    new_ttm = ttm - div_time * dt
    new_nperiods = nperiods - div_time
    
    for j in range(div_time, -1, -1):
        for i in range(j, -1, -1):
            curr_price = price * u ** i * d ** (j - i)
            value_exercise = max(0, is_call * (curr_price - strike))
            if j < div_time:
                value_no_exercise = np.exp(-r_f * dt) * \
                                    (pu * option_values[node_idx(i + 1, j + 1)] + pd * option_values[
                                        node_idx(i, j + 1)])
            else:
                value_no_exercise = bt_american(True if is_call == 1 else False, curr_price - div_amount, strike,
                                                new_ttm,
                                                r_f, div_rate, ivol, new_nperiods, new_dividends)
            option_values[node_idx(i, j)] = max(value_exercise, value_no_exercise)
    
    return option_values[0]

def am_bs(S,K,T,r,CR,v,option):
    d1 = (np.log(S/K)+(r-CR+v**2/2)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    if option == True:
        return S*np.exp(-CR*T)*st.norm.cdf(d1)-K*np.exp(-r*T)*st.norm.cdf(d2)
    elif option == False:
        return K*np.exp(-r*T)*st.norm.cdf(-d2)-S*np.exp(-CR*T)*st.norm.cdf(-d1)
    
def implied_vol(S, K, T,r,q,opt_price,opt=True):
    volatility = lambda x: am_bs(S,K,T,r,q,x,opt) -opt_price
    return fsolve(volatility,x0 = 0.5)[0]


