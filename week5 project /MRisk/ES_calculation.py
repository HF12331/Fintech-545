import numpy as np
from scipy.stats import norm,t
from scipy import stats

def Normal_ES(x,alpha):
    mu = np.mean(x)
    sd = np.std(x)
    ES_Normal = -mu + sd * norm.pdf(norm.ppf(alpha, 0, 1)) / alpha

    return ES_Normal

def Simulation_ES(dist,x,alpha):
    if dist=='normal':
        mu = np.mean(x)
        sd = np.std(x)
        sim = norm.rvs(mu,sd,10000)
        VaR_Normal = norm.ppf(1 - alpha, mu, sd)
        ES = -np.mean(sim[sim <= -VaR_Normal])

    if dist=='t':
        temp = getattr(stats, 't')
        params = temp.fit(x)
        VaR_t = -t.ppf(alpha, df=params[0], loc=params[1], scale=params[2])
        sim = t.rvs(df=params[0], loc=params[1], scale=params[2], size=10000)
        ES = -np.mean(sim[sim <= -VaR_t])

    return ES

