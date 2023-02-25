import numpy as np
from scipy.stats import norm,t
from scipy import stats

def Normal_VaR(x,alpha):

    mu = np.mean(x)
    sd = np.std(x)
    VaR_Normal = -norm.ppf(alpha, mu, sd)

    return VaR_Normal

def t_VaR(x,alpha):
    dist = getattr(stats, 't')
    params = dist.fit(x)
    VaR_t = -t.ppf(alpha, df=params[0], loc=params[1], scale=params[2])

    return VaR_t

def Historical_VaR(x,alpha):
    sample = np.random.choice(x, size=10000, replace=True)
    VAR_HS = -np.percentile(sample, alpha*100)

    return VAR_HS

def VaR(data, mean, alpha: float = 0.05):
    return mean - np.quantile(data, q=alpha, method='midpoint')



