import numpy as np
from scipy import linalg
from scipy.stats import norm,spearmanr

def normal_simulation(mu,cov_mat,nsim):
    L=linalg.cholesky(cov_mat,lower=True)
    num=cov_mat.shape[0]
    mat=np.random.randn(nsim,num)

    rs=np.dot(L,mat.T).T
    rs=rs+mu

    return rs

def pca_simulation(mu,cov_mat,nsim,percent):
    eigval, eigvec = np.linalg.eig(cov_mat)
    pos=np.where(np.real(eigval)>0)[0]
    eigval_2=np.real(eigval[pos])
    cumulative_variance=np.cumsum(eigval_2/np.sum(eigval_2))
    pos_2=np.where(cumulative_variance>=percent)[0][0]
    pos_2=np.arange(pos_2+1)
    B=np.dot(eigvec[:,pos_2],np.diag(eigval[pos_2]))
    Z=np.random.randn(len(pos_2),nsim)
    X=np.dot(B,Z).transpose()
    X=X+mu

    return X

def copula_simulation(df,nsim):
    row_num=df.shape[0]
    col_num=df.shape[1]
    rs=np.zeros([row_num,col_num])
    for i in range(col_num):
        rs[:,i]=norm.ppf(df[:,i])

    cov_mat=spearmanr(rs)[0]
    mu=np.zeros(col_num)
    sim=normal_simulation(mu, cov_mat, nsim)
    sigma=np.sqrt(np.diag(cov_mat))
    sim=np.real(sim)

    rs_2=np.zeros([nsim,col_num])
    for i in range(col_num):
        rs_2[:,i]=norm.cdf(sim[:,i],mu[i],sigma[i])

    return rs_2


















