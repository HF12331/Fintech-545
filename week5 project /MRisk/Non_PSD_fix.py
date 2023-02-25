import numpy as np
from numpy import inf
from numpy import copy
from numpy.linalg import norm

def g_matrix(n):
    npsdm = np.full((n,n),0.9)
    for i in range(n):
        npsdm[i,i]=float(1.0)
    npsdm[0,1]=0.7357
    npsdm[1,0]=0.7357
    return npsdm

def chol_psd(m):
    root = np.full(m.shape, 0.0)
    n = root.shape[1]
    for j in range(n):
        diag_val = m[j, j] - root[j, :j] @ root[j, :j].T
        if -1e-6 <= diag_val <= 0:
            diag_val = 0.0
        elif diag_val < -1e-6:
            raise ValueError("Matrix is none-psd")
        root[j, j] = np.sqrt(diag_val)
        if root[j, j] == 0:
            continue
        for i in range(j + 1, n):
            root[i, j] = (m[i, j] - root[i, :j] @ root[j, :j].T) / root[j, j]

    return np.matrix(root)

def near_psd(m):
    w, v = np.linalg.eigh(m)
    w[w<0] = 0.0
    s_square = np.square(v)
    T = 1 / (s_square @ w)
    T = np.diagflat(np.sqrt(T))
    L = np.diag(np.sqrt(w))
    B = T @ v @ L
    out = B @ B.T # B * B'
    return out

def higham(A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
             weights=None, verbose=False,
             except_on_too_many_iterations=True):

    # If input is an ExceededMaxIterationsError object this
    # is a restart computation
    if (isinstance(A, ValueError)):
        ds = copy(A.ds)
        A = copy(A.matrix)
    else:
        ds = np.zeros(np.shape(A))

    eps = np.spacing(1)
    if not np.all((np.transpose(A) == A)):
        raise ValueError('Input Matrix is not symmetric')
    if not tol:
        tol = eps * np.shape(A)[0] * np.array([1, 1])
    if weights is None:
        weights = np.ones(np.shape(A)[0])
    X = copy(A)
    Y = copy(A)
    rel_diffY = inf
    rel_diffX = inf
    rel_diffXY = inf

    Whalf = np.sqrt(np.outer(weights, weights))

    iteration = 0
    while max(rel_diffX, rel_diffY, rel_diffXY) > tol[0]:
        iteration += 1
        if iteration > max_iterations:
            if except_on_too_many_iterations:
                if max_iterations == 1:
                    message = "No solution found in "\
                              + str(max_iterations) + " iteration"
                else:
                    message = "No solution found in "\
                              + str(max_iterations) + " iterations"
                raise ExceededMaxIterationsError(message, X, iteration, ds)
            else:
                # exceptOnTooManyIterations is false so just silently
                # return the result even though it has not converged
                return X

        Xold = copy(X)
        R = X - ds
        R_wtd = Whalf*R
        if flag == 0:
            X = proj_spd(R_wtd)
        elif flag == 1:
            raise NotImplementedError("Setting 'flag' to 1 is currently\
                                 not implemented.")
        X = X / Whalf
        ds = X - R
        Yold = copy(Y)
        Y = copy(X)
        np.fill_diagonal(Y, 1)
        normY = norm(Y, 'fro')
        rel_diffX = norm(X - Xold, 'fro') / norm(X, 'fro')
        rel_diffY = norm(Y - Yold, 'fro') / normY
        rel_diffXY = norm(Y - X, 'fro') / normY

        X = copy(Y)

    return X

def proj_spd(A):
    # NOTE: the input matrix is assumed to be symmetric
    d, v = np.linalg.eigh(A)
    A = (v * np.maximum(d, 0)).dot(v.T)
    A = (A + A.T) / 2
    return(A)






