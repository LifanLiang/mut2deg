### Numba implementation of deep Boolean learning with one hidden OR-gate Boolean layer
from numba import jit, prange
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain

@jit('float64[:,:](float64[:,:], float64[:,:])', nopython=True, nogil=True, parallel=True)
def reconstruct(U, Z):
    res = np.zeros((U.shape[1],U.shape[0], Z.shape[1]))
    for l in prange(U.shape[1]):
        temp = np.outer(U[:,l], Z[l,:])
        res[l] = np.log(1 - temp)
    return 1 - np.exp(res.sum(0))

def clip(m, thre):
    m[m>thre] = thre
    m[m<-thre] = -thre
    return m

@jit('Tuple((float64[:,:], float64[:,:], float64[:,:]))(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64, float64)', nopython=True, nogil=True, parallel=True)
def lr_update(grad_prev, grad, change, lr, lr_min=1e-6, lr_max=1.0, plus_factor=1.2, minus_factor=0.5):
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            sign = grad_prev[i,j] * grad[i,j]
            if sign > 0:
                lr[i,j] = min((lr[i,j] * plus_factor, lr_max))
                change[i,j] = -np.sign(grad[i,j]) * lr[i,j]
            elif sign < 0:
                lr[i,j] = max((lr[i,j] * minus_factor, lr_min))
                change[i,j] = -change[i,j]
                grad[i,j] = 0
            elif sign == 0:
                change[i,j] = -np.sign(grad[i,j]) * lr[i,j]
    return lr, grad, change


@jit('Tuple((float64[:,:], float64[:,:]))(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64, float64)', nopython=True, nogil=True, parallel=True)
def compute_gradient(deg, mut, X, Y, Z, alpha_X, beta_X, alpha_Z, beta_Z):
    '''
    The only difference from the unmasked version is that the gradients computed from masked elements is not taken into account.
    There should be a much more efficint implement, but this is the solution right now for the sake of time.
    deg: the output (DEG)
    mut: the input (Mutation)
    X: relationships between mutations and pathways, latent variable whose gradient is returned
    Y: status of pathway perturbation, intermediate variable
    Z: relationships between pathways and DEGs, latent variable whose gradient is returned
    '''
    
    deg_hat = reconstruct(Y, Z)
    temp = np.zeros((Y.shape[1], Y.shape[0], Z.shape[1]), dtype=np.float64)
    temp1 = (1 - deg / deg_hat)
    
    for l in prange(Y.shape[1]):
        temp_yz = np.outer(Y[:,l], Z[l,:])
        temp[l] = temp_yz / (1 - temp_yz) * temp1
        
    grad_wy = np.empty((Y.shape))
    grad_wz = np.empty((Z.shape))
    grad_wy = temp.sum(2).T * (1 - Y)
    grad_wz = temp.sum(1) * (1 - Z) + (alpha_Z - 1) * (1 - Z) - (beta_Z - 1) * Z
    
    grad_X = np.zeros((mut.shape[1], mut.shape[0], X.shape[1]), dtype=np.float64)
    for l in prange(mut.shape[1]):
        temp_mutx = np.outer(mut[:,l], X[l,:])
        grad_X[l] = temp_mutx / (1 - temp_mutx) * (1 - Y) * grad_wy
    grad_wx = grad_X.sum(1) * (1 - X) + (alpha_X - 1) * (1 - X) - (beta_X - 1) * X
    return grad_wx, grad_wz


def m_step(deg, mut, w_x, w_z, X, Z, alpha_X, beta_X, alpha_Z, beta_Z, max_iter, initial_lr=0.1, lr_min=1e-6, lr_max=1.0, plus_factor=1.2, minus_factor=0.5):
    grad_wx_prev, grad_wz_prev = np.zeros(w_x.shape), np.zeros(w_z.shape)
    lr_wx, lr_wz = np.ones(w_x.shape) * initial_lr, np.ones(w_z.shape) * initial_lr
    change_wx, change_wz = np.zeros(w_x.shape), np.zeros(w_z.shape)
    loss_trace = []
    X_prev = X > 0.5
    Z_prev = Z > 0.5
    
    for i in range(max_iter):
        Y = reconstruct(mut, X)
        Y[Y>0.999] = 0.999
        grad_wx, grad_wz = compute_gradient(deg, mut, X, Y, Z, alpha_X, beta_X, alpha_Z, beta_Z)
        lr_wx, grad_wx_prev, change_wx = lr_update(grad_wx_prev, grad_wx, change_wx, lr_wx, lr_min, lr_max, plus_factor, minus_factor)
        w_x = w_x + change_wx
        lr_wz, grad_wz_prev, change_wz = lr_update(grad_wz_prev, grad_wz, change_wz, lr_wz, lr_min, lr_max, plus_factor, minus_factor)
        w_z = w_z + change_wz
        
        w_x = clip(w_x,10)
        w_z = clip(w_z,10)
        
        X = expit(w_x)
        Z = expit(w_z)

        if (i+1) % 10 == 0:
            temp = reconstruct(reconstruct(mut, X), Z)
            temp[temp > 0.999]  = 0.999
            loss = - deg * np.log(temp) - (1-deg) * np.log(1-temp)
            loss_trace.append(loss.sum())
            print('iteration:', i+1,'--loss:', loss_trace[-1], end='\n')
            X_bool = X > 0.5
            Z_bool = Z > 0.5
            if np.all(X_prev==X_bool) & np.all(Z_prev==Z_bool): break
            X_prev, Z_prev = X_bool, Z_bool
        
    return w_x, w_z, loss_trace

def mut2deg(deg, mut, pathway_size, alpha_X=1.0, beta_X=1.0, alpha_Z=1.0, beta_Z=1.0, max_iter=100):
    '''
    Boolean matrix factorization by using an identity matrix as input and observation as output for a neural network.
    The edge weights learned through the network can then be interpreted as the factor matrixes.
    The lesser size of x will be used as the output.
    deg: the matrix of differential expression, each row should be a sample.
    mut: the matrix of somatic mutations, each row should be a sample. The number of samples should the same as deg.
    pathway_size: the number of pathways.
    '''
    
    w_x = np.random.normal(loc=0, scale=0.1, size=(mut.shape[1], pathway_size))
    w_z = np.random.normal(loc=0, scale=0.1, size=(pathway_size, deg.shape[1]))
    X = expit(w_x)
    Z = expit(w_z)
    w_x, w_z, loss_trace = m_step(deg, mut, w_x, w_z, X, Z, alpha_X, beta_X, alpha_Z, beta_Z, max_iter)
    return expit(w_x), expit(w_z)

def synthetic_mut2deg(n_path=20, n_sample=1000, n_deg=1000, n_mutation=1500, p_mutation=0.01, p_m2p=(0.02,0.05), p_p2d=(0.05,0.1)):
    mut = np.random.binomial(1, p_mutation, (n_sample, n_mutation))
    U = np.zeros((n_mutation, n_path))
    Z = np.zeros((n_path, n_deg))
    prior_U = np.random.uniform(p_m2p[0], p_m2p[1], n_path)
    prior_Z = np.random.uniform(p_p2d[0], p_p2d[1], n_path)
    for i in range(n_path):
        U[:,i] = np.random.binomial(1, prior_U[i], n_mutation)
        Z[i,:] = np.random.binomial(1, prior_Z[i], n_deg)
    path = mut.dot(U)
    # Prune the mutation matrix so that it would be mutually exclusive
    impure = np.where(path>1)
    for i in range(len(impure[0])):
        mut[impure[0][i], U[:,impure[1][i]]==1] = 0
        mut[impure[0][i], np.random.choice(np.where(U[:,impure[1][i]]==1)[0])] = 1
    print(path.sum())
    path[path>1] = 1
    deg = path.dot(Z)
    deg[deg>1] = 1

    return mut, U, path, Z, deg

def prior_est(mut, deg, L):
    '''
    Estimate the prior expected values from binary mutation matrix and binary DEG matrix.
    '''
    p_deg = deg.mean()
    p2d = np.sqrt(1-np.exp(np.log(1-p_deg)/L))
    p_mut = mut.sum(1)
    m2p = np.zeros(len(p_mut))
    for i in range(len(p_mut)):
        m2p[i] = 1 - np.exp(np.log(1-p2d)/p_mut[i])
    m2p = m2p.mean()
    return p2d/(1-p2d), m2p/(1-m2p)



def jaccard_quality(X_hat, Z_hat, X_true, Z_true):
    X_and = X_hat.T.dot(X_true)
    Z_and = Z_hat.dot(Z_true.T)
    
    X_or = np.zeros((X_hat.shape[1], X_hat.shape[1]))
    for i in range(X_hat.shape[1]):
        temp = X_hat.T[i] + X_true.T
        temp[temp > 1] = 1
        X_or[i] = temp.sum(1)
    X_qual = X_and / X_or
    print(X_qual)
    Z_or = np.zeros((Z_hat.shape[0], Z_hat.shape[0]))
    for i in range(Z_hat.shape[0]):
        temp = Z_hat[i] + Z_true
        temp[temp > 1] = 1
        Z_or[i] = temp.sum(1)
    Z_qual = Z_and/Z_or
    print(Z_qual)
    return np.max(X_qual * Z_qual, 0)
    

    

### Synthetic experiments on the OR-propagation
mut1, X_true, path, Z_true, deg1 = synthetic_mut2deg(n_path=5)
beta_Z, beta_X = prior_est(mut1, deg1, 5)
res = mut2deg(deg1.astype(np.float64), mut1.astype(np.float64), 5, 1.0, 0.5, 1.0, 1.0, max_iter=200)
jaccard_quality(res[0]>0.5, res[1]>0.5, X_true, Z_true).mean()







