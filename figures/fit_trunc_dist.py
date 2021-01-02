import numpy as np
from numpy import sqrt, pi, exp
from scipy.special import erf
from scipy.optimize import minimize
from numba import jit, prange

@jit
def fit_mu_sig(x, mean, var):
    mu = np.array(x[0])
    logvar = np.array(x[1])
    
    sigma = np.exp(logvar/2)
    
    testmean = mu - sqrt(2)*sigma*exp(-1/2*(mu - 4)**2/sigma**2)/(sqrt(pi)*(erf(-1/2*sqrt(2)*(mu - 4)/sigma) - 1))
    testvar = sigma**2*(
        sqrt(2)*(mu - 4)*exp(-1/2*(mu - 4)**2/sigma**2)/(sqrt(pi)*sigma*(erf(-1/2*sqrt(2)*(mu - 4)/sigma) - 1))
        - 2*exp(-(mu - 4)**2/sigma**2)/(pi*(erf(-1/2*sqrt(2)*(mu - 4)/sigma) - 1)**2) 
        + 1)
    
    return (testmean - mean)**2 + (testvar - var)**2


@jit(parallel=True)
def find_mu_sig(sample_preds):
    """Samples are passed in as [n_batch, n_samples]. Returns [n_batch, 2] for (mu, sigma) pairs for a distribution cutoff at 4"""
    sample_preds = sample_preds.T
    result = []
    for i in prange(sample_preds.shape[1]):
        to_fit = sample_preds[:, i]
        mean = np.average(to_fit) #=mu + (gauss_pdf(alpha))/Z
        var = np.std(to_fit)**2
        
        x0 = np.array([mean, np.log(var)])
        out = minimize(fit_mu_sig, x0, args=(mean, var), bounds=[(4, 20), (-1, 2)]).x
        
        mu = out[0]
        sig = np.exp(out[1]/2)
        
        result.append([mu, sig])
        
    return np.array(result)

from scipy.stats import truncnorm

@jit
def fit_mu_sig_likelihood(x0, x):
    mu = np.array(x0[0])
    logvar = np.array(x0[1])
    
    std = np.exp(logvar/2)
    a = (4-mu)/std
    b = np.inf

    return truncnorm.nnlf([a, b, mu, std], x)


@jit(parallel=True)
def find_mu_sig_likelihood(sample_preds):
    """Samples are passed in as [n_batch, n_samples]. Returns [n_batch, 2] for (mu, sigma) pairs for a distribution cutoff at 4"""
    sample_preds = sample_preds.T
    result = []
    for i in prange(sample_preds.shape[1]):
        to_fit = sample_preds[:, i]
        mean = np.average(to_fit) #=mu + (gauss_pdf(alpha))/Z
        var = np.std(to_fit)**2
        
        x0 = np.array([mean, np.log(var)])
        out = minimize(fit_mu_sig_likelihood, x0, args=(to_fit), bounds=[(4, 20), (-1, 2)]).x
        
        mu = out[0]
        sig = np.exp(out[1]/2)
        
        result.append([mu, sig])
        
    return np.array(result)
