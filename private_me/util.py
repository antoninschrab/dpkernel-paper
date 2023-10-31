from __future__ import division 
"""A module containing convenient methods for general machine learning
The structure and part of this code was adapted from Wittawat Jitkrittum
Linear-time interpretable nonparametric two-sample test"""
__author__ = 'leon'
import os
import time
import pickle

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import private_me
class ContextTimer(object):
    """
    A class used to time an executation of a code snippet. 
    Use it with with .... as ...
    For example, 

        with ContextTimer() as t:
            # do something 
        time_spent = t.secs

    From https://www.huyng.com/posts/python-performance-analysis
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start 
        if self.verbose:
            print('elapsed time: %f ms' % (self.secs*1000))

# end class ContextTimer

class NumpySeedContext(object):
    """
    A context manager to reset the random seed by numpy.random.seed(..).
    Set the seed back at the end of the block. 
    """
    def __init__(self, seed):
        self.seed = seed 

    def __enter__(self):
        rstate = np.random.get_state()
        self.cur_state = rstate
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.cur_state)

def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X**2, 1)
    sy = np.sum(Y**2, 1)
    D2 =  sx[:, np.newaxis] - 2.0*X.dot(Y.T) + sy[np.newaxis, :] 
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D

def med_sq_distance(all_Xs, n_sub=1000):
    N = all_Xs.shape[0]
    sub = all_Xs[np.random.choice(N, min(n_sub, N), replace=False)]
    D2 = euclidean_distances(sub, squared=True)
    return np.median(D2[np.triu_indices_from(D2, k=1)], overwrite_input=True) / 2.0 #gwidth2 



def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and 
        there are more slightly more 0 than 1.

    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)

def is_real_num(x):
    """return true if x is a real number"""
    try:
        float(x)
        return not (np.isnan(x) or np.isinf(x))
    except ValueError:
        return False

def construct_z(data, test_freqs, gaussian_width):
    """Construct the features Z to be used for testing with T^2 statistics.
    Z is defined in Eq.14 of Chwialkovski et al., 2015 (NIPS). 

    test_freqs: J x d test frequencies
    
    Return a n x 2J numpy array. 2J because of sin and cos for each frequency.
    """
    data = data / gaussian_width
    n, d = data.shape
    J = test_freqs.shape[0]
    # inverse Fourier transform (upto scaling) of the unit-width Gaussian kernel 
    f = np.exp(-np.sum(data**2, 1)/2)[:, np.newaxis]
    # n x J
    data_freq = data.dot(test_freqs.T)
    # zx: n x 2J
    z = np.hstack((np.sin(data_freq)*f, np.cos(data_freq)*f))
    assert z.shape == (n, 2*J)
    return z 

def PSD(A, reg=1e-8, tol=0.0):
    evals, eV = np.linalg.eig(A)
    evals = np.real(evals) #due to numerical error
    eV = np.real(eV)
    if not np.all(evals > tol): #small tolerance allowed
        if isinstance(reg, float) and reg > 0.0:
            ev_small = np.sort(evals[evals > 0])[0]
            evals[evals <= 0] = min(reg, ev_small) #if reg too large
        else:
            raise ValueError('float {} is not positive float'.format(reg))
    psd_A = eV.dot(np.diag(evals)).dot(eV.T) # reconstruction
    return psd_A

def isPSD(A, tol=1e-8): #small tolerance allowed
    evals, eV = np.linalg.eig(A)
    evals = np.real(evals)
    print(evals)
    return np.all(evals > -tol)

def sum_normal_chi(J, sigma, no_samples=50000):
    chi_samples = np.random.chisquare(J, size=no_samples)
    normal_samples = np.random.normal(loc=0.0, scale=sigma, size=no_samples)
    chi_add_normal = chi_samples + normal_samples
    return chi_add_normal

def weight_chi_simulate(list_weights, no_samples=50000):
    J = len(list_weights)
    block_weights = np.tile(list_weights, (no_samples,1))
    chi_samples = np.random.chisquare(1, size=(no_samples, J))
    assert chi_samples.shape == block_weights.shape
    weighted_chi_samples = np.sum(np.multiply(block_weights, chi_samples), axis = 1)
    return weighted_chi_samples

def weight_chi_p_value(statistic, list_weights, no_samples=50000):
    samples = weight_chi_simulate(list_weights, no_samples)
    p_value = float(np.sum(samples > statistic)) / len(samples)
    return p_value

def normal_chi_p_value(statistic, J, sigma, no_samples=50000):
    samples = sum_normal_chi(J, sigma, no_samples)
    p_value = float(np.sum(samples > statistic)) / len(samples)
    return p_value

def tr_te_indices(n, tr_proportion, seed=9282 ):
    """Get two logical vectors for indexing train/test points.

    Return (tr_ind, te_ind)
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion*n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)

def subsample_ind(n, k, seed=28):
    """
    Return a list of indices to choose k out of n without replacement
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    ind = np.random.choice(n, k, replace=False)
    np.random.set_state(rand_state)
    return ind

