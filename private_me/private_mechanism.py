from __future__ import division
# Differential private mech
from math import sqrt, log, floor, exp

import numpy as np
from scipy.special import erf

import private_me.util as util

__author__ = "leon"

def analyse_gauss_mech(quantity, sensitivity, epsilon=0.5, 
                       delta=1e-4, gauss_noise='Normal', seed=23):
	with util.NumpySeedContext(seed=seed+3000):
		shape = quantity.shape
		if shape[0] != shape[1]:
			raiseValueError('Shape of input must be square')
		J = shape[0]
		if gauss_noise == 'Normal':
			sigma = sensitivity * sqrt(2.0 * log(1.25 / delta)) / epsilon
		elif gauss_noise == 'Improved':
			sigma = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)
		# construct symmetric matrix
		print('analyse_gauss_mech sigma {}'.format(sigma))
		eta = np.random.normal( loc=0.0, scale=sigma, size=(J,J))
		diag_upper = np.triu(eta)
		noise = diag_upper + np.tril(diag_upper.T, k=-1)
		private_quantity = quantity + noise
		# make it psd 
		private_PSD = util.PSD(private_quantity, reg=1e-4)
		return private_PSD

def improve_gauss_mech(quantity, sensitivity, epsilon=1.0, 
                       delta=1e-4, return_sigma=False, seed=23):
	with util.NumpySeedContext(seed=seed+1000):
		shape = quantity.shape
		sigma = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)
		print('Improve')
		print('sigma:{}'.format(sigma))
		private_quantity = quantity + np.random.normal(loc=0.0, scale=sigma, size=shape)
	if return_sigma:
		return private_quantity, sigma
	else:
		return private_quantity

def gauss_mech(quantity, sensitivity, epsilon=1.0, delta=1e-4, return_sigma=False, seed=23):
	with util.NumpySeedContext(seed=seed+1000):
		shape = quantity.shape
		sigma = sensitivity * sqrt(2.0 * log(1.25 / delta)) / epsilon
		print('Normal')
		print('sigma:{}'.format(sigma))
		private_quantity = quantity + np.random.normal(loc=0.0, scale=sigma, size=shape)
	if return_sigma:
		return private_quantity, sigma
	else:
		return private_quantity

# B. Balle and Y.-X. Wang. Improving the Gaussian Mechanism for Differential Privacy: 
# Analytical Calibration and Optimal Denoising. International Conference on Machine Learning (ICML), 2018.
# https://github.com/BorjaBalle/analytic-gaussian-mechanism
def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)
    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma