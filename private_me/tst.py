"""Module containing many types of two sample test algorithms
The structure and part of this code was adapted from Wittawat Jitkrittum
Linear-time interpretable nonparametric two-sample test"""
__author__ = "leon"

from abc import ABCMeta, abstractmethod
from math import sqrt

import numpy as np
import scipy.stats as stats
import theano
import theano.tensor as tensor
import theano.tensor.nlinalg as nlinalg
import theano.tensor.slinalg as slinalg
import matplotlib.pyplot as plt

from private_me.data import TSTData
import private_me.util as util
import private_me.kernel as kernel
from private_me.private_mechanism import gauss_mech, improve_gauss_mech, analyse_gauss_mech
from scipy.linalg import block_diag, sqrtm, inv, svd

class TwoSampleTest(object):
    """Abstract class for two sample tests."""
    __metaclass__ = ABCMeta

    def __init__(self, alpha=0.01):
        """
        alpha: significance level of the test
        """
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, tst_data, epsilon=0.5, delta=1e-4, null=False, seed=23):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, tst_data, epsilon=0.5, delta=1e-4, seed=23):
        """Compute the test statistic"""
        raise NotImplementedError()

class PrivateMC_Test(TwoSampleTest):
    """
    A generic mean embedding test using a specified kernel, with noise added onto mean, covariance.
    """
    def __init__(self, test_type, test_locs, k, gwidth2, alpha=0.01):
        """
        :param test_locs: J x d numpy array of J locations to test the difference
        :param k: a instance of Kernel
        """
        super(PrivateMC_Test, self).__init__(alpha)
        self.test_type = test_type
        self.gwidth2 = gwidth2
        self.test_locs = test_locs
        self.k = k

    def perform_test(self, tst_data, epsilon=0.5, delta=1e-4, null='private', 
                     noise='analyse_gauss', gauss_noise='Normal', mean_noise_prop=0.5, seed=23):
        stat = self.compute_stat(tst_data, epsilon=epsilon, delta=delta, gauss_noise=gauss_noise,
                                 noise=noise, mean_noise_prop=mean_noise_prop, 
                                 seed=seed)
        if self.test_type == 'ME':
            J = self.test_locs.shape[0]
            kappa = self.k.kappa
        elif self.test_type == 'SCF':
            J = self.test_locs.shape[0] * 2
            kappa = 1.0
        alpha = self.alpha
        if null in ['private', 'true']:
            n = tst_data.len_x()
            Sig_pri_pd_inv = inv(self.Sig_private_pd)
            U, s, Vh = svd(Sig_pri_pd_inv)
            S = np.diag(np.sqrt(s))
            Sig_pri_pd_inv_sqrt = np.dot(U, np.dot(S, Vh))
            V = self.sigma**2 * np.eye(J)
            if null == 'private':
                mid = self.Sig_private + n * V
            else:
                mid = self.Sig + n * V
            cov_null = np.matmul(np.matmul(Sig_pri_pd_inv_sqrt, mid), Sig_pri_pd_inv_sqrt)
            evals, eV = np.linalg.eig(cov_null)
            evals = np.real(evals)
            check_evals, check_eV = np.linalg.eig(np.matmul(Sig_pri_pd_inv, mid))
            check_2evals, check_2eV = np.linalg.eig(self.Sig_private)
            evals[evals < 0] = 0 # numerical instability
            pvalue = util.weight_chi_p_value(stat, evals, no_samples=100000)
        elif null == 'asymptotic':
            pvalue = stats.chi2.sf(stat, J)
        else:
            raiseValueError('Please specify a correct null formulation.')
        results = {'alpha': self.alpha, 'epsilon': epsilon, 'delta': delta,
                   'null': null, 'pvalue': pvalue, 'test_stat': stat,
                   'h0_rejected': pvalue < alpha, 'kappa': kappa,
                   'mean_noise_prop': mean_noise_prop, 'noise': noise}
        return results

    def compute_stat(self, tst_data, epsilon=0.5, delta=1e-4, noise='analyse_gauss', 
                     gauss_noise='Normal', mean_noise_prop=0.5, seed=23):
        assert 0 < mean_noise_prop < 1.0
        lambda_noise_prop = 1.0 - mean_noise_prop
        mean_epsilon = epsilon * mean_noise_prop # mean will be epsilon/2.0 private
        lambda_epsilon = epsilon * lambda_noise_prop
        mean_delta = delta * mean_noise_prop
        lambda_delta = delta * lambda_noise_prop
        if self.test_locs is None: 
            raise ValueError('test_locs must be specified.')
        X, Y = tst_data.xy()
        test_locs = self.test_locs
        if self.test_type == 'ME':
            k = self.k
            kappa = k.kappa
            g = k.eval(X, test_locs)
            h = k.eval(Y, test_locs)
        elif self.test_type == 'SCF':
            g = util.construct_z(X, test_locs, self.gwidth2)
            h = util.construct_z(Y, test_locs, self.gwidth2)
            kappa = 1.0
        Z = g-h 
        n, J = Z.shape
        W = np.mean(Z, 0, keepdims=True).T 
        sensitivity = kappa * sqrt(J) / n
        if gauss_noise == 'Normal':
            W_private, self.sigma = gauss_mech(W, sensitivity, epsilon=mean_epsilon, 
                                               delta=mean_delta, return_sigma=True, 
                                               seed=seed)
        elif gauss_noise == 'Improved':
            W_private, self.sigma = improve_gauss_mech(W, sensitivity, epsilon=mean_epsilon, 
                                                       delta=mean_delta, return_sigma=True, 
                                                       seed=seed)
        else:
            raise 'gauss_noise: {} not recognised'.format(gauss_noise)
        Lambda = np.matmul(Z.T, Z) / n # can change to n-1
        if noise == 'analyse_gauss':
            sensitivity = (kappa ** 2) * J / (n - 1.0)
            Lambda_private = analyse_gauss_mech(Lambda, sensitivity, epsilon=lambda_epsilon, gauss_noise=gauss_noise, 
                                                delta=lambda_delta, seed=seed+222)
        else:
            raise 'Noise mech not recognised'
        self.Sig_private = util.PSD(Lambda_private - np.matmul(W_private, W_private.T), reg=1e-4)
        self.Sig = Lambda - np.matmul(W, W.T)
        s_private, self.Sig_private_pd = nc_parameter(n, W_private, self.Sig_private, return_sig=True, reg='auto')
        return s_private

class PrivateSt_Test(TwoSampleTest):
    """
    A generic mean embedding test using a specified kernel, with noise added onto statistic.
    """
    def __init__(self, test_type, test_locs, k, gwidth2, reg, alpha=0.01):
        """
        :param test_locs: J x d numpy array of J locations to test the difference
        :param k: a instance of Kernel
        """
        super(PrivateSt_Test, self).__init__(alpha)
        self.test_type = test_type
        self.test_locs = test_locs
        self.k = k
        self.gwidth2 = gwidth2
        self.reg = reg

    def perform_test(self, tst_data, epsilon=0.5, delta=1e-4, 
                     gauss_noise='Normal', null='private', seed=23):
        n = tst_data.len_x()
        if self.test_type == 'ME':
            J = self.test_locs.shape[0]
            kappa = self.k.kappa
        elif self.test_type == 'SCF':
            J = self.test_locs.shape[0] * 2
            kappa = 1.0
        stat = self.compute_stat(tst_data, epsilon=epsilon, delta=delta, 
                                 gauss_noise=gauss_noise, seed=seed)
        if null in ['private', 'true']:
            pvalue = util.normal_chi_p_value(stat, J, self.sigma, no_samples=100000)
        elif null == 'asymptotic':
            pvalue = stats.chi2.sf(stat, J)
        else:
            raiseValueError('Please specify the correct null formulation.')
        alpha = self.alpha
        results = {'alpha': alpha, 'epsilon': epsilon, 'delta': delta,
                   'null': null, 'pvalue': pvalue, 'test_stat': stat,
                   'h0_rejected': pvalue < alpha, 'kappa': kappa}
        return results

    def compute_stat(self, tst_data, epsilon=0.5, delta=1e-4, gauss_noise='Normal', seed=23):
        if self.test_locs is None: 
            raise ValueError('test_locs must be specified.')
        X, Y = tst_data.xy()
        test_locs = self.test_locs
        if self.test_type == 'ME':
            k = self.k
            kappa = k.kappa
            g = k.eval(X, test_locs)
            h = k.eval(Y, test_locs)
        elif self.test_type == 'SCF':
            g = util.construct_z(X, test_locs, self.gwidth2)
            h = util.construct_z(Y, test_locs, self.gwidth2)
            kappa = 1.0
        Z = g-h 
        n, J = Z.shape
        W = np.mean(Z, 0, keepdims=True).T
        Lambda = np.matmul(Z.T, Z) / n
        if self.reg == 'auto':
            self.reg = 0.00001
        Lambda_stable = Lambda + self.reg*np.eye(Lambda.shape[0])
        Sig = Lambda_stable - np.matmul(W, W.T)
        s = nc_parameter(n, W, Sig) # should be fine!
        sensitivity = self.sensitivity_lvl(kappa, J, n, self.reg)
        if gauss_noise == 'Normal':
            s_private, self.sigma = gauss_mech(s, sensitivity, epsilon=epsilon, 
                                               delta=delta, return_sigma=True, 
                                               seed=seed)
        elif gauss_noise == 'Improved':
            s_private, self.sigma = improve_gauss_mech(s, sensitivity, epsilon=epsilon, 
                                   delta=delta, return_sigma=True, 
                                   seed=seed)
        else:
            raise 'Gauss noise not recognised'
        return s_private 

    @staticmethod
    def sensitivity_lvl(kappa, J, n, reg):
        B_square = float(kappa**2 * J) / n
        print('reg: {}, J: {}, kappa: {}, n: {}'.format(reg, J, kappa, n))
        constants = ( 4.0 * kappa**2 * J**(3.0/2.0) )  / n
        sensitivity = constants * ((B_square + 1.0) / reg)
        print('sensitivity', sensitivity)
        return sensitivity

class PrivateLocal_PairTest(TwoSampleTest):
    """
    A generic mean embedding (ME) test using a specified kernel
    with noise added onto mean and covariance seperately in a local setting
    Uses a pair test.
    """

    def __init__(self, test_type, test_locs, k, gwidth2, alpha=0.01):
        """
        :param test_locs: J x d numpy array of J locations to test the difference
        :param k: a instance of Kernel
        """
        super(PrivateLocal_PairTest, self).__init__(alpha)
        self.test_type = test_type
        self.test_locs = test_locs
        self.k = k
        self.gwidth2 = gwidth2

    def perform_test(self, tst_data, epsilon=0.5, delta=1e-4, null='private', 
                     noise='analyse_gauss', gauss_noise='Normal', 
                     mean_noise_prop=0.5, seed=23):
        stat = self.compute_stat(tst_data, epsilon=epsilon, delta=delta, 
                                 noise=noise, mean_noise_prop=mean_noise_prop, 
                                 gauss_noise=gauss_noise, seed=seed)
        if self.test_type == 'ME':
            J = self.test_locs.shape[0]
            kappa = self.k.kappa
        elif self.test_type == 'SCF':
            J = self.test_locs.shape[0] * 2
            kappa = 1.0
        if null in ['private','true']:
            n_x = tst_data.len_x()
            n_y = tst_data.len_y()
            constant = float(n_x*n_y) / (n_x + n_y)
            Sig_pri_pd_inv = inv(self.Sig_private_pd)
            U, s, Vh = svd(Sig_pri_pd_inv)
            S = np.diag(np.sqrt(s))
            Sig_pri_pd_inv_sqrt = np.dot(U, np.dot(S, Vh))
            V_x = self.sigma_x**2 * np.eye(J)
            V_y = self.sigma_y**2 * np.eye(J)
            if null == 'private':
                mid = self.Sig_x_private / n_x + self.Sig_y_private / n_y + V_x + V_y 
            else:
                mid = self.Sig_x / n_x + self.Sig_y / n_y + V_x + V_y
            cov_null = constant * np.matmul(np.matmul(Sig_pri_pd_inv_sqrt, mid), Sig_pri_pd_inv_sqrt)
            evals, eV = np.linalg.eig(cov_null)
            evals = np.real(evals)
            print(evals)
            evals[evals < 0] = 0 # numerical instability
            pvalue = util.weight_chi_p_value(stat, evals, no_samples=100000)
        elif null == 'asymptotic':
            pvalue = stats.chi2.sf(stat, J)
        else:
            raiseValueError('Please specify correct null formulation')
        alpha = self.alpha
        results = {'alpha': self.alpha, 'epsilon': epsilon, 'delta': delta,
                   'null': null, 'pvalue': pvalue, 'test_stat': stat,
                   'h0_rejected': pvalue < alpha, 'kappa': kappa}
        return results

    def compute_stat(self, tst_data, epsilon=0.5, delta=1e-4, 
                     noise='analyse_gauss', mean_noise_prop=0.5, 
                     gauss_noise='Normal', seed=23):
        if self.test_locs is None:
            raise ValueError('test_locs must be specified.')
        X, Y = tst_data.xy()
        test_locs = self.test_locs
        if self.test_type == 'ME':
            k = self.k
            kappa = k.kappa
            g = k.eval(X, test_locs)
            h = k.eval(Y, test_locs)
        elif self.test_type == 'SCF':
            g = util.construct_z(X, test_locs, self.gwidth2)
            h = util.construct_z(Y, test_locs, self.gwidth2)
            kappa = 1.0
        n_x, W_x_private, Sig_x_private, self.sigma_x, self.Sig_x = self.local_mean_cov(
                                                              g, kappa, epsilon=epsilon, gauss_noise=gauss_noise,
                                                              delta=delta, mean_noise_prop=mean_noise_prop, 
                                                              noise=noise, seed=seed)
        n_y, W_y_private, Sig_y_private, self.sigma_y, self.Sig_y = self.local_mean_cov(
                                                              h, kappa, epsilon=epsilon, gauss_noise=gauss_noise,
                                                              delta=delta, mean_noise_prop=mean_noise_prop, 
                                                              noise=noise, seed=seed+4848)
        self.Sig_private = ((n_x - 1.0) * Sig_x_private + (n_y - 1.0) * Sig_y_private) / (n_x + n_y - 2.0)
        s, self.Sig_private_pd = nc_parameter_pair(n_x, n_y, W_x_private, W_y_private, 
                                                   self.Sig_private, return_sig=True, reg='auto')
        self.Sig_x_private = Sig_x_private
        self.Sig_y_private = Sig_y_private
        return s

    @staticmethod # TODO: Optimise this
    def local_mean_cov(data_k, kappa, epsilon=0.5, delta=1e-4, gauss_noise='Normal',
                       mean_noise_prop=0.5, noise='wishart', seed=23):
        assert 0 < mean_noise_prop < 1.0
        lambda_noise_prop = 1.0 - mean_noise_prop
        mean_epsilon = epsilon * mean_noise_prop 
        lambda_epsilon = epsilon * lambda_noise_prop
        mean_delta = delta * mean_noise_prop
        lambda_delta = delta * lambda_noise_prop
        n, J = data_k.shape
        W = np.mean(data_k, 0, keepdims=True).T
        Lambda = np.matmul(data_k.T, data_k) / n
        sensitivity = kappa * sqrt(J) / n
        if gauss_noise == 'Normal':
            W_private, sigma = gauss_mech(W, sensitivity, epsilon=mean_epsilon,
                                        delta=mean_delta, return_sigma=True, seed=seed)
        elif gauss_noise == 'Improved':
            W_private, sigma = improve_gauss_mech(W, sensitivity, epsilon=mean_epsilon,
                            delta=mean_delta, return_sigma=True, seed=seed)
        else:
            raise 'Gauss noise not recognised'
        if noise == 'analyse_gauss':
            sensitivity = (kappa ** 2) * J / (n - 1)
            Lambda_private = analyse_gauss_mech(Lambda, sensitivity, epsilon=lambda_epsilon, gauss_noise=gauss_noise, delta=lambda_delta, seed=seed+222)
        else:
            raise 'Noise mech not recognised'
        Sig_private = Lambda_private - np.matmul(W_private, W_private.T)
        Sig = Lambda - np.matmul(W, W.T)
        Sig_private = util.PSD(Sig_private, reg=1e-4) # Make PD
        return n, W_private, Sig_private, sigma, Sig

class Local_PairTest(TwoSampleTest):
    """
    A generic mean embedding (ME) test using a specified kernel
    in a local setting, uses a pair test.
    """
    def __init__(self, test_locs, k, alpha=0.01):
        """
        :param test_locs: J x d numpy array of J locations to test the difference
        :param k: a instance of Kernel
        """
        super(Local_PairTest, self).__init__(alpha)
        self.test_locs = test_locs
        self.k = k

    def perform_test(self, tst_data, epsilon=0.5, delta=1e-4, null='private', 
                     gauss_noise='Normal', noise='wishart', mean_noise_prop=0.5, seed=23):
        stat = self.compute_stat(tst_data, seed=seed)
        J, d = self.test_locs.shape
        pvalue = stats.chi2.sf(stat, J)
        alpha = self.alpha
        results = {'alpha': self.alpha, 'epsilon': epsilon, 'delta': delta,
                   'null': null, 'pvalue': pvalue, 'test_stat': stat,
                   'h0_rejected': pvalue < alpha, 'kappa': self.k.kappa}
        return results

    def compute_stat(self, tst_data, seed=23):
        if self.test_locs is None: 
            raise ValueError('test_locs must be specified.')
        X, Y = tst_data.xy()
        test_locs = self.test_locs
        k = self.k
        J, d = test_locs.shape
        g = k.eval(X, test_locs)
        h = k.eval(Y, test_locs)
        n_x = g.shape[0]
        n_y = h.shape[0]
        W_x = np.mean(g, 0, keepdims=True).T
        W_y = np.mean(h, 0, keepdims=True).T
        Lambda_x = np.matmul(g.T, g) / n_x
        Lambda_y = np.matmul(h.T, h) / n_y
        Sig_x = Lambda_x - np.matmul(W_x, W_x.T)
        Sig_y = Lambda_y - np.matmul(W_y, W_y.T)
        Sig = ((n_x - 1.0) * Sig_x + (n_y - 1.0) * Sig_y) / (n_x + n_y - 2.0)
        s = nc_parameter_pair(n_x, n_y, W_x, W_y, Sig, return_sig=False, reg='auto')
        return s

class SmoothCFTest(TwoSampleTest):
    """Class for two-sample test using smooth characteristic functions.
    Use Gaussian kernel."""
    def __init__(self, test_freqs, gaussian_width, alpha=0.01):
        """
        :param test_freqs: J x d numpy array of J frequencies to test the difference
        gaussian_width: The width is used to divide the data. The test will be 
            equivalent if the data is divided beforehand and gaussian_width=1.
        """
        super(SmoothCFTest, self).__init__(alpha)
        self.test_freqs = test_freqs
        self.gaussian_width = gaussian_width

    @property
    def gaussian_width(self):
        # Gaussian width. Positive number.
        return self._gaussian_width
    
    @gaussian_width.setter
    def gaussian_width(self, width):
        if util.is_real_num(width) and float(width) > 0:
            self._gaussian_width = float(width)
        else:
            raise ValueError('gaussian_width must be a float > 0. Was %s'%(str(width)))

    def compute_stat(self, tst_data):
        # test freqs or Gaussian width undefined 
        if self.test_freqs is None: 
            raise ValueError('test_freqs must be specified.')

        X, Y = tst_data.xy()
        test_freqs = self.test_freqs
        gamma = self.gaussian_width
        s = SmoothCFTest.compute_nc_parameter(X, Y, test_freqs, gamma)
        return s

    def perform_test(self, tst_data, epsilon=None, delta=None, gauss_noise='Normal', null=None, seed=None):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        stat = self.compute_stat(tst_data)
        J, d = self.test_freqs.shape
        # 2J degrees of freedom because of sin and cos
        pvalue = stats.chi2.sf(stat, 2*J)
        alpha = self.alpha
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': stat,
                'h0_rejected': pvalue < alpha}
        return results

    #---------------------------------
    @staticmethod
    def compute_nc_parameter(X, Y, T, gwidth, reg='auto'):
        """
        Compute the non-centrality parameter of the non-central Chi-squared 
        which is the distribution of the test statistic under the H_1 (and H_0).
        The nc parameter is also the test statistic. 
        """
        if gwidth is None or gwidth <= 0:
            raise ValueError('require gaussian_width > 0. Was %s'%(str(gwidth)))

        Z = SmoothCFTest.construct_z(X, Y, T, gwidth)
        s = generic_nc_parameter(Z, reg)
        return s

    @staticmethod
    def grid_search_gwidth(tst_data, T, list_gwidth, alpha):
        """
        Linear search for the best Gaussian width in the list that maximizes 
        the test power, fixing the test locations ot T. 
        The test power is given by the CDF of a non-central Chi-squared 
        distribution.
        return: (best width index, list of test powers)
        """
        func_nc_param = SmoothCFTest.compute_nc_parameter
        J = T.shape[0]
        return generic_grid_search_gwidth(tst_data, T, 2*J, list_gwidth, alpha,
                func_nc_param)
            

    @staticmethod
    def create_randn(tst_data, J, alpha=0.01, seed=19):
        """Create a SmoothCFTest whose test frequencies are drawn from 
        the standard Gaussian """

        rand_state = np.random.get_state()
        np.random.seed(seed)

        gamma = tst_data.mean_std()*tst_data.dim()**0.5

        d = tst_data.dim()
        T = np.random.randn(J, d)
        np.random.set_state(rand_state)
        scf_randn = SmoothCFTest(T, gamma, alpha=alpha)
        return scf_randn

    @staticmethod 
    def construct_z(X, Y, test_freqs, gaussian_width):
        """Construct the features Z to be used for testing with T^2 statistics.
        Z is defined in Eq.14 of Chwialkovski et al., 2015 (NIPS). 

        test_freqs: J x d test frequencies
        
        Return a n x 2J numpy array. 2J because of sin and cos for each frequency.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Sample size n must be the same for X and Y.')
        X = X/gaussian_width
        Y = Y/gaussian_width 
        n, d = X.shape
        J = test_freqs.shape[0]
        # inverse Fourier transform (upto scaling) of the unit-width Gaussian kernel 
        fx = np.exp(-np.sum(X**2, 1)/2)[:, np.newaxis]
        fy = np.exp(-np.sum(Y**2, 1)/2)[:, np.newaxis]
        # n x J
        x_freq = X.dot(test_freqs.T)
        y_freq = Y.dot(test_freqs.T)
        # zx: n x 2J
        zx = np.hstack((np.sin(x_freq)*fx, np.cos(x_freq)*fx))
        zy = np.hstack((np.sin(y_freq)*fy, np.cos(y_freq)*fy))
        z = zx-zy
        assert z.shape == (n, 2*J)
        return z

    @staticmethod 
    def construct_z_theano(Xth, Yth, Tth, gwidth_th):
        """Construct the features Z to be used for testing with T^2 statistics.
        Z is defined in Eq.14 of Chwialkovski et al., 2015 (NIPS). 
        Theano version.
        
        Return a n x 2J numpy array. 2J because of sin and cos for each frequency.
        """
        Xth = Xth/gwidth_th
        Yth = Yth/gwidth_th 
        # inverse Fourier transform (upto scaling) of the unit-width Gaussian kernel 
        fx = tensor.exp(-(Xth**2).sum(1)/2).reshape((-1, 1))
        fy = tensor.exp(-(Yth**2).sum(1)/2).reshape((-1, 1))
        # n x J
        x_freq = Xth.dot(Tth.T)
        y_freq = Yth.dot(Tth.T)
        # zx: n x 2J
        zx = tensor.concatenate([tensor.sin(x_freq)*fx, tensor.cos(x_freq)*fx], axis=1)
        zy = tensor.concatenate([tensor.sin(y_freq)*fy, tensor.cos(y_freq)*fy], axis=1)
        z = zx-zy
        return z

    @staticmethod
    def optimize_freqs_width(tst_data, alpha, n_test_locs=10, max_iter=400,
            locs_step_size=0.2, gwidth_step_size=0.01, batch_proportion=1.0,
            tol_fun=1e-3, seed=1):
        print('n_test_locs', n_test_locs)
        """Optimize the test frequencies and the Gaussian kernel width by 
        maximizing the test power. X, Y should not be the same data as used 
        in the actual test (i.e., should be a held-out set). 

        - max_iter: #gradient descent iterations
        - batch_proportion: (0,1] value to be multipled with nx giving the batch 
            size in stochastic gradient. 1 = full gradient ascent.
        - tol_fun: termination tolerance of the objective value
        
        Return (test_freqs, gaussian_width, info)
        """
        J = n_test_locs
        """
        Optimize the empirical version of Lambda(T) i.e., the criterion used 
        to optimize the test locations, for the test based 
        on difference of mean embeddings with Gaussian kernel. 
        Also optimize the Gaussian width.

        :return a theano function T |-> Lambda(T)
        """
        d = tst_data.dim()
        # set the seed
        rand_state = np.random.get_state()
        np.random.seed(seed)

        # draw frequencies randomly from the standard Gaussian. 
        # TODO: Can we do better?
        T0 = np.random.randn(J, d)
        # reset the seed back to the original
        np.random.set_state(rand_state)

        # grid search to determine the initial gwidth
        mean_sd = tst_data.mean_std()
        scales = 2.0**np.linspace(-4, 4, 20)
        list_gwidth = np.hstack( (mean_sd*scales*(d**0.5), 2**np.linspace(-7, 7, 20) ))
        list_gwidth.sort()
        besti, powers = SmoothCFTest.grid_search_gwidth(tst_data, T0,
                list_gwidth, alpha)
        # initialize with the best width from the grid search
        gwidth0 = list_gwidth[besti]
        assert util.is_real_num(gwidth0), 'gwidth0 not real. Was %s'%str(gwidth0)
        assert gwidth0 > 0, 'gwidth0 not positive. Was %.3g'%gwidth0

        func_z = SmoothCFTest.construct_z_theano
        # info = optimization info 
        T, gamma, info = optimize_T_gaussian_width(tst_data, T0, gwidth0, func_z, 
                max_iter=max_iter, T_step_size=locs_step_size, 
                gwidth_step_size=gwidth_step_size, batch_proportion=batch_proportion,
                tol_fun=tol_fun)
        assert util.is_real_num(gamma), 'gamma is not real. Was %s' % str(gamma)

        ninfo = {'test_freqs': info['Ts'], 'test_freqs0': info['T0'], 
                'gwidths': info['gwidths'], 'obj_values': info['obj_values'],
                'gwidth0': gwidth0, 'gwidth0_powers': powers}
        return (T, gamma, ninfo  )

    @staticmethod
    def optimize_gwidth(tst_data, T, gwidth0, max_iter=400, 
            gwidth_step_size=0.1, batch_proportion=1.0, tol_fun=1e-3):
        """Optimize the Gaussian kernel width by 
        maximizing the test power, fixing the test frequencies to T. X, Y should
        not be the same data as used in the actual test (i.e., should be a
        held-out set). 

        - max_iter: #gradient descent iterations
        - batch_proportion: (0,1] value to be multipled with nx giving the batch 
            size in stochastic gradient. 1 = full gradient ascent.
        - tol_fun: termination tolerance of the objective value
        
        Return (gaussian_width, info)
        """

        func_z = SmoothCFTest.construct_z_theano
        # info = optimization info 
        gamma, info = optimize_gaussian_width(tst_data, T, gwidth0, func_z, 
                max_iter=max_iter, gwidth_step_size=gwidth_step_size,
                batch_proportion=batch_proportion, tol_fun=tol_fun)

        ninfo = {'test_freqs': T, 'gwidths': info['gwidths'], 'obj_values':
                info['obj_values']}
        return ( gamma, ninfo  )


class METest(TwoSampleTest):
    """
    A generic mean embedding (ME) test using a specified kernel.
    """
    def __init__(self, test_locs, k, alpha=0.01):
        """
        :param test_locs: J x d numpy array of J locations to test the difference
        :param k: a instance of Kernel
        """
        super(METest, self).__init__(alpha)
        self.test_locs = test_locs
        self.k = k

    def perform_test(self, tst_data, epsilon=0.5, delta=1e-4, gauss_noise='Normal', null='asymptotic', seed=23):
        stat = self.compute_stat(tst_data)
        J, d = self.test_locs.shape
        pvalue = stats.chi2.sf(stat, J)
        alpha = self.alpha
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': stat,
                'h0_rejected': pvalue < alpha}
        return results

    def compute_stat(self, tst_data, epsilon=0.5, delta=1e-4):
        if self.test_locs is None: 
            raise ValueError('test_locs must be specified.')

        X, Y = tst_data.xy()
        test_locs = self.test_locs
        k = self.k
        g = k.eval(X, test_locs)
        h = k.eval(Y, test_locs)
        Z = g-h
        n = Z.shape[0]
        W = np.mean(Z, 0, keepdims=True).T
        Lambda = np.matmul(Z.T, Z) / n 
        Sig = Lambda - np.matmul(W, W.T)
        s = nc_parameter(n, W, Sig, reg='auto')
        return s

#-------------------------------------------------
class MeanEmbeddingTest(TwoSampleTest):
    """Class for two-sample test using squared difference of mean embeddings. 
    Use Gaussian kernel."""

    def __init__(self, test_locs, gaussian_width, alpha=0.01):
        """
        :param test_locs: J x d numpy array of J locations to test the difference
        gaussian_width: The width is used to divide the data. The test will be 
            equivalent if the data is divided beforehand and gaussian_width=1.
        """
        # intialise the parent class with siginficance alpaha 
        super(MeanEmbeddingTest, self).__init__(alpha)

        self.test_locs = test_locs
        self.gaussian_width = gaussian_width

    @property
    def gaussian_width(self):
        # Gaussian width. Positive number.
        return self._gaussian_width
    
    @gaussian_width.setter
    def gaussian_width(self, width):
        if util.is_real_num(width) and float(width) > 0:
            self._gaussian_width = float(width)
        else:
            raise ValueError('gaussian_width must be a float > 0. Was %s'%(str(width)))

    def perform_test(self, tst_data):
        stat = self.compute_stat(tst_data)
        #print('stat: %.3g'%stat)
        J, d = self.test_locs.shape
        pvalue = stats.chi2.sf(stat, J)
        alpha = self.alpha
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': stat,
                'h0_rejected': pvalue < alpha}
        return results

    def compute_stat(self, tst_data):
        # test locations or Gaussian width undefined 
        if self.test_locs is None: 
            raise ValueError('test_locs must be specified.')

        X, Y = tst_data.xy()
        test_locs = self.test_locs
        gamma = self.gaussian_width
        stat = MeanEmbeddingTest.compute_nc_parameter(X, Y, test_locs, gamma)
        return stat

    def visual_test(self, tst_data):
        results = self.perform_test(tst_data)
        s = results['test_stat']
        pval = results['pvalue']
        J = self.test_locs.shape[0]
        domain = np.linspace(stats.chi2.ppf(0.001, J), stats.chi2.ppf(0.9999, J), 200)
        plt.plot(domain, stats.chi2.pdf(domain, J), label='$\chi^2$ (df=%d)'%J)
        plt.stem([s], [stats.chi2.pdf(J, J)/2], 'or-', label='test stat')
        plt.legend(loc='best', frameon=True)
        plt.title('%s. p-val: %.3g. stat: %.3g'%(type(self).__name__, pval, s))
        plt.show()

    #===============================
    @staticmethod
    def compute_nc_parameter(X, Y, T, gwidth, reg='auto'):
        """
        Compute the non-centrality parameter of the non-central Chi-squared 
        which is the distribution of the test statistic under the H_1 (and H_0).
        The nc parameter is also the test statistic. 
        """
        if gwidth is None or gwidth <= 0:
            raise ValueError('require gaussian_width > 0. Was %s.'%(str(gwidth)))
        n = X.shape[0]
        g = MeanEmbeddingTest.gauss_kernel(X, T, gwidth)
        h = MeanEmbeddingTest.gauss_kernel(Y, T, gwidth)
        Z = g-h
        s = generic_nc_parameter(Z, reg)
        return s


    @staticmethod 
    def construct_z_theano(Xth, Yth, T, gaussian_width):
        """Construct the features Z to be used for testing with T^2 statistics.
        Z is defined in Eq.12 of Chwialkovski et al., 2015 (NIPS). 

        T: J x d test locations
        
        Return a n x J numpy array. 
        """
        g = MeanEmbeddingTest.gauss_kernel_theano(Xth, T, gaussian_width)
        h = MeanEmbeddingTest.gauss_kernel_theano(Yth, T, gaussian_width)
        # Z: nx x J
        Z = g-h
        return Z

    @staticmethod
    def gauss_kernel(X, test_locs, gwidth2):
        """Compute a X.shape[0] x test_locs.shape[0] Gaussian kernel matrix 
        """
        n, d = X.shape
        D2 = np.sum(X**2, 1)[:, np.newaxis] - 2*X.dot(test_locs.T) + np.sum(test_locs**2, 1)
        K = np.exp(-D2/(2.0*gwidth2))
        return K

    @staticmethod
    def gauss_kernel_theano(X, test_locs, gwidth2):
        """Gaussian kernel for the two sample test. Theano version.
        :return kernel matrix X.shape[0] x test_locs.shape[0]
        """
        T = test_locs
        n, d = X.shape

        D2 = (X**2).sum(1).reshape((-1, 1)) - 2*X.dot(T.T) + tensor.sum(T**2, 1).reshape((1, -1))
        K = tensor.exp(-D2/(2.0*gwidth2))
        return K

    @staticmethod
    def create_fit_gauss_heuristic(tst_data, n_test_locs, alpha=0.01, seed=1):
        """Construct a MeanEmbeddingTest where test_locs are drawn from  Gaussians
        fitted to the data x, y.
        """
        #if cov_xy.ndim == 0:
        #    # 1d dataset. 
        #    cov_xy = np.array([[cov_xy]])
        X, Y = tst_data.xy()
        T = MeanEmbeddingTest.init_locs_2randn(tst_data, n_test_locs, seed)

        # Gaussian (asymmetric) kernel width is set to the average standard
        # deviations of x, y
        #gamma = tst_data.mean_std()*(tst_data.dim()**0.5)
        gwidth2 = util.med_sq_distance(tst_data.stack_xy(), 1000)
        
        met = MeanEmbeddingTest(test_locs=T, gaussian_width=gwidth2, alpha=alpha)
        return met

    @staticmethod
    def optimize_locs_width(tst_data, alpha, n_test_locs=10, max_iter=400, 
            locs_step_size=0.1, gwidth_step_size=0.01, batch_proportion=1.0, 
            tol_fun=1e-3, seed=1):
        """Optimize the test locations and the Gaussian kernel width by 
        maximizing the test power. X, Y should not be the same data as used 
        in the actual test (i.e., should be a held-out set). 

        - max_iter: #gradient descent iterations
        - batch_proportion: (0,1] value to be multipled with nx giving the batch 
            size in stochastic gradient. 1 = full gradient ascent.
        - tol_fun: termination tolerance of the objective value
        
        Return (test_locs, gaussian_width, info)
        """
        J = n_test_locs
        """
        Optimize the empirical version of Lambda(T) i.e., the criterion used 
        to optimize the test locations, for the test based 
        on difference of mean embeddings with Gaussian kernel. 
        Also optimize the Gaussian width.

        :return a theano function T |-> Lambda(T)
        """

        med = util.med_sq_distance(tst_data.stack_xy(), 1000)
        T0 = MeanEmbeddingTest.init_locs_2randn(tst_data, n_test_locs,
                subsample=10000, seed=seed)
        #T0 = MeanEmbeddingTest.init_check_subset(tst_data, n_test_locs, med**2,
        #      n_cand=30, seed=seed+10)
        func_z = MeanEmbeddingTest.construct_z_theano
        # Use grid search to initialize the gwidth
        list_gwidth2 = np.hstack( ( (med**2) *(2.0**np.linspace(-3, 4, 30) ) ) )
        list_gwidth2.sort()
        besti, powers = MeanEmbeddingTest.grid_search_gwidth(tst_data, T0,
                list_gwidth2, alpha)
        gwidth0 = list_gwidth2[besti]
        assert util.is_real_num(gwidth0), 'gwidth0 not real. Was %s'%str(gwidth0)
        assert gwidth0 > 0, 'gwidth0 not positive. Was %.3g'%gwidth0

        # info = optimization info
        T, gamma, info = optimize_T_gaussian_width(tst_data, T0, gwidth0, func_z, 
                max_iter=max_iter, T_step_size=locs_step_size, 
                gwidth_step_size=gwidth_step_size, batch_proportion=batch_proportion,
                tol_fun=tol_fun)
        assert util.is_real_num(gamma), 'gamma is not real. Was %s' % str(gamma)

        ninfo = {'test_locs': info['Ts'], 'test_locs0': info['T0'], 
                'gwidths': info['gwidths'], 'obj_values': info['obj_values'],
                'gwidth0': gwidth0, 'gwidth0_powers': powers}
        return (T, gamma, ninfo  )


    @staticmethod
    def init_check_subset(tst_data, n_test_locs, gwidth2, n_cand=20, subsample=2000,
            seed=3):
        """
        Evaluate a set of locations to find the best locations to initialize. 
        The location candidates are randomly drawn subsets of n_test_locs vectors.
        - subsample the data when computing the objective 
        - n_cand: number of times to draw from the joint and the product 
            of the marginals.
        Return V, W
        """

        X, Y = tst_data.xy()
        n = X.shape[0]

        # from the joint 
        objs = np.zeros(n_cand)
        seed_seq_joint = util.subsample_ind(7*n_cand, n_cand, seed=seed*5)
        for i in range(n_cand):
            V = MeanEmbeddingTest.init_locs_subset(tst_data, n_test_locs,
                    seed=seed_seq_joint[i])
            if subsample < n:
                I = util.subsample_ind(n, n_test_locs, seed=seed_seq_joint[i]+1)
                XI = X[I, :]
                YI = Y[I, :]
            else:
                XI = X
                YI = Y

            objs[i] = MeanEmbeddingTest.compute_nc_parameter(XI, YI, V,
                    gwidth2, reg='auto')

        objs[np.logical_not(np.isfinite(objs))] = -np.infty
        # best index 
        bind = np.argmax(objs)
        Vbest = MeanEmbeddingTest.init_locs_subset(tst_data, n_test_locs,
                seed=seed_seq_joint[bind])
        return Vbest


    @staticmethod
    def init_locs_subset(tst_data, n_test_locs, seed=2):
        """
        Randomly choose n_test_locs from the union of X and Y in tst_data.
        """
        XY = tst_data.stack_xy()
        n2 = XY.shape[0]
        I = util.subsample_ind(n2, n_test_locs, seed=seed)
        V = XY[I, :]
        return V


    @staticmethod 
    def init_locs_randn(tst_data, n_test_locs, seed=1):
        """Fit a Gaussian to the merged data of the two samples and draw 
        n_test_locs points from the Gaussian"""
        # set the seed
        rand_state = np.random.get_state()
        np.random.seed(seed)

        X, Y = tst_data.xy()
        d = X.shape[1]
        # fit a Gaussian in the middle of X, Y and draw sample to initialize T
        xy = np.vstack((X, Y))
        mean_xy = np.mean(xy, 0)
        cov_xy = np.cov(xy.T)
        [Dxy, Vxy] = np.linalg.eig(cov_xy + 1e-3*np.eye(d))
        Dxy = np.real(Dxy)
        Vxy = np.real(Vxy)
        Dxy[Dxy<=0] = 1e-3
        eig_pow = 0.9 # 1.0 = not shrink
        reduced_cov_xy = Vxy.dot(np.diag(Dxy**eig_pow)).dot(Vxy.T) + 1e-3*np.eye(d)

        T0 = np.random.multivariate_normal(mean_xy, reduced_cov_xy, n_test_locs)
        # reset the seed back to the original
        np.random.set_state(rand_state)
        return T0

    @staticmethod 
    def init_locs_2randn(tst_data, n_test_locs, subsample=10000, seed=1):
        """Fit a Gaussian to each dataset and draw half of n_test_locs from 
        each. This way of initialization can be expensive if the input
        dimension is large.
        
        """
        if n_test_locs == 1:
            return MeanEmbeddingTest.init_locs_randn(tst_data, n_test_locs, seed)

        X, Y = tst_data.xy()
        n = X.shape[0]
        with util.NumpySeedContext(seed=seed):
            # Subsample X, Y if needed. Useful if the data are too large.
            if n > subsample:
                I = util.subsample_ind(n, subsample, seed=seed+2)
                X = X[I, :]
                Y = Y[I, :]
            

            d = X.shape[1]
            # fit a Gaussian to each of X, Y
            mean_x = np.mean(X, 0)
            mean_y = np.mean(Y, 0)
            cov_x = np.cov(X.T)
            [Dx, Vx] = np.linalg.eig(cov_x + 1e-3*np.eye(d))
            Dx = np.real(Dx)
            Vx = np.real(Vx)
            # a hack in case the data are high-dimensional and the covariance matrix 
            # is low rank.
            Dx[Dx<=0] = 1e-3

            # shrink the covariance so that the drawn samples will not be so 
            # far away from the data
            eig_pow = 0.9 # 1.0 = not shrink
            reduced_cov_x = Vx.dot(np.diag(Dx**eig_pow)).dot(Vx.T) + 1e-3*np.eye(d)
            cov_y = np.cov(Y.T)
            [Dy, Vy] = np.linalg.eig(cov_y + 1e-3*np.eye(d))
            Vy = np.real(Vy)
            Dy = np.real(Dy)
            Dy[Dy<=0] = 1e-3
            reduced_cov_y = Vy.dot(np.diag(Dy**eig_pow).dot(Vy.T)) + 1e-3*np.eye(d)
            # integer division
            Jx = n_test_locs/2
            Jy = n_test_locs - Jx

            #from IPython.core.debugger import Tracer
            #t = Tracer()
            #t()
            assert Jx+Jy==n_test_locs, 'total test locations is not n_test_locs'
            Tx = np.random.multivariate_normal(mean_x, reduced_cov_x, Jx)
            Ty = np.random.multivariate_normal(mean_y, reduced_cov_y, Jy)
            T0 = np.vstack((Tx, Ty))

        return T0

    @staticmethod
    def grid_search_gwidth(tst_data, T, list_gwidth, alpha):
        """
        Linear search for the best Gaussian width in the list that maximizes 
        the test power, fixing the test locations ot T. 
        return: (best width index, list of test powers)
        """
        func_nc_param = MeanEmbeddingTest.compute_nc_parameter
        J = T.shape[0]
        return generic_grid_search_gwidth(tst_data, T, J, list_gwidth, alpha,
                func_nc_param)
            

    @staticmethod
    def optimize_gwidth(tst_data, T, gwidth0, max_iter=400, 
            gwidth_step_size=0.1, batch_proportion=1.0, tol_fun=1e-3):
        """Optimize the Gaussian kernel width by 
        maximizing the test power, fixing the test locations to T. X, Y should
        not be the same data as used in the actual test (i.e., should be a
        held-out set). 

        - max_iter: #gradient descent iterations
        - batch_proportion: (0,1] value to be multipled with nx giving the batch 
            size in stochastic gradient. 1 = full gradient ascent.
        - tol_fun: termination tolerance of the objective value
        
        Return (gaussian_width, info)
        """

        func_z = MeanEmbeddingTest.construct_z_theano
        # info = optimization info 
        gamma, info = optimize_gaussian_width(tst_data, T, gwidth0, func_z, 
                max_iter=max_iter, gwidth_step_size=gwidth_step_size,
                batch_proportion=batch_proportion, tol_fun=tol_fun)

        ninfo = {'test_locs': T, 'gwidths': info['gwidths'], 'obj_values':
                info['obj_values']}
        return ( gamma, ninfo  )


# ///////////// global functions ///////////////
def nc_parameter_pair(n_x, n_y, W_x, W_y, Sig, return_sig=False, reg='auto'):
    reg_tolerance = 0.00001
    n_features = len(W_x)
    W_x = np.squeeze(W_x)
    W_y = np.squeeze(W_y)
    constant = float(n_x * n_y) / (n_x + n_y)
    print('n_x: {}'.format(n_x))
    print('n_y: {}'.format(n_y))
    if n_features == 1:
        reg = 0 if reg=='auto' else reg
        s = constant * ( (W_x-W_y)**2 )/(reg+Sig)
    else:
        W = (W_x - W_y)
        if reg=='auto':
            # First compute with reg=0. If no problem, do nothing. 
            # If the covariance is singular, make 0 eigenvalues positive.
            try:
                evals, eV = np.linalg.eig(Sig)
                print(evals)
                s = constant*np.linalg.solve(Sig, W).dot(W)
            except np.linalg.LinAlgError:
                try:
                    # singular matrix 
                    # eigen decompose
                    evals, eV = np.linalg.eig(Sig)
                    evals = np.real(evals) # due to numerical error potentially 
                    eV = np.real(eV)
                    print('before',evals)
                    evals = np.maximum(0, evals)
                    # find the non-zero smallest eigenvalue 
                    ev_small = np.sort(evals[evals > 0])[0]
                    evals[evals <= 0] = min(reg_tolerance, ev_small)
                    print('after',evals)
                    # reconstruct Sig 
                    Sig = eV.dot(np.diag(evals)).dot(eV.T)
                    # try again
                    s = constant*np.linalg.solve(Sig, W).dot(W)
                except:
                    s = -1
        else:
            # assume reg is a number 
            # test statistic
            try:
                s = constant*np.linalg.solve(Sig + reg*np.eye(Sig.shape[0]), W).dot(W)
                Sig = Sig + reg*np.eye(Sig.shape[0])
                #evals, eV = np.linalg.eig(Sig + reg*np.eye(Sig.shape[0]))
                #evals = np.real(evals)
                #ev_small = np.sort(evals)[0]
            except np.linalg.LinAlgError:
                print('LinAlgError. Return -1 as the nc_parameter.')
                s = -1
        if return_sig:
            return s, Sig
        else:
            return s

def nc_parameter(n, W, Sig, return_sig=False, reg='auto'):
    reg_tolerance = 0.00001
    n_features = len(W)
    W = np.squeeze(W)
    if n_features == 1: 
        reg = 0 if reg=='auto' else reg
        s = float(n)*(W**2)/(reg+Sig)
        ev_small = reg + Sig
        Sig = reg + Sig
    else:
        if reg=='auto':
            # First compute with reg=0. If no problem, do nothing. 
            # If the covariance is singular, make 0 eigenvalues positive.
            try:
                s = n*np.linalg.solve(Sig, W).dot(W)
            except np.linalg.LinAlgError:
                try:
                    # singular matrix 
                    # eigen decompose
                    evals, eV = np.linalg.eig(Sig)
                    evals = np.real(evals) # due to numerical error potentially 
                    eV = np.real(eV)
                    evals = np.maximum(0, evals)
                    # find the non-zero smallest eigenvalue 
                    ev_small = np.sort(evals[evals > 0])[0]
                    evals[evals <= 0] = min(reg_tolerance, ev_small) #set to be the one that is smaller.
                    # reconstruct Sig 
                    Sig = eV.dot(np.diag(evals)).dot(eV.T)
                    # try again
                    s = n*np.linalg.solve(Sig, W).dot(W)
                except:
                    s = -1
        else:
            # assume reg is a number 
            # test statistic
            try:
                s = n*np.linalg.solve(Sig + reg*np.eye(Sig.shape[0]), W).dot(W)
                Sig = Sig + reg*np.eye(Sig.shape[0])
                print('fdfdfdd')
                #evals, eV = np.linalg.eig(Sig + reg*np.eye(Sig.shape[0]))
                #evals = np.real(evals)
                #ev_small = np.sort(evals)[0]
            except np.linalg.LinAlgError:
                print('LinAlgError. Return -1 as the nc_parameter.')
                s = -1
    if return_sig:
        return s, Sig
    else:
        return s

def generic_nc_parameter(Z, reg='auto'):
    """
    Compute the non-centrality parameter of the non-central Chi-squared 
    which is approximately the distribution of the test statistic under the H_1
    (and H_0). The empirical nc parameter is also the test statistic. 

    - reg can be 'auto'. This will automatically determine the lowest value of 
    the regularization parameter so that the statistic can be computed.
    """
    #from IPython.core.debugger import Tracer 
    #Tracer()()

    n = Z.shape[0]
    Sig = np.cov(Z.T)
    W = np.mean(Z, 0)
    n_features = len(W)
    if n_features == 1:
        reg = 0 if reg=='auto' else reg
        s = float(n)*(W[0]**2)/(reg+Sig)
        ev_small = reg + Sig
    else:
        if reg=='auto':
            # First compute with reg=0. If no problem, do nothing. 
            # If the covariance is singular, make 0 eigenvalues positive.
            try:
                s = n*np.linalg.solve(Sig, W).dot(W)
            except np.linalg.LinAlgError:
                try:
                    # singular matrix 
                    # eigen decompose
                    evals, eV = np.linalg.eig(Sig)
                    evals = np.real(evals)
                    eV = np.real(eV)
                    evals = np.maximum(0, evals)
                    # find the non-zero second smallest eigenvalue
                    snd_small = np.sort(evals[evals > 0])[0]
                    evals[evals <= 0] = snd_small

                    # reconstruct Sig 
                    Sig = eV.dot(np.diag(evals)).dot(eV.T)
                    # try again
                    s = n*np.linalg.solve(Sig, W).dot(W)
                except:
                    s = -1
        else:
            # assume reg is a number 
            # test statistic
            try:
                s = n*np.linalg.solve(Sig + reg*np.eye(Sig.shape[0]), W).dot(W)
            except np.linalg.LinAlgError:
                print('LinAlgError. Return -1 as the nc_parameter.')
                s = -1 
    return s

def generic_grid_search_gwidth(tst_data, T, df, list_gwidth, alpha, func_nc_param):
    """
    Linear search for the best Gaussian width in the list that maximizes 
    the test power, fixing the test locations to T. 
    The test power is given by the CDF of a non-central Chi-squared 
    distribution.
    return: (best width index, list of test powers)
    """
    # number of test locations
    X, Y = tst_data.xy()
    powers = np.zeros(len(list_gwidth))
    lambs = np.zeros(len(list_gwidth))
    thresh = stats.chi2.isf(alpha, df=df)
    #print('thresh: %.3g'% thresh)
    for wi, gwidth in enumerate(list_gwidth):
        # non-centrality parameter
        try:

            #from IPython.core.debugger import Tracer 
            #Tracer()()
            lamb = func_nc_param(X, Y, T, gwidth, reg=0)
            if lamb <= 0:
                # This can happen when Z, Sig are ill-conditioned. 
                #print('negative lamb: %.3g'%lamb)
                raise np.linalg.LinAlgError
            if np.iscomplex(lamb):
                # complext value can happen if the covariance is ill-conditioned?
                print('Lambda is complex. Truncate the imag part. lamb: %s'%(str(lamb)))
                lamb = np.real(lamb)

            #print('thresh: %.3g, df: %.3g, nc: %.3g'%(thresh, df, lamb))
            power = stats.ncx2.sf(thresh, df=df, nc=lamb)
            powers[wi] = power
            lambs[wi] = lamb
            print('i: %2d, lamb: %5.3g, gwidth: %5.3g, power: %.4f'
                   %(wi, lamb, gwidth, power))
        except np.linalg.LinAlgError:
            # probably matrix inverse failed. 
            print('LinAlgError. skip width (%d, %.3g)'%(wi, gwidth))
            powers[wi] = np.NINF
            lambs[wi] = np.NINF
    # to prevent the gain of test power from numerical instability, 
    # consider upto 3 decimal places. Widths that come early in the list 
    # are preferred if test powers are equal.
    besti = np.argmax(np.around(powers, 3))
    return besti, powers


# Used by SmoothCFTest and MeanEmbeddingTest
def optimize_gaussian_width(tst_data, T, gwidth0, func_z, max_iter=400, 
        gwidth_step_size=0.1, batch_proportion=1.0, 
        tol_fun=1e-3 ):
    """Optimize the Gaussian kernel width by gradient ascent 
    by maximizing the test power.
    This does the same thing as optimize_T_gaussian_width() without optimizing 
    T (T = test locations / test frequencies).

    Return (optimized Gaussian width, info)
    """

    X, Y = tst_data.xy()
    nx, d = X.shape
    # initialize Theano variables
    Tth = theano.shared(T, name='T')
    Xth = tensor.dmatrix('X')
    Yth = tensor.dmatrix('Y')
    it = theano.shared(1, name='iter')
    # square root of the Gaussian width. Use square root to handle the 
    # positivity constraint by squaring it later.
    gamma_sq_init = gwidth0**0.5
    gamma_sq_th = theano.shared(gamma_sq_init, name='gamma')

    #sqr(x) = x^2
    Z = func_z(Xth, Yth, Tth, tensor.sqr(gamma_sq_th))
    W = Z.sum(axis=0)/nx
    # covariance 
    Z0 = Z - W
    Sig = Z0.T.dot(Z0)/nx

    # gradient computation does not support solve()
    #s = slinalg.solve(Sig, W).dot(nx*W)
    s = nlinalg.matrix_inverse(Sig).dot(W).dot(W)*nx
    gra_gamma_sq = tensor.grad(s, gamma_sq_th)
    step_pow = 0.5
    max_gam_sq_step = 1.0
    func = theano.function(inputs=[Xth, Yth], outputs=s, 
           updates=[
              (it, it+1), 
              #(gamma_sq_th, gamma_sq_th+gwidth_step_size*gra_gamma_sq\
              #        /it**step_pow/tensor.sum(gra_gamma_sq**2)**0.5 ) 
              (gamma_sq_th, gamma_sq_th+gwidth_step_size*tensor.sgn(gra_gamma_sq) \
                      *tensor.minimum(tensor.abs_(gra_gamma_sq), max_gam_sq_step) \
                      /it**step_pow) 
              ] 
           )
    # //////// run gradient ascent //////////////
    S = np.zeros(max_iter)
    gams = np.zeros(max_iter)
    for t in range(max_iter):
        # stochastic gradient ascent
        ind = np.random.choice(nx, min(int(batch_proportion*nx), nx), replace=False)
        # record objective values 
        S[t] = func(X[ind, :], Y[ind, :])
        gams[t] = gamma_sq_th.get_value()**2

        # check the change of the objective values 
        if t >= 2 and abs(S[t]-S[t-1]) <= tol_fun:
            break

    S = S[:t]
    gams = gams[:t]

    # optimization info 
    info = {'T': T, 'gwidths': gams, 'obj_values': S}
    return (gams[-1], info  )




# Used by SmoothCFTest and MeanEmbeddingTest
def optimize_T_gaussian_width(tst_data, T0, gwidth0, func_z, max_iter=400, 
        T_step_size=0.05, gwidth_step_size=0.01, batch_proportion=1.0, 
        tol_fun=1e-3, reg=1e-5):
    """Optimize the T (test locations for MeanEmbeddingTest, frequencies for 
    SmoothCFTest) and the Gaussian kernel width by 
    maximizing the test power. X, Y should not be the same data as used 
    in the actual test (i.e., should be a held-out set). 
    Optimize the empirical version of Lambda(T) i.e., the criterion used 
    to optimize the test locations.

    - T0: Jxd numpy array. initial value of T,  where
      J = the number of test locations/frequencies
    - gwidth0: initial Gaussian width (width squared for the MeanEmbeddingTest)
    - func_z: function that works on Theano variables 
        to construct features to be used for the T^2 test. 
        (X, Y, T, gaussian_width) |-> n x J'
    - max_iter: #gradient descent iterations
    - batch_proportion: (0,1] value to be multipled with nx giving the batch 
        size in stochastic gradient. 1 = full gradient ascent.
    - tol_fun: termination tolerance of the objective value
    - reg: a regularization parameter. Must be a non-negative number.
    
    Return (test_locs, gaussian_width, info)
    """

    #print 'T0: '
    #print(T0)
    X, Y = tst_data.xy()
    nx, d = X.shape
    J = T0.shape[0]
    # initialize Theano variables
    T = theano.shared(T0, name='T')
    Xth = tensor.dmatrix('X')
    Yth = tensor.dmatrix('Y')
    it = theano.shared(1, name='iter')
    # square root of the Gaussian width. Use square root to handle the 
    # positivity constraint by squaring it later.
    gamma_sq_init = gwidth0**0.5
    gamma_sq_th = theano.shared(gamma_sq_init, name='gamma')
    regth = theano.shared(reg, name='reg')
    diag_regth = regth*tensor.eye(J)

    #sqr(x) = x^2
    Z = func_z(Xth, Yth, T, tensor.sqr(gamma_sq_th))
    W = Z.sum(axis=0)/nx
    # covariance 
    Z0 = Z - W
    Sig = Z0.T.dot(Z0)/nx

    # gradient computation does not support solve()
    #s = slinalg.solve(Sig, W).dot(nx*W)
    s = nlinalg.matrix_inverse(Sig + diag_regth).dot(W).dot(W)*nx
    gra_T, gra_gamma_sq = tensor.grad(s, [T, gamma_sq_th])
    step_pow = 0.5
    max_gam_sq_step = 1.0
    func = theano.function(inputs=[Xth, Yth], outputs=s, 
           updates=[
              (T, T+T_step_size*gra_T/it**step_pow/tensor.sum(gra_T**2)**0.5 ), 
              (it, it+1), 
              #(gamma_sq_th, gamma_sq_th+gwidth_step_size*gra_gamma_sq\
              #        /it**step_pow/tensor.sum(gra_gamma_sq**2)**0.5 ) 
              (gamma_sq_th, gamma_sq_th+gwidth_step_size*tensor.sgn(gra_gamma_sq) \
                      *tensor.minimum(tensor.abs_(gra_gamma_sq), max_gam_sq_step) \
                      /it**step_pow) 
              ] 
           )
           #updates=[(T, T+T_step_size*gra_T), (it, it+1), 
           #    (gamma_sq_th, gamma_sq_th+gwidth_step_size*gra_gamma_sq) ] )
                           #updates=[(T, T+0.1*gra_T), (it, it+1) ] )

    # //////// run gradient ascent //////////////
    S = np.zeros(max_iter)
    J = T0.shape[0]
    Ts = np.zeros((max_iter, J, d))
    gams = np.zeros(max_iter)
    for t in range(max_iter):
        # stochastic gradient ascent
        ind = np.random.choice(nx, min(int(batch_proportion*nx), nx), replace=False)
        # record objective values 
        try:
            S[t] = func(X[ind, :], Y[ind, :])
        except: 
            print('Exception occurred during gradient descent. Stop optimization.')
            print('Return the value from previous iter. ')
            import traceback as tb 
            tb.print_exc()
            t = t -1
            break

        Ts[t] = T.get_value()
        gams[t] = gamma_sq_th.get_value()**2

        # check the change of the objective values 
        if t >= 2 and abs(S[t]-S[t-1]) <= tol_fun:
            break

    S = S[:t+1]
    Ts = Ts[:t+1]
    gams = gams[:t+1]

    # optimization info 
    info = {'Ts': Ts, 'T0':T0, 'gwidths': gams, 'obj_values': S, 'gwidth0':
            gwidth0}

    if t >= 0:
        opt_T = Ts[-1]
        # for some reason, optimization can give a non-numerical result
        opt_gwidth = gams[-1] if util.is_real_num(gams[-1]) else gwidth0

        if np.linalg.norm(opt_T) <= 1e-5:
            opt_T = T0
            opt_gwidth = gwidth0
    else:
        # Probably an error occurred in the first iter.
        opt_T = T0
        opt_gwidth = gwidth0
    return (opt_T, opt_gwidth, info  )


