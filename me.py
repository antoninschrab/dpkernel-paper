"""
A Differentially Private Kernel Two-Sample Test
Anant Raj, Ho Chung Leon Law, Dino Sejdinovic, Mijung Park
https://arxiv.org/abs/1808.00380
https://github.com/hcllaw/private_tst
"""


import os, sys
import numpy as np
from private_me.data import TSTData
import private_me.tst as tst
from private_me.kernel import KGauss


class HiddenPrints:
    """
    Hide prints and warnings.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr.close()
        sys.stderr = self._original_stderr

        
def me(X, Y, epsilon, delta=1e-5, alpha=0.05, seed=42):
    """    
    Differential Privacy MeanEmbeddingTest with test locations optimzied.
    Return results from calling perform_test().
    
    Parameters
    ----------
    X: array_like
        The shape of X must be of the form (n, d) where n is the number
        of samples and d is the dimension.
    Y: array_like
        The shape of Y must be of the form (n, d) where n is the number
        of samples and d is the dimension.
    epsilon: scalar
        Differential privacy level (positive scalar).
    delta: scalar
        Approximate differential privacy level (0 <= delta <= 1).
    alpha: scalar
        Test significance level between 0 and 1.
    seed: int
        Random seed.
        
    Returns
    -------
    output: int
        0 if the DP ME test fails to reject the null 
            (i.e. data comes from the same distribution)
        1 if the DP ME test rejects the null 
            (i.e. data comes from different distributions)
    """
    X = np.array(X)
    Y = np.array(Y)
    epsilon = np.array(epsilon)
    delta = np.array(delta)
    assert X.shape[0] == Y.shape[0]
    
    with HiddenPrints():
        data = TSTData(X, Y)
        tr, te = data.subsample(X.shape[0], seed=seed+4).split_tr_te(
            tr_proportion=0.2, 
            seed=seed+5,
        )
        test_locs, gwidth2, info = tst.MeanEmbeddingTest.optimize_locs_width(
            tr, 
            alpha, 
            n_test_locs=5, 
            max_iter=200,
            locs_step_size=0.1, 
            gwidth_step_size=0.1, 
            batch_proportion=1.0,
            tol_fun=1e-3, 
            seed=seed+6,
        )
        kernel = KGauss(gwidth2)
        test = tst.PrivateSt_Test('ME', test_locs, kernel, gwidth2, 'auto', alpha)
        output = test.perform_test(
            te, 
            epsilon=epsilon, 
            delta=delta,
            gauss_noise='Improved',
            null='private',
            seed=seed+7,
        )

    return int(output['h0_rejected'])        
