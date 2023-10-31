import jax
import jax.numpy as jnp
from jax import vmap, random, jit, lax
from jax.flatten_util import ravel_pytree
from functools import partial
from kernel import kernel_matrix, distances
    

@partial(jit, static_argnums=(7, 8, 9, 10, 11))
def dpmmd(
    key,
    X,
    Y,
    epsilon,
    delta=0,
    alpha=0.05,
    bandwidth_multiplier=1,
    kernel="gaussian",
    number_permutations=2000, 
    return_dictionary=False,
    min_mem_kernel=False,
    min_mem_permutations=False,
):
    """
    Differential Privacy Two-Sample dpMMD test.
     
    Given data from one distribution and data from another distribution,
    return 0 if the test fails to reject the null 
        (i.e. data comes from the same distribution), 
    or return 1 if the test rejects the null 
        (i.e. data comes from different distributions).
    
    Fixing the two sample sizes and the dimension, the first time the function is
    run it is getting compiled. After that, the function can fastly be evaluated on 
    any data with the same sample sizes and dimension.
    
    Parameters
    ----------
    key:
        Jax random key (can be generated by jax.random.PRNGKey(seed) for an integer seed).
    X: array_like
        The shape of X must be of the form (m, d) where m is the number
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
    bandwidth_multiplier: scalar
        Bandwidth for l1 kernels is d * bandwidth_multiplier.
        Bandwidth for l2 kernels is sqrt(d) * bandwidth_multiplier.
        So that the l1 and l2 norms are of constant order wrt dimension.
    kernel: str
        The value of kernel must be "gaussian" or "laplace" or "imq".
    number_permutations: int
        Number of permutations to approximate the quantiles.
    return_dictionary: bool
        If True, a dictionary is also returned containing:
        - Test output
        - DP level epsilon
        - DP level delta
        - Test level alpha
        - Number of permutations
        - Kernel
        - Bandwidth
        - Non-privatised MMD V-statistic
        - Privacy Laplace noise for MMD V-statistic
        - Privatised MMD V-statistic
        - Privatised MMD quantile
        - Privatised p-value
        - Privatised p-value threshold        
    min_mem_kernel: bool
        If True then compute kernel matrix sequentially (lower memory).
        The speed improvement can vary depending on the use of CPU/GPU.
    min_mem_permutations: bool
        If True then compute the permuted statistics sequentially (lower memory).
        The speed improvement can vary depending on the use of CPU/GPU.
        
    Returns
    -------
    output: int
        0 if the dpMMD test fails to reject the null 
            (i.e. data comes from the same distribution)
        1 if the dpMMD test rejects the null 
            (i.e. data comes from different distributions)
    dictionary: dict
        Returned only if return_dictionary is True (see return_dictionary above).
    """    
    # Assertions
    B = number_permutations
    m = X.shape[0]
    n = Y.shape[0]
    assert n >= 2 and m >= 2
    assert X.shape[1] == Y.shape[1]
    d = X.shape[1]
    assert kernel in ("gaussian", "laplace", "imq")
    assert B > 0 and type(B) == int
    l = "l2" if kernel in ("gaussian", "imq") else "l1"
    d_scale = d if l == 'l1' else jnp.sqrt(d)
    bandwidth = bandwidth_multiplier * d_scale

    # DP level
    # threshold delta between zero and one
    delta = jnp.minimum(delta, 1)
    delta = jnp.maximum(delta, 0)
    # epsilon cannot be negative or zero
    epsilon = jnp.abs(epsilon)
    privacy_level = epsilon + jnp.log(1 / (1 - delta))
    
    # DP noise
    key, subkey = random.split(key)
    sensitivity = jnp.sqrt(2) / jnp.minimum(n, m)
    laplace_noise = random.laplace(subkey, shape=(B + 1,))
    dp_noise = laplace_noise * 2 * sensitivity / privacy_level
    
    # Compute kernel matrix
    Z = jnp.concatenate((X, Y))
    pairwise_matrix = distances(Z, Z, l, matrix=True, min_mem=min_mem_kernel)
    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
    if min_mem_permutations:
        # Setup for permutations
        key, subkey = random.split(key)
        # (B+1, m): rows of permuted indices
        idx = random.permutation(subkey, jnp.array([[i for i in range(m + n)]] * (B + 1)), axis=1, independent=True)  
        idx = idx.at[B].set(jnp.array([i for i in range(m + n)]))
        # compute MMD permuted values (B + 1, )
        v = jnp.concatenate((jnp.ones(m) / m, - jnp.ones(n) / n))
        compute_mmd = lambda index : jnp.sqrt(v[index] @ K @ v[index])
        mmd_values = lax.map(compute_mmd, idx)  # (B + 1, )
    else:
        # Setup for permutations
        key, subkey = random.split(key)
        # (B+1, m+n): rows of permuted indices
        idx = random.permutation(subkey, jnp.array([[i for i in range(m + n)]] * (B + 1)), axis=1, independent=True)   
        v = jnp.concatenate((jnp.ones(m) / m, - jnp.ones(n) / n))  # (m+n, )
        V_stack = jnp.tile(v, (B + 1, 1))  # (B+1, m+n)
        V = jnp.take_along_axis(V_stack, idx, axis=1)  # (B+1, m+n): permute the entries of the rows
        V = V.at[B].set(v) # (B+1)th entry is the original MMD (no permutation)
        V = V.transpose()  # (m+n, B+1)
        # compute MMD permuted values (B + 1, )
        mmd_values = jnp.sqrt(jnp.sum(V * (K @ V), 0)) 

    # add DP noise
    mmd_no_noise_original = mmd_values[B]
    mmd_values = mmd_values + dp_noise # differential privacy
    mmd_original = mmd_values[B]
    mmd_values_sorted = jnp.sort(mmd_values) # (B + 1, )
        
    # test output test (p-value)
    p_val = jnp.mean(mmd_values >= mmd_original)
    threshold = alpha
    # reject if p_val <= threshold
    reject_p_val = p_val <= threshold

    # test output test (quantile)
    quantile = mmd_values_sorted[(jnp.ceil((B + 1) * (1 - alpha))).astype(int) - 1]
    # reject if mmd_original > quantile
    reject_mmd_val = mmd_original > quantile

    # assert reject_p_val == reject_mmd_val
    output = reject_p_val

    # create rejection dictionary 
    reject_dictionary = {}
    reject_dictionary["dpMMD test reject"] = reject_p_val
    reject_dictionary["DP epsilon"] = epsilon
    reject_dictionary["DP delta"] = delta
    reject_dictionary["Test level"] = alpha
    reject_dictionary["Number of permutations"] = number_permutations
    reject_dictionary["Kernel " + kernel] = True
    reject_dictionary["Bandwidth"] = bandwidth
    reject_dictionary["Non-privatised MMD V-statistic"] = mmd_no_noise_original
    reject_dictionary["Privacy Laplace noise for MMD V-statistic"] = dp_noise[B]
    reject_dictionary["Privatised MMD V-statistic"] = mmd_original
    reject_dictionary["Privatised MMD quantile"] = quantile
    reject_dictionary["Privatised p-value"] = p_val
    reject_dictionary["Privatised p-value threshold"] = threshold

    # dpMMD test output
    if return_dictionary:
        return reject_p_val.astype(int), reject_dictionary
    else:
        return reject_p_val.astype(int)
    

@partial(jit, static_argnums=(8, 9, 10, 11, 12))
def dphsic(
    key,
    X,
    Y,
    epsilon,
    delta=0,
    alpha=0.05,
    bandwidth_multiplier_X=1,
    bandwidth_multiplier_Y=1,
    kernel_X="gaussian",
    kernel_Y="gaussian",
    number_permutations=2000,
    return_dictionary=False,
    min_mem_kernel=False,
):
    """
    Differential Privacy Independence dpHSIC test.
     
    Given paired data from a joint distribution,
    return 0 if the test fails to reject the null 
        (i.e. paired data is independent), 
    or return 1 if the test rejects the null 
        (i.e. paired data is dependent).
    
    Fixing the sample size and the dimension, the first time the function is
    run it is getting compiled. After that, the function can fastly be evaluated on 
    any data with the same sample size and dimension.
    
    Parameters
    ----------
    key:
        Jax random key (can be generated by jax.random.PRNGKey(seed) for an integer seed).
    X : array_like
        The shape of X must be of the form (n, d_X) where m is the number
        of samples and d_X is the dimension.
    Y: array_like
        The shape of Y must be of the form (n, d_Y) where m is the number
        of samples and d_Y is the dimension.
    epsilon: scalar
        Differential privacy level (positive scalar).
    delta: scalar
        Approximate differential privacy level (0 <= delta <= 1).
    alpha: scalar
        Test significance level between 0 and 1.
    bandwidth_multiplier_X: scalar
        Bandwidth for l1 X-kernels is d_X * bandwidth_multiplier.
        Bandwidth for l2 X-kernels is sqrt(d_X) * bandwidth_multiplier.
        So that the l1 and l2 norms are of constant order wrt dimension.
    bandwidth_multiplier_Y: scalar
        Bandwidth for l1 Y-kernels is d_Y * bandwidth_multiplier.
        Bandwidth for l2 Y-kernels is sqrt(d_Y) * bandwidth_multiplier.
        So that the l1 and l2 norms are of constant order wrt dimension.
    kernel_X: str
        The value of kernel_X for X must be "gaussian", "laplace", "imq"
    kernel_Y: str
        The value of kernel_Y for Y must be "gaussian", "laplace", "imq"
    number_permutations: int
        Number of permutations to approximate the quantiles.
    return_dictionary: bool
        If True, a dictionary is also returned containing:
        - Test output
        - DP level epsilon
        - DP level delta
        - Test level alpha
        - Number of permutations
        - Kernel
        - Bandwidth
        - Non-privatised HSIC V-statistic
        - Privacy Laplace noise for HSIC V-statistic
        - Privatised HSIC V-statistic
        - Privatised HSIC quantile
        - Privatised p-value
        - Privatised p-value threshold   
    min_mem_kernel: bool
        If True then compute kernel matrix sequentially (lower memory).
        The speed improvement can vary depending on the use of CPU/GPU.  
        
    Returns
    -------
    output: int
        0 if the dpHSIC test fails to reject the null 
            (i.e. paired data is independent)
        1 if the dpHSIC test rejects the null 
            (i.e. paired data is dependent)
    dictionary: dict
        Returned only if return_dictionary is True (see return_dictionary above).
    """    
    # Assertions
    B = number_permutations
    assert X.shape[0] == Y.shape[0]
    n = X.shape[0]
    d_X = X.shape[1]
    d_Y = Y.shape[1]
    assert n >= 2
    assert kernel_X in ("gaussian", "laplace", "imq")
    assert kernel_Y in ("gaussian", "laplace", "imq")
    assert B > 0 and type(B) == int
    l_X = "l2" if kernel_X in ("gaussian", "imq") else "l1"
    d_scale_X = d_X if l_X == "l1" else jnp.sqrt(d_X)
    bandwidth_X = bandwidth_multiplier_X * d_scale_X
    l_Y = "l2" if kernel_Y in ("gaussian", "imq") else "l1"
    d_scale_Y = d_Y if l_Y == "l1" else jnp.sqrt(d_Y)
    bandwidth_Y = bandwidth_multiplier_Y * d_scale_Y
    
    # DP level
    # threshold delta between zero and one
    delta = jnp.minimum(delta, 1)
    delta = jnp.maximum(delta, 0)
    # epsilon cannot be negative or zero
    epsilon = jnp.abs(epsilon)
    privacy_level = epsilon + jnp.log(1 / (1 - delta))

    # DP noise
    key, subkey = random.split(key)
    sensitivity = 4 * (n - 1) / n ** 2
    laplace_noise = random.laplace(subkey, shape=(B + 1,))
    dp_noise = laplace_noise * 2 * sensitivity / privacy_level

    # Setup for permutations
    key, subkey = random.split(key)
    # (B+1, n): rows of permuted indices
    idx = random.permutation(subkey, jnp.array([[i for i in range(n)]] * (B + 1)), axis=1, independent=True)
    idx = idx.at[B].set(jnp.array([i for i in range(n)]))
    
    # compute both kernel matrices
    pairwise_matrix_X = distances(X, X, l_X, matrix=True, min_mem=min_mem_kernel)
    K = kernel_matrix(pairwise_matrix_X, l_X, kernel_X, bandwidth_X)
    pairwise_matrix_Y = distances(Y, Y, l_Y, matrix=True, min_mem=min_mem_kernel)
    L = kernel_matrix(pairwise_matrix_Y, l_Y, kernel_Y, bandwidth_Y)
    
    # center kernel matrix L (HLH for H = I - 1 @ 1.T / n)
    center_rows = lambda mat : mat - mat.mean(1).reshape(-1, 1)
    center_columns = lambda mat : mat - mat.mean(0)
    L = center_rows(center_columns(L))

    # compute HSIC permuted values (B + 1, )
    compute_hsic = lambda index : jnp.sqrt(jnp.sum(K[index][:, index] * L) / n ** 2)
    hsic_values = lax.map(compute_hsic, idx)  # (B + 1, )
    hsic_no_noise_original = hsic_values[B]
    hsic_values = hsic_values + dp_noise # differential privacy
    hsic_original = hsic_values[B]
    hsic_values_sorted = jnp.sort(hsic_values) # (B + 1, )

    # test output test (p-value)
    p_val = jnp.mean(hsic_values >= hsic_original)
    threshold = alpha
    # reject if p_val <= threshold
    reject_p_val = p_val <= threshold

    # test output test (quantile)
    quantile = hsic_values_sorted[(jnp.ceil((B + 1) * (1 - alpha))).astype(int) - 1]
    # reject if hsic_original > quantile
    reject_hsic_val = hsic_original > quantile
   
    # assert reject_p_val == reject_hsic_val
    output = reject_p_val

    # create rejection dictionary 
    reject_dictionary = {}
    reject_dictionary["dpHSIC test reject"] = reject_p_val
    reject_dictionary["DP epsilon"] = epsilon
    reject_dictionary["DP delta"] = delta
    reject_dictionary["Test level"] = alpha
    reject_dictionary["Number of permutations"] = number_permutations
    reject_dictionary["Kernel X " + kernel_X] = True
    reject_dictionary["Kernel Y " + kernel_Y] = True
    reject_dictionary["Bandwidth X"] = bandwidth_X
    reject_dictionary["Bandwidth Y"] = bandwidth_Y
    reject_dictionary["Non-privatised HSIC V-statistic"] = hsic_no_noise_original
    reject_dictionary["Privacy Laplace noise for HSIC V-statistic"] = dp_noise[B]    
    reject_dictionary["Privatised HSIC V-statistic"] = hsic_original
    reject_dictionary["Privatised HSIC quantile"] = quantile
    reject_dictionary["Privatised p-value"] = p_val
    reject_dictionary["Privatised p-value threshold"] = threshold

    # dpHSIC test output
    if return_dictionary:
        return reject_p_val.astype(int), reject_dictionary
    else:
        return reject_p_val.astype(int)


def human_readable_dict(dictionary):
    """
    Transform all jax arrays of one element into scalars.
    """
    meta_keys = dictionary.keys()
    for meta_key in meta_keys:
        if isinstance(dictionary[meta_key], jnp.ndarray):
            dictionary[meta_key] = dictionary[meta_key].item()
        elif isinstance(dictionary[meta_key], dict):
            for key in dictionary[meta_key].keys():
                if isinstance(dictionary[meta_key][key], jnp.ndarray):
                    dictionary[meta_key][key] = dictionary[meta_key][key].item()
