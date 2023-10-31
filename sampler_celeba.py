import jax.numpy as jnp
import jax
from jax import random
from PIL import Image
import pandas as pd


def load_celeba_2ST(number_images):
    """
    Load CelebA images of women and of men.
    
    Parameters
    ----------
    number_images: int
        Number of CelebA images of both women and men to load.
        
    Returns
    -------
    output: array_like
        Array of shape (2, number_images, 3 * 218 * 178)
        with images of women (axis 0, index 0) and of men (axis 0, index 1).
    """
    keyword = 'Male'
    df = pd.read_csv('list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=[keyword])
    celeba = jnp.zeros((2, num_images, 218 * 178 * 3))
    celeba = celeba.at[0].set(
        jnp.array(
            [
                jnp.array(
                    Image.open('img_align_celeba/' + fname)
                ).reshape(-1) / 255. for fname in list(df.loc[df[keyword] == -1].index.values[:number_images]) 
            ]
        )
    )
    celeba = celeba.at[1].set(
        jnp.array(
            [
                jnp.array(
                    Image.open('img_align_celeba/' + fname)
                ).reshape(-1) / 255. for fname in list(df.loc[df[keyword] == 1].index.values[:number_images]) 
            ]
        )
    )
    return celeba


def sampler_celeba_2ST(key, m, n, corruption, celeba):
    """
    CelebA sampler for two-sample testing.
    
    Parameters
    ----------
    key:
        Jax random key (can be generated by jax.random.PRNGKey(seed) for an integer seed).
    m: int
        Number of samples of CelebA women images for X.
    n: int
        Number of samples of CelebA women/men images for Y.
    corruption: scalar
        Corruption parameter between 0 and 1 (explained above).
    celeba: array_like
        Array of shape (2, number_images, 3 * 218 * 178) as outputed by 
        load_celeba_2ST(number_images) for number_images greater than m and n.
        
    Returns
    -------
    output: tuple
        tuple (X, Y) where
            X: array_like
                Array of shape (m, 3 * 218 * 178) consisting of uniformly sampled CelebA images of women.
            Y: array_like
                Array of shape (n, 3 * 218 * 178) consisting of:
                - with probability 'corruption' uniformly sampled CelebA images of men,
                - with probability '1 - corruption' uniformly sampled CelebA images of women.
    """
    # sample sizes
    num_images = celeba.shape[1]
    assert n <= num_images
    assert m <= num_images
    
    # keys
    key, subkey = random.split(key)
    subkeys = random.split(subkey, num=5)
    
    # X
    key, subkey = random.split(key)
    indices_women = jax.random.permutation(subkeys[0], jnp.array([False,] * num_images).at[:m].set(True))
    X = celeba[0, indices_women]
        
    # Y
    choice = jax.random.choice(
        subkeys[1], 
        jnp.arange(2), 
        shape=(n,), 
        p=jnp.array(
            [
                1 - corruption, 
                corruption
            ]
        )
    )
    n_women = jnp.sum(choice == 0)  # n = n_women + n_men
    n_men = jnp.sum(choice == 1)
    indices_women = jax.random.permutation(subkeys[2], jnp.array([False,] * num_images).at[:n_women].set(True))
    indices_men = jax.random.permutation(subkeys[3], jnp.array([False,] * num_images).at[:n_men].set(True))
    Y = jnp.concatenate((celeba[0, indices_women], celeba[1, indices_men]))
    Y = jax.random.permutation(subkeys[4], Y, axis=0)
    
    return X, Y
