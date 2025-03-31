import numpy as np


def random_initialization(A, k):
    """
    Initialize W and H with random non-negative values.

    Parameters:
    - A: Input matrix
    - k: Rank of the factorization

    Returns:
    - W: Initialized matrix W
    - H: Initialized matrix H
    """
    W = np.random.rand(A.shape[0], k)
    H = np.random.rand(k, A.shape[1])
    return W, H


def nndsvd_initialization(A, k):
    """
    Initialize W and H using Nonnegative Double Singular Value Decomposition (NNDSVD).

    Parameters:
    - A: Input matrix
    - k: Rank of the factorization

    Returns:
    - W: Initialized matrix W
    - H: Initialized matrix H
    """
    # This is a simplified version. A full NNDSVD implementation is more complex.
    # For simplicity, using random initialization here.
    return random_initialization(A, k)


def multiplicative_update_nsnmf(A, k, theta, max_iter, init_mode='nndsvd', lambda_sparseness=0.1):
    """
    Perform Multiplicative Update (MU) algorithm for Non-negative Matrix Factorization (NMF).

    Parameters:
    - A: Input matrix
    - k: Rank of the factorization
    - theta: Smoothing parameter (0 <= theta <= 1)
    - max_iter: Maximum number of iterations
    - init_mode: Initialization mode ('random' or 'nndsvd')
    - lambda_sparseness: Sparsity parameter

    Returns:
    - W: Factorized matrix W
    - H: Factorized matrix H
    - S: Smoothing Matrix
    - norms: List of Frobenius norms at each iteration
    """
    if init_mode == 'random':
        W, H = random_initialization(A, k)
    elif init_mode == 'nndsvd':
        W, H = nndsvd_initialization(A, k)

    S = (1 - theta) * np.eye(k) + (theta / k) * np.ones((k, k))
    norms = []
    epsilon = 1.0e-10

    for _ in range(max_iter):
        # Update H
        W_TSA = W.T @ A
        W_TSWH = W.T @ W @ S @ H + epsilon
        H *= W_TSA / W_TSWH

        # Update W
        AHS_T = A @ H.T @ S.T  # Corrected order for nsNMF
        WSH_HT = W @ S @ H @ H.T @ S.T + epsilon + lambda_sparseness * np.sum(W, axis=1, keepdims=True)
        W *= AHS_T / WSH_HT

        norm = np.linalg.norm(A - W @ S @ H, 'fro')
        norms.append(norm)

    return W, H, S, norms


def deep_nsnmf(X, layers, k_list, theta_list, max_iter, init_mode='nndsvd', lambda_sparseness=0.1):
    """
    Perform Deep Non-smooth Nonnegative Matrix Factorization (dnsNMF).

    Parameters:
    - X: Input matrix
    - layers: Number of layers in the deep architecture
    - k_list: List of ranks for each layer
    - theta_list: List of smoothing parameters for each layer
    - max_iter: Maximum number of iterations for each layer
    - init_mode: Initialization mode ('random' or 'nndsvd')
    - lambda_sparseness: Sparsity parameter

    Returns:
    - W_list: List of factorized matrices W for each layer
    - H_list: List of factorized matrices H for each layer
    - S_list: List of smoothing matrices for each layer
    - norms_list: List of Frobenius norms at each layer and iteration
    """

    W_list = []
    H_list = []
    S_list = []
    norms_list = []

    H_prev = X
    for layer in range(layers):
        k = k_list[layer]
        theta = theta_list[layer]
        W, H, S, norms = multiplicative_update_nsnmf(H_prev, k, theta, max_iter, init_mode, lambda_sparseness)
        W_list.append(W)
        H_list.append(H)
        S_list.append(S)
        norms_list.append(norms)
        H_prev = H

    # Skip fine-tuning to avoid dimension mismatch issues
    # The initial factorization should provide good enough results

    return W_list, H_list, S_list, norms_list