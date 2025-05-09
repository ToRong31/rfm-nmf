import numpy as np
import torch

def random_initialization(A, rank):
    """
    Initialize matrices W and H randomly using PyTorch.

    Parameters:
    - A: Input matrix (torch tensor)
    - rank: Rank of the factorization

    Returns:
    - W: Initialized W matrix (torch tensor)
    - H: Initialized H matrix (torch tensor)
    """
    num_docs, num_terms = A.shape
    device = A.device  
    W = torch.empty(num_docs, rank, device=device).uniform_(1, 2)
    H = torch.empty(rank, num_terms, device=device).uniform_(1, 2)
    return W, H


def nndsvd_initialization(A, rank):
    """
    Initialize matrices W and H using Non-negative Double Singular Value Decomposition (NNDSVD) with PyTorch.

    Parameters:
    - A: Input matrix (torch tensor)
    - rank: Rank of the factorization

    Returns:
    - W: Initialized W matrix (torch tensor)
    - H: Initialized H matrix (torch tensor)
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    V = Vh.T
    device = A.device
    W = torch.zeros((A.shape[0], rank), dtype=A.dtype, device=device)
    H = torch.zeros((rank, A.shape[1]), dtype=A.dtype, device=device)


    W[:, 0] = torch.sqrt(S[0]) * torch.abs(U[:, 0])
    H[0, :] = torch.sqrt(S[0]) * torch.abs(V[:, 0])

    for i in range(1, rank):
        ui = U[:, i]
        vi = V[:, i]

        ui_pos = torch.clamp(ui, min=0)
        ui_neg = torch.clamp(-ui, min=0)
        vi_pos = torch.clamp(vi, min=0)
        vi_neg = torch.clamp(-vi, min=0)

        ui_pos_norm = torch.norm(ui_pos)
        ui_neg_norm = torch.norm(ui_neg)
        vi_pos_norm = torch.norm(vi_pos)
        vi_neg_norm = torch.norm(vi_neg)

        norm_pos = ui_pos_norm * vi_pos_norm
        norm_neg = ui_neg_norm * vi_neg_norm

        if norm_pos >= norm_neg:
            W[:, i] = torch.sqrt(S[i] * norm_pos) * ui_pos / (ui_pos_norm + 1e-10)
            H[i, :] = torch.sqrt(S[i] * norm_pos) * vi_pos / (vi_pos_norm + 1e-10)
        else:
            W[:, i] = torch.sqrt(S[i] * norm_neg) * ui_neg / (ui_neg_norm + 1e-10)
            H[i, :] = torch.sqrt(S[i] * norm_neg) * vi_neg / (vi_neg_norm + 1e-10)

    return W, H

def compute_local_S(W, sigma=1.0):
    """
    Tạo ma trận smoothing S theo locality từ vector W.
    Dùng Gaussian affinity để tăng tính cục bộ cho dnsNMF.

    Parameters:
    - W: torch.Tensor, shape (rank, d) — hàng là vector cơ sở
    - sigma: hệ số điều chỉnh độ mượt

    Returns:
    - S: torch.Tensor, shape (rank, rank)
    """
    with torch.no_grad():
        # Chuẩn hóa W để ổn định
        W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)

        # Tính khoảng cách Euclidean giữa các hàng
        dist = torch.cdist(W_norm, W_norm, p=2)  # shape: (k, k)

        # Tính affinity theo Gaussian
        affinity = torch.exp(-dist ** 2 / (2 * sigma ** 2))

        # Chuẩn hóa theo hàng (row stochastic)
        S = affinity / (affinity.sum(dim=1, keepdim=True) + 1e-8)
    return S

def multiplicative_update_nsnmf(A, k, theta, max_iter, init_mode='nndsvd', lambda_sparseness=0.1):
    """
    Multiplicative Update algorithm for Non-smooth Non-negative Matrix Factorization using PyTorch.
    """
    if init_mode == 'random':
        W, H = random_initialization(A, k)
    elif init_mode == 'nndsvd':
        W, H = nndsvd_initialization(A, k)
    else:
        raise ValueError("Invalid init_mode. Use 'random' or 'nndsvd'.")

    device = A.device
    dtype = A.dtype

    # S = (1 - theta) * torch.eye(k, device=device, dtype=dtype) + (theta / k) * torch.ones((k, k), device=device, dtype=dtype)
    S = compute_local_S(W.T, sigma=1.0)


    norms = []
    epsilon = 1.0e-10

    for _ in range(max_iter):
        # Update H
        W_TSA = W.t() @ A
        W_TSWH = W.t() @ W @ S @ H
        H = H * (W_TSA / torch.clamp(W_TSWH, min=epsilon))

        # Update W
        AHS_T = A @ H.t() @ S.t()
        WSH_HT = W @ S @ H @ H.t() @ S.t() + lambda_sparseness * torch.sum(W, dim=1, keepdim=True)
        W = W * (AHS_T / torch.clamp(WSH_HT, min=epsilon))

        norm = torch.norm(A - W @ S @ H, p='fro').item()
        norms.append(norm)

    return W, H, S, norms


def deep_nsnmf(X, layers, k_list, theta_list, max_iter, init_mode='nndsvd', lambda_sparseness=0.1):
    """
    Deep Non-smooth Nonnegative Matrix Factorization .
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

    return W_list, H_list, S_list, norms_list


def reconstruct_X(W_list, S_list, H_final):
    device = H_final.device
    A = W_list[0].to(device)

    for i in range(len(W_list) - 1):
        A = A @ S_list[i].to(device) @ W_list[i + 1].to(device)

    return A @ S_list[-1].to(device) @ H_final

