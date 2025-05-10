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

# --- Phần còn lại giữ nguyên như trước ---
def apg_minimize(H0, C, gradient_fn, max_iters=200, tol=1e-6):
    L = torch.linalg.norm(C, 2)**2
    H_k, Y_k, alpha = H0.clone(), H0.clone(), 1.0
    for _ in range(max_iters):
        grad = gradient_fn(Y_k)
        H_next = (Y_k - grad / L).clamp(min=0)
        alpha_next = (1 + (1 + 4*alpha**2)**0.5) / 2
        Y_k = H_next + ((alpha-1)/alpha_next)*(H_next - H_k)
        if (H_next - H_k).norm() / (H_k.norm()+1e-12) < tol:
            break
        H_k, alpha = H_next, alpha_next
    return H_k

def apg_nsnmf(A, k, S, W0, H0, lambda_s=0.1, max_iter=200):
    W, H, norms = W0.clone(), H0.clone(), []
    for _ in range(max_iter):
        # cập nhật H
        C = W @ S
        def grad_H(h): return C.T @ (C @ h - A)
        H = apg_minimize(H, C, grad_H, max_iters=50)

        # cập nhật W (qua W^T)
        Q = S @ H
        C_tilde, M_tilde = Q.T, A.T
        def grad_Wt(Wt): return C_tilde.T @ (C_tilde @ Wt - M_tilde) + lambda_s
        W = apg_minimize(W.T, C_tilde, grad_Wt, max_iters=50).T

        norms.append((A - W @ S @ H).norm().item())
    return W, H, norms

def deep_nsnmf_apg(X, layers, k_list, theta_list,
                   max_iter_inner=200, init_mode='nndsvd', lambda_s=0.1):
    W_list, H_list, S_list, errs = [], [], [], []
    H_prev = X
    for layer in range(layers):
        k, theta = k_list[layer], theta_list[layer]

        # thuần torch NNDSVD khởi tạo
        W0, H0 = nndsvd_initialization(H_prev, k)

        # tính S
        S = compute_local_S(W0.T, sigma=theta)

        # giải lớp này bằng APG
        W, H, norms = apg_nsnmf(H_prev, k, S, W0, H0,
                                lambda_s=lambda_s,
                                max_iter=max_iter_inner)

        W_list.append(W); H_list.append(H); S_list.append(S); errs.append(norms)
        H_prev = H

    return W_list, H_list, S_list, errs