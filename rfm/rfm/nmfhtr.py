'''Construct kernel model with EigenPro optimizer.'''
import collections
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from .svd import nystrom_kernel_svd
from .nmf import deep_nsnmf
#248,288
def asm_nmf_fn(samples, map_fn, rank=10, max_iter=100, init_mode='nndsvd', verbose=True, device="cuda"):
    """
    Approximate kernel matrix using dnsNMF.
    """
    kernel_matrix = map_fn(samples, samples).cpu().numpy()
    kernel_matrix = np.maximum(kernel_matrix, 0)  # Ensure non-negativity

    # Define parameters for dnsNMF
    layers = 2  # Example: 2-layer architecture
    k_list = [rank, rank]  # Same rank for both layers
    theta_list = [0.5, 0.5]  # Example smoothing parameters

    W_list, H_list, S_list, norms_list = deep_nsnmf(
        kernel_matrix, layers, k_list, theta_list, max_iter, init_mode
    )

    if verbose:
        print(f"dnsNMF completed with final reconstruction error: {norms_list[-1][-1]:.4f}")

    # Return the top-layer W and H
    W, H = W_list[-1], H_list[-1]
    return torch.from_numpy(W).float().to(device), torch.from_numpy(H).float().to(device), norms_list

class KernelModel(nn.Module):
    '''Fast Kernel Regression using EigenPro iteration.'''
    def __init__(self, kernel_fn, centers, y_dim, device="cuda"):
        super(KernelModel, self).__init__()
        self.kernel_fn = kernel_fn
        self.n_centers, self.x_dim = centers.shape
        self.device = device
        self.pinned_list = []

        self.centers = self.tensor(centers, release=True, dtype=centers.dtype)
        self.weight = self.tensor(torch.zeros(
            self.n_centers, y_dim), release=True, dtype=centers.dtype)
        
        self.save_kernel_matrix = False
        self.kernel_matrix = [] if self.save_kernel_matrix else None

    def __del__(self):
        for pinned in self.pinned_list:
            _ = pinned.to("cpu")

    def tensor(self, data, dtype=None, release=False):
        if torch.is_tensor(data) and data.device == self.device:
            tensor = data.detach()
        elif torch.is_tensor(data):
            tensor = data.detach().to(self.device)
        else:
            tensor = torch.tensor(data, requires_grad=False, device=self.device)

        if release:
            self.pinned_list.append(tensor)
        return tensor

    def get_kernel_matrix(self, batch, batch_ids, samples=None, sample_ids=None):
        if batch_ids is not None and self.save_kernel_matrix and isinstance(self.kernel_matrix, torch.Tensor):
            if samples is None or sample_ids is None:
                kmat = self.kernel_matrix[batch_ids]
            else:
                kmat = self.kernel_matrix[batch_ids][:, sample_ids]
        else:
            if samples is None or sample_ids is None:
                kmat = self.kernel_fn(batch, self.centers)
            else:
                kmat = self.kernel_fn(batch, samples)
        return kmat

    def forward(self, batch, batch_ids=None, weight=None, save_kernel_matrix=False):
        if weight is None:
            weight = self.weight
        kmat = self.get_kernel_matrix(batch, batch_ids)
        if save_kernel_matrix: # only call if self.kernel_matrix is a list
            self.kernel_matrix.append((batch_ids.cpu(), kmat.cpu()))
        pred = kmat.mm(weight)
        return pred

    def primal_gradient(self, batch, labels, weight, batch_ids, save_kernel_matrix=False):
        pred = self.forward(batch, batch_ids, weight, save_kernel_matrix)
        grad = pred - labels
        return grad

    @staticmethod
    def _compute_opt_params(bs, bs_gpu, beta, top_eigval):
        if bs is None:
            bs = min(np.int32(beta / top_eigval + 1), bs_gpu)

        if bs < beta / top_eigval + 1:
            eta = bs / beta
        else:
            eta = 0.99 * 2 * bs / (beta + (bs - 1) * top_eigval)
        return bs, float(eta)

    def nmf_iterate(self, samples, x_batch, y_batch, nmf_fn,
                         eta, sample_ids, batch_ids, save_kernel_matrix=False):
        # update random coordiate block (for mini-batch)
        grad = self.primal_gradient(x_batch, y_batch, self.weight, batch_ids, save_kernel_matrix)
        self.weight.index_add_(0, batch_ids, -eta * grad)

        # update fixed coordinate block (for EigenPro)
        kmat = self.get_kernel_matrix(x_batch, batch_ids, samples, sample_ids)
        correction = nmf_fn(grad, kmat)
        self.weight.index_add_(0, sample_ids, eta * correction)
        return

    def evaluate(self, X_eval, y_eval, bs=None, metrics=('mse')):
        p_list = []
        n_eval, _ = X_eval.shape

        if bs is None:
            n_batch = 1
        else:
            n_batch = n_eval // min(n_eval, bs)

        for batch_ids in np.array_split(np.arange(n_eval), n_batch):
            x_batch = self.tensor(X_eval[batch_ids], dtype=X_eval.dtype)
            p_batch = self.forward(x_batch)
            p_list.append(p_batch)
        p_eval = torch.concat(p_list, dim=0).to(self.device)

        eval_metrics = collections.OrderedDict()
        if 'mse' in metrics:
            eval_metrics['mse'] = torch.mean(torch.square(p_eval - y_eval)).item()
        if 'multiclass-acc' in metrics:
            y_class = torch.argmax(y_eval, dim=-1)
            p_class = torch.argmax(p_eval, dim=-1)
            eval_metrics['multiclass-acc'] = torch.sum(y_class == p_class).item() / len(y_eval)
        if 'binary-acc' in metrics:
            y_class = torch.where(y_eval > 0.5, 1, 0).reshape(-1)
            p_class = torch.where(p_eval > 0.5, 1, 0).reshape(-1)
            eval_metrics['binary-acc'] = torch.sum(y_class == p_class).item() / len(y_eval)
        if 'f1' in metrics:
            y_class = torch.where(y_eval > 0.5, 1, 0).reshape(-1)
            p_class = torch.where(p_eval > 0.5, 1, 0).reshape(-1)
            eval_metrics['f1'] = torch.mean(2 * (y_class * p_class) / (y_class + p_class + 1e-8)).item()
        if 'auc' in metrics:
            eval_metrics['auc'] = roc_auc_score(y_eval.cpu().flatten(), p_eval.cpu().flatten())

        return eval_metrics

    def fit(self, X_train, y_train, X_val, y_val, epochs, mem_gb,
        n_subsamples=None, bs=None, eta=None, n_eval=1000, 
        run_epoch_eval=True, lr_scale=1, verbose=True, seed=1, 
        classification=False, threshold=1e-5, 
        early_stopping_window_size=7, eval_interval=1):

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        n_eval = min(n_eval, len(X_train), len(X_val))
        train_eval_ids = np.random.choice(len(X_train), n_eval, replace=False)

        X_train_eval = X_train[train_eval_ids].clone()
        y_train_eval = y_train[train_eval_ids].clone()

        assert(len(X_train) == len(y_train))
        assert(len(X_val) == len(y_val))

        metrics = ('mse',)
        if classification:
            if y_train.shape[-1] == 1:
                metrics += ('binary-acc', 'f1', 'auc')
            else:
                metrics += ('multiclass-acc',)

        n_samples, n_labels = y_train.shape
        if n_subsamples is None:
            n_subsamples = min(n_samples, 12000)
        n_subsamples = min(n_subsamples, n_samples)

        mem_bytes = (mem_gb - 1) * 1024**3  # preserve 1GB
        bsizes = np.arange(n_subsamples)
        mem_usages = ((self.x_dim + 3 * n_labels + bsizes + 1)
                    * self.n_centers + n_subsamples * 1000) * 4
        bs_gpu = np.sum(mem_usages < mem_bytes)

        np.random.seed(seed)
        sample_ids_np = np.random.choice(n_samples, n_subsamples, replace=False)
        sample_ids = self.tensor(sample_ids_np)
        samples = self.centers[sample_ids]

        # Use NMF instead of SVD
        W_nmf, H_nmf, nmf_norms = asm_nmf_fn(samples, self.kernel_fn, rank=n_labels, verbose=verbose)
        

        def nmf_projection_fn(grad, kmat):
            return W_nmf @ (H_nmf @ grad).to(self.device)

        # Learning rate
        if eta is None:
            eta = 1.0
        eta = self.tensor(lr_scale * eta / (bs or bs_gpu), dtype=torch.float)

        if bs is None:
            bs = bs_gpu

        if verbose:
            print(f"Using NMF-based projection with rank={W_nmf.shape[1]}, eta={eta.item():.4f}, bs={bs}")

        res = dict()
        initial_epoch = 0
        train_sec = 0
        best_weights = None
        best_metric = 0 if classification else float('inf')
        val_loss_history = []

        for epoch in range(epochs):
            start = time.time()
            epoch_ids = np.random.permutation(n_samples)
            save_kernel_matrix = epoch == 1 and self.save_kernel_matrix

            for batch_ids in tqdm(np.array_split(epoch_ids, n_samples // bs)):
                batch_ids = self.tensor(batch_ids)
                x_batch = self.tensor(X_train[batch_ids], dtype=X_train.dtype)
                y_batch = self.tensor(y_train[batch_ids], dtype=y_train.dtype)

                self.nmf_iterate(samples, x_batch, y_batch,
                                    nmf_projection_fn, eta,
                                    sample_ids, batch_ids,
                                    save_kernel_matrix=save_kernel_matrix)
                del x_batch, y_batch, batch_ids

            if save_kernel_matrix:
                print("Storing kernel matrix")
                concat_matrix = torch.cat([pair[1] for pair in self.kernel_matrix], dim=0)
                all_batch_ids = torch.cat([pair[0] for pair in self.kernel_matrix])
                _, sort_indices = torch.sort(all_batch_ids)
                self.kernel_matrix = concat_matrix[sort_indices].to(self.device)

            if run_epoch_eval and epoch % eval_interval == 0:
                train_sec += time.time() - start

                tr_score = self.evaluate(X_train_eval, y_train_eval, bs=bs, metrics=metrics)
                tv_score = self.evaluate(X_val, y_val, bs=bs, metrics=metrics)

                if verbose:
                    out_str = f"({epoch} epochs, {train_sec:.1f}s)\ttrain mse: {tr_score['mse']:.4f} \tval mse: {tv_score['mse']:.4f}"
                    if classification:
                        if 'binary-acc' in tr_score:
                            out_str += f"\ttrain acc: {tr_score['binary-acc']:.4f} \tval acc: {tv_score['binary-acc']:.4f}"
                        else:
                            out_str += f"\ttrain acc: {tr_score['multiclass-acc']:.4f} \tval acc: {tv_score['multiclass-acc']:.4f}"
                        if 'f1' in tr_score:
                            out_str += f"\tf1: {tv_score['f1']:.4f}"
                        if 'auc' in tr_score:
                            out_str += f"\tauc: {tv_score['auc']:.4f}"
                    print(out_str)

                res[epoch] = (tr_score, tv_score, train_sec)

                # Early stopping check
                val_metric_key = (
                    'auc' if 'auc' in tv_score else
                    'binary-acc' if 'binary-acc' in tv_score else
                    'multiclass-acc' if 'multiclass-acc' in tv_score else
                    'mse'
                )
                current_metric = tv_score[val_metric_key]
                improved = (current_metric > best_metric) if classification else (current_metric < best_metric)
                val_loss_history.append(not improved)

                if improved:
                    best_metric = current_metric
                    best_weights = self.weight.cpu().clone()
                    val_loss_history = []
                    print(f"New best {val_metric_key}: {best_metric:.4f}")

                if len(val_loss_history) > early_stopping_window_size:
                    val_loss_history.pop(0)
                    if sum(val_loss_history) / len(val_loss_history) >= 0.8:
                        print("Early stopping triggered")
                        break

                if tr_score['mse'] < threshold:
                    break

            initial_epoch = epoch

        self.weight = best_weights.to(self.device)

        if self.kernel_matrix is not None:
            del self.kernel_matrix

        return res
