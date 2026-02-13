import numpy as np
import torch
from scipy import linalg

class StreamingStats:
    """
    Streaming mean/cov bez zapisywania wszystkich cech:
    trzymamy sum(x) i sum(xx^T).
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.sum = np.zeros((dim,), dtype=np.float64)
        self.sum_sq = np.zeros((dim, dim), dtype=np.float64)

    def update(self, feats: torch.Tensor):
        x = feats.detach().cpu().numpy().astype(np.float64)
        self.n += x.shape[0]
        self.sum += x.sum(axis=0)
        self.sum_sq += x.T @ x

    def finalize(self):
        mu = self.sum / max(self.n, 1)
        cov = self.sum_sq / max(self.n, 1) - np.outer(mu, mu)
        return mu, cov

def compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)

