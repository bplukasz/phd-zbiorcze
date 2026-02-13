import os, argparse, math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_distances

from utils.io import load_yaml, ensure_dir, save_json
from utils.seed import seed_all
from models.gan_resnet import Generator
from eval.clip_feat import load_openclip, clip_image_features
from eval.fid_utils import compute_fid_from_stats

def poly_mmd2_unbiased(X, Y, degree=3, gamma=None, coef0=1.0):
    # X: (n,d), Y: (m,d)
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    n = X.shape[0]; m = Y.shape[0]
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx = (gamma * (X @ X.T) + coef0) ** degree
    Kyy = (gamma * (Y @ Y.T) + coef0) ** degree
    Kxy = (gamma * (X @ Y.T) + coef0) ** degree
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return (Kxx.sum() / (n*(n-1)) + Kyy.sum() / (m*(m-1)) - 2.0 * Kxy.mean())

def kid_score(X, Y, subsets=50, subset_size=1000, degree=3, gamma=None, coef0=1.0, seed=0):
    rng = np.random.RandomState(seed)
    n = X.shape[0]; m = Y.shape[0]
    s = min(subset_size, n, m)
    vals = []
    for _ in range(subsets):
        ix = rng.choice(n, size=s, replace=False)
        iy = rng.choice(m, size=s, replace=False)
        vals.append(poly_mmd2_unbiased(X[ix], Y[iy], degree=degree, gamma=gamma, coef0=coef0))
    return float(np.mean(vals)), float(np.std(vals))

def cov_matrix(X):
    X = X.astype(np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(X.shape[0], 1)
    return mu, cov

def tail10(values):
    v = np.array(values, dtype=np.float64)
    k = max(1, int(math.ceil(0.1 * len(v))))
    return float(np.mean(np.sort(v)[-k:]))

def js_divergence(p, q, eps=1e-12):
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a,b): return np.sum(a * np.log(a/b))
    return float(0.5 * kl(p,m) + 0.5 * kl(q,m))

def chunked_cdist_min(A, B, chunk=512):
    # A: (n,d), B: (m,d) numpy float32/64
    A = torch.from_numpy(A).float().cuda()
    B = torch.from_numpy(B).float().cuda()
    mins = []
    for i in range(0, A.size(0), chunk):
        a = A[i:i+chunk]
        d = torch.cdist(a, B)  # (chunk, m)
        mins.append(d.min(dim=1).values.detach().cpu())
    return torch.cat(mins, dim=0).numpy()

def nn_dist_real_real(X, chunk=512):
    # nearest neighbor distance inside set (exclude self)
    X = torch.from_numpy(X).float().cuda()
    mins = []
    for i in range(0, X.size(0), chunk):
        a = X[i:i+chunk]
        d = torch.cdist(a, X)
        # ustawiamy diagonalę tylko dla odpowiednich indeksów
        for j in range(a.size(0)):
            d[j, i+j] = 1e9
        mins.append(d.min(dim=1).values.detach().cpu())
    return torch.cat(mins, dim=0).numpy()

@torch.no_grad()
def generate_fakes(G, z_dim, n, device, batch=256):
    imgs = []
    remaining = n
    while remaining > 0:
        cur = min(batch, remaining)
        z = torch.randn(cur, z_dim, device=device)
        x = G(z)
        imgs.append(x.cpu())
        remaining -= cur
    return torch.cat(imgs, dim=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    seed_all(int(cfg["run"]["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = cfg["run"]["outdir"]
    ensure_dir(outdir)

    # load G
    ckpt = torch.load(args.ckpt, map_location="cpu")
    G = Generator(z_dim=cfg["model"]["z_dim"], ch=cfg["model"]["g_ch"], resolution=cfg["data"]["resolution"]).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # CLIP
    clip_model = load_openclip(cfg["e11"]["clip_model"], cfg["e11"]["clip_pretrained"], device=device)

    cache_dir = cfg["e11"]["cache_dir"]
    dsname = cfg["data"]["dataset"]

    # load real feats once
    real_feats = np.load(os.path.join(cache_dir, f"{dsname}_real_clip_feats.npy")).astype(np.float32)  # (N,d)
    N, d = real_feats.shape

    results_all = []

    for K in cfg["e11"]["K_list"]:
        labels_real = np.load(os.path.join(cache_dir, f"{dsname}_K{K}_real_labels.npy")).astype(np.int32)
        centroids = np.load(os.path.join(cache_dir, f"{dsname}_K{K}_centroids.npy")).astype(np.float32)  # (K,d)

        # generate fakes + compute clip feats
        x_fake = generate_fakes(G, cfg["model"]["z_dim"], int(cfg["e11"]["n_fake"]), device, batch=cfg["data"]["batch_size"])
        # move to GPU in batches for CLIP
        feats_fake = []
        for i in range(0, x_fake.size(0), cfg["data"]["batch_size"]):
            xf = x_fake[i:i+cfg["data"]["batch_size"]].to(device)
            ff = clip_image_features(clip_model, xf)
            feats_fake.append(ff.cpu())
        feats_fake = torch.cat(feats_fake, dim=0).numpy().astype(np.float32)

        # assign fake to clusters (cosine distance to centroids)
        # cosine_distances works on CPU; OK for 20k x 100
        dist = cosine_distances(feats_fake, centroids)  # (Nf,K)
        hard = dist.argmin(axis=1).astype(np.int32)

        # hist + JS
        hist_real = np.bincount(labels_real, minlength=int(K)).astype(np.float64)
        hist_fake = np.bincount(hard, minlength=int(K)).astype(np.float64)
        js = js_divergence(hist_real, hist_fake)

        per_cluster = []
        fid_list = []
        kid_list = []
        cov_list = []
        weights = []

        min_cluster = int(cfg["e11"]["min_cluster_size"])

        for c in range(int(K)):
            idx_r = np.where(labels_real == c)[0]
            n_r = idx_r.shape[0]
            if n_r < min_cluster:
                continue

            if cfg["e11"]["assignment"] == "topM":
                M = int(cfg["e11"]["topM"])
                idx_f = np.argsort(dist[:, c])[:min(M, dist.shape[0])]
            else:
                idx_f = np.where(hard == c)[0]

            n_f = idx_f.shape[0]
            if n_f < min_cluster:
                continue

            Xr = real_feats[idx_r]
            Xf = feats_fake[idx_f]

            mu_r, cov_r = cov_matrix(Xr)
            mu_f, cov_f = cov_matrix(Xf)
            fid_c = compute_fid_from_stats(mu_r, cov_r, mu_f, cov_f)
            fid_list.append(fid_c)

            kid_c = None
            kid_std = None
            if cfg["e11"]["kid"]["enabled"]:
                kid_c, kid_std = kid_score(
                    Xr, Xf,
                    subsets=int(cfg["e11"]["kid"]["subsets"]),
                    subset_size=int(cfg["e11"]["kid"]["subset_size"]),
                    degree=int(cfg["e11"]["kid"]["degree"]),
                    gamma=cfg["e11"]["kid"]["gamma"],
                    coef0=float(cfg["e11"]["kid"]["coef0"]),
                    seed=int(cfg["run"]["seed"])
                )
                kid_list.append(kid_c)

            cov_c = None
            if cfg["e11"]["coverage"]["enabled"]:
                mr = int(cfg["e11"]["coverage"]["max_real_per_cluster"])
                mf = int(cfg["e11"]["coverage"]["max_fake_per_cluster"])
                chunk = int(cfg["e11"]["coverage"]["chunk"])
                q = float(cfg["e11"]["coverage"]["threshold_percentile"])

                # subsample for speed
                rng = np.random.RandomState(int(cfg["run"]["seed"]))
                sr = idx_r if n_r <= mr else rng.choice(idx_r, size=mr, replace=False)
                sf = idx_f if n_f <= mf else rng.choice(idx_f, size=mf, replace=False)

                R = real_feats[sr].astype(np.float32)
                Gf = feats_fake[sf].astype(np.float32)

                # threshold from real-real nn distances
                rr = nn_dist_real_real(R, chunk=chunk)
                t = np.percentile(rr, q)

                # coverage: real->fake nn distances
                rg = chunked_cdist_min(R, Gf, chunk=chunk)
                cov_c = float((rg < t).mean())
                cov_list.append(cov_c)

            weights.append(n_r)

            per_cluster.append({
                "K": int(K),
                "cluster": int(c),
                "n_real": int(n_r),
                "n_fake": int(n_f),
                "fid": float(fid_c),
                "kid": None if kid_c is None else float(kid_c),
                "kid_std": None if kid_std is None else float(kid_std),
                "coverage": cov_c,
            })

        # aggregates
        weights = np.array(weights, dtype=np.float64)
        w = weights / max(weights.sum(), 1.0)

        agg = {"K": int(K), "js_hist": float(js)}
        if len(fid_list) > 0:
            fid_arr = np.array(fid_list, dtype=np.float64)
            agg["cluster_fid_mean_w"] = float((w * fid_arr).sum())
            agg["cluster_fid_worst"] = float(fid_arr.max())
            agg["cluster_fid_tail10"] = tail10(fid_arr)
        if len(kid_list) > 0:
            kid_arr = np.array(kid_list, dtype=np.float64)
            agg["cluster_kid_mean_w"] = float((w * kid_arr).sum())
            agg["cluster_kid_worst"] = float(kid_arr.max())
            agg["cluster_kid_tail10"] = tail10(kid_arr)
        if len(cov_list) > 0:
            cov_arr = np.array(cov_list, dtype=np.float64)
            agg["coverage_mean_w"] = float((w * cov_arr).sum())
            agg["coverage_worst"] = float(cov_arr.min())  # worst = min coverage
            agg["coverage_tail10"] = float(np.mean(np.sort(cov_arr)[:max(1, int(np.ceil(0.1*len(cov_arr))))]))

        # save per K
        outK = os.path.join(outdir, f"e11_{dsname}_K{K}")
        ensure_dir(outK)
        save_json({
            "ckpt": args.ckpt,
            "K": int(K),
            "aggregates": agg,
            "per_cluster": per_cluster,
        }, os.path.join(outK, "metrics.json"))

        results_all.append(agg)

    save_json({"ckpt": args.ckpt, "results": results_all},
              os.path.join(outdir, "aggregates_allK.json"))
    print("Saved E11 results to:", outdir)

if __name__ == "__main__":
    main()

