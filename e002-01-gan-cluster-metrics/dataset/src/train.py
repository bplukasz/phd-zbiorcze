import os, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils.io import load_yaml, ensure_dir, save_json
from utils.seed import seed_all
from utils.logger import CSVLogger
from data.datasets import get_dataset, ClusterDropWrapper
from models.gan_resnet import Generator, Discriminator
from models.ema import EMA
from eval.inception_feat import InceptionFeatureExtractor
from eval.fid_utils import compute_fid_from_stats, StreamingStats

def hinge_d_loss(d_real, d_fake):
    return (F.relu(1. - d_real).mean() + F.relu(1. + d_fake).mean())

def hinge_g_loss(d_fake):
    return (-d_fake).mean()

@torch.no_grad()
def sample_grid(G, z_dim, device, path, n=64):
    z = torch.randn(n, z_dim, device=device)
    x = G(z).cpu()
    grid = make_grid(x, nrow=8, normalize=True, value_range=(-1, 1))
    save_image(grid, path)

@torch.no_grad()
def compute_global_fid(G, z_dim, device, real_loader, n_fake=10000):
    # Inception stats for real (streaming, without storing all features)
    feat = InceptionFeatureExtractor(device=device)
    real_stats = StreamingStats(dim=2048)
    for (x, *_) in real_loader:
        x = x.to(device)
        f = feat(x)  # (B,2048)
        real_stats.update(f)

    # fake stats
    fake_stats = StreamingStats(dim=2048)
    bs = real_loader.batch_size
    remaining = n_fake
    while remaining > 0:
        cur = min(bs, remaining)
        z = torch.randn(cur, z_dim, device=device)
        xg = G(z)
        fg = feat(xg)
        fake_stats.update(fg)
        remaining -= cur

    mu_r, sig_r = real_stats.finalize()
    mu_f, sig_f = fake_stats.finalize()
    fid = compute_fid_from_stats(mu_r, sig_r, mu_f, sig_f)
    return float(fid)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    run_name = cfg["run"]["name"]
    outdir = os.path.join(cfg["run"]["outdir"], run_name)
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "checkpoints"))
    ensure_dir(os.path.join(outdir, "samples"))

    seed_all(int(cfg["run"]["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = get_dataset(cfg["data"]["dataset"], cfg["data"]["root"], cfg["data"]["resolution"], train=True)

    # optional: drop clusters (sanity-check)
    drop_clusters = cfg["data"].get("drop_clusters", [])
    drop_fraction = cfg["data"].get("drop_fraction", 0.0)
    if len(drop_clusters) > 0 and drop_fraction > 0:
        # Oczekujemy pliku z klastrami w cache E11:
        # /kaggle/working/e11_cache/cifar10_K50_real_labels.npy
        cache_guess = os.path.join("/kaggle/working/e11_cache", f"{cfg['data']['dataset']}_K50_real_labels.npy")
        if not os.path.exists(cache_guess):
            raise FileNotFoundError(
                f"Brak labels do drop: {cache_guess}. "
                "Najpierw uruchom eval/precompute_real.py dla K=50."
            )
        import numpy as np
        labels = np.load(cache_guess)
        ds = ClusterDropWrapper(ds, labels, drop_clusters, drop_fraction, seed=int(cfg["run"]["seed"]))

    dl = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                    num_workers=cfg["data"]["num_workers"], drop_last=True, pin_memory=True)

    G = Generator(z_dim=cfg["model"]["z_dim"], ch=cfg["model"]["g_ch"], resolution=cfg["data"]["resolution"]).to(device)
    D = Discriminator(ch=cfg["model"]["d_ch"], resolution=cfg["data"]["resolution"],
                      sn_enabled=bool(cfg["model"]["spectral_norm"])).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=cfg["train"]["lr_g"], betas=tuple(cfg["train"]["betas"]))
    optD = torch.optim.Adam(D.parameters(), lr=cfg["train"]["lr_d"], betas=tuple(cfg["train"]["betas"]))

    ema = None
    if cfg["train"]["ema"]["enabled"]:
        ema = EMA(G, beta=float(cfg["train"]["ema"]["beta"]))

    logger = CSVLogger(os.path.join(outdir, "log.csv"),
                       fieldnames=["step", "loss_d", "loss_g", "fid", "time"])

    it = iter(dl)
    steps = int(cfg["train"]["steps"])
    n_critic = int(cfg["train"]["n_critic"])

    t0 = time.time()
    for step in range(1, steps + 1):
        # --- D updates
        for _ in range(n_critic):
            try:
                x_real = next(it)[0]
            except StopIteration:
                it = iter(dl)
                x_real = next(it)[0]
            x_real = x_real.to(device)

            z = torch.randn(x_real.size(0), cfg["model"]["z_dim"], device=device)
            with torch.no_grad():
                x_fake = G(z)

            d_real = D(x_real)
            d_fake = D(x_fake)
            loss_d = hinge_d_loss(d_real, d_fake)

            optD.zero_grad(set_to_none=True)
            loss_d.backward()
            optD.step()

        # --- G update
        z = torch.randn(cfg["data"]["batch_size"], cfg["model"]["z_dim"], device=device)
        x_fake = G(z)
        d_fake = D(x_fake)
        loss_g = hinge_g_loss(d_fake)

        optG.zero_grad(set_to_none=True)
        loss_g.backward()
        optG.step()

        if ema is not None:
            ema.update(G)

        # --- logging / samples / ckpt / eval
        fid_val = None
        if step % int(cfg["log"]["sample_every"]) == 0:
            sample_grid(G, cfg["model"]["z_dim"], device, os.path.join(outdir, "samples", f"grid_{step:06d}.png"))

        if step % int(cfg["log"]["ckpt_every"]) == 0:
            ckpt = {
                "step": step,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "cfg": cfg,
            }
            if ema is not None:
                ckpt["ema"] = ema.shadow
            torch.save(ckpt, os.path.join(outdir, "checkpoints", f"ckpt_{step:06d}.pt"))

        if step % int(cfg["log"]["eval_every"]) == 0:
            # global FID: użyj EMA jeśli jest
            G_eval = G
            backup = None
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in G.state_dict().items()}
                ema.copy_to(G)
                G_eval = G

            real_eval_dl = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                                      num_workers=cfg["data"]["num_workers"], drop_last=False, pin_memory=True)
            fid_val = compute_global_fid(G_eval, cfg["model"]["z_dim"], device, real_eval_dl,
                                         n_fake=int(cfg["log"]["n_eval_fake"]))

            if backup is not None:
                G.load_state_dict(backup)

        logger.log({
            "step": step,
            "loss_d": float(loss_d.item()),
            "loss_g": float(loss_g.item()),
            "fid": "" if fid_val is None else fid_val,
            "time": time.time() - t0
        })

    save_json({"status": "done", "outdir": outdir}, os.path.join(outdir, "done.json"))

if __name__ == "__main__":
    main()

