import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from utils.io import load_yaml, ensure_dir, save_json
from utils.seed import seed_all
from data.datasets import get_dataset
from eval.clip_feat import load_openclip, clip_image_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    seed_all(int(cfg["run"]["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = cfg["run"]["outdir"]
    ensure_dir(outdir)

    ds = get_dataset(cfg["data"]["dataset"], cfg["data"]["root"], cfg["data"]["resolution"], train=True)
    dl = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                    num_workers=cfg["data"]["num_workers"], pin_memory=True)

    clip_model = load_openclip(cfg["e11"]["clip_model"], cfg["e11"]["clip_pretrained"], device=device)

    feats = []
    for batch in dl:
        x = batch[0].to(device)
        f = clip_image_features(clip_model, x)
        feats.append(f.cpu())
    feats = torch.cat(feats, dim=0).numpy().astype(np.float32)  # (N, d)
    N, d = feats.shape

    if cfg["e11"].get("save_real_clip_features", True):
        np.save(os.path.join(outdir, f"{cfg['data']['dataset']}_real_clip_feats.npy"),
                feats.astype(np.float16))

    meta = {"N": int(N), "d": int(d), "K_list": cfg["e11"]["K_list"]}
    save_json(meta, os.path.join(outdir, f"{cfg['data']['dataset']}_meta.json"))

    for K in cfg["e11"]["K_list"]:
        km = KMeans(n_clusters=int(K), random_state=int(cfg["run"]["seed"]),
                    n_init="auto", max_iter=int(cfg["e11"]["kmeans_max_iter"]))
        labels = km.fit_predict(feats)
        centroids = km.cluster_centers_.astype(np.float32)

        np.save(os.path.join(outdir, f"{cfg['data']['dataset']}_K{K}_real_labels.npy"), labels.astype(np.int32))
        np.save(os.path.join(outdir, f"{cfg['data']['dataset']}_K{K}_centroids.npy"), centroids.astype(np.float32))

        # cluster sizes
        sizes = np.bincount(labels, minlength=int(K)).astype(int).tolist()
        save_json({"K": int(K), "sizes": sizes},
                  os.path.join(outdir, f"{cfg['data']['dataset']}_K{K}_sizes.json"))

    print("Saved to:", outdir)

if __name__ == "__main__":
    main()

