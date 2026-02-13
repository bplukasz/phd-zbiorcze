import os, json, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e11_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rows_agg = []
    rows_cluster = []

    for root, dirs, files in os.walk(args.e11_dir):
        if "metrics.json" in files:
            p = os.path.join(root, "metrics.json")
            with open(p, "r") as f:
                data = json.load(f)
            agg = data["aggregates"]
            agg = dict(agg)
            agg["ckpt"] = data.get("ckpt", "")
            rows_agg.append(agg)
            for r in data["per_cluster"]:
                r = dict(r)
                r["ckpt"] = data.get("ckpt", "")
                rows_cluster.append(r)

    dfA = pd.DataFrame(rows_agg).sort_values(["K"])
    dfC = pd.DataFrame(rows_cluster).sort_values(["K", "cluster"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dfA.to_csv(args.out.replace(".csv", "_aggregates.csv"), index=False)
    dfC.to_csv(args.out.replace(".csv", "_per_cluster.csv"), index=False)
    print("Saved:", args.out.replace(".csv", "_aggregates.csv"))
    print("Saved:", args.out.replace(".csv", "_per_cluster.csv"))

if __name__ == "__main__":
    main()

