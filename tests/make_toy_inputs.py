"""Make fake pred/truth tables so you can check that score.py runs end to end.
Truth is sparse random compositions; prediction = truth + noise (so scores are meaningful but synthetic)."""
import os, numpy as np, pandas as pd
rng = np.random.default_rng(0)
os.makedirs("toy_results", exist_ok=True)
states = ["CAF_a", "CAF_b", "DUCTAL_x", "DUCTAL_y", "IMM_z"]
for suf in ["", "_coarse"]:
    sts = states if suf == "" else ["CAF_a", "DUCTAL_x", "IMM_z"]
    for mode in ["overlap", "dominant"]:
        # like the real spots: each spot is 2-10 cells drawn from a region-level mix, so many states are exactly absent
        w = rng.dirichlet(np.ones(len(sts)) * (1.0 if mode == "overlap" else 0.3), size=6)
        T = np.vstack([rng.multinomial(k := rng.integers(2, 11), w[i % 6]) / k for i in range(300)])
        P = np.clip(T + rng.normal(0, 0.08 if suf else 0.15, T.shape), 1e-4, None); P /= P.sum(1, keepdims=True)
        idx = [f"{mode}_{i}" for i in range(300)]
        pd.DataFrame(T, index=idx, columns=sts).to_csv(f"toy_results/resolv_A{suf}_{mode}_truth.csv.gz")
        pd.DataFrame(P, index=idx, columns=sts).to_csv(f"toy_results/resolv_A{suf}_{mode}_pred.csv.gz")
print("toy tables written to toy_results/")
