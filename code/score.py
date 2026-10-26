"""Score saved DestVI predictions against the known synthetic-spot truth.

Reads resolv_<TAG><SUF>_<mode>_{pred,truth}.csv.gz written by resolvability.py and writes:
  metrics_per_state.csv  full metric battery per state x setting x spot regime
  metrics_per_spot.csv   within-spot Jensen-Shannon divergence and top-1 accuracy vs a floor model
  jsd_by_group.csv       the same, restricted to each compartment
  leakage_*.csv          true-state x predicted-state correlation matrices
  decision_table.csv     worst case over the two regimes and the pass/fail rule

Pass rule (fix CUT before looking at any result). A state passes in a setting if, in BOTH regimes:
  * its own predicted column is the best match for it (diagonal is the row maximum of the leakage matrix),
  * worst-case Pearson r >= CUT and worst-case Lin's CCC >= CUT,
  * RMSE beats a predict-the-mean floor,
  * false-positive weight in spots where the state is absent is below the state's own mean abundance,
  * the within-compartment mix (JSD) beats the floor in both regimes.
PASS_absolute_only is the same rule without the within-compartment mix criterion.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score, average_precision_score

RES = os.environ.get("RES", "results")            # folder holding resolv_<TAG><SUF>_<mode>_{pred,truth}.csv.gz
OUTDIR = os.environ.get("OUTDIR", "scores")       # where the tables are written
os.makedirs(OUTDIR, exist_ok=True)
# settings to score, "TAG:SUFFIX" pairs, e.g. "A:,A:_coarse,B:,B:_coarse". Suffix is empty for the fine run.
CONFIGS = [tuple(x.split(":")) for x in os.environ.get("CONFIGS", "A:,A:_coarse").split(",")]
MODES = ["overlap", "dominant"]
# state-name prefixes that define compartments (the within-compartment mix is scored per prefix)
PREFIXES = tuple(os.environ.get("PREFIXES", "CAF_,DUCTAL_").split(","))
CUT = float(os.environ.get("CUT", "0.20"))        # pass cutoff for worst-case Pearson r and Lin's CCC; set it BEFORE looking at results

def ccc(t, p):
    mt, mp = t.mean(), p.mean()
    vt, vp = t.var(ddof=0), p.var(ddof=0)
    cov = ((t - mt) * (p - mp)).mean()
    den = vt + vp + (mt - mp) ** 2
    return np.nan if den == 0 else 2 * cov / den

rows, srows, logs = [], [], []
for TAG, SUF in CONFIGS:
    mp = f"{RES}/resolv_{TAG}{SUF}_meta.json"
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    for mode in MODES:
        fp = f"{RES}/resolv_{TAG}{SUF}_{mode}_pred.csv.gz"
        ft = f"{RES}/resolv_{TAG}{SUF}_{mode}_truth.csv.gz"
        if not (os.path.exists(fp) and os.path.exists(ft)):
            logs.append(f"MISSING {TAG}{SUF} {mode}")
            continue
        pred = pd.read_csv(fp, index_col=0)
        truth = pd.read_csv(ft, index_col=0).reindex(index=pred.index).fillna(0.0)
        shared = [c for c in truth.columns if c in pred.columns]
        only_t = [c for c in truth.columns if c not in pred.columns]
        only_p = [c for c in pred.columns if c not in truth.columns]
        cfg = f"{TAG}{SUF or '_fine'}"
        logs.append(f"[{cfg} {mode}] spots={len(pred)} shared={len(shared)} truth_only={only_t} pred_only={len(only_p)}")

        T = truth[shared].values.astype(float)
        P = pred[shared].values.astype(float)
        Tn = T / np.clip(T.sum(1, keepdims=True), 1e-12, None)
        Pn = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)

        jsd = np.array([jensenshannon(Tn[i], Pn[i], base=2) ** 2 for i in range(len(Tn))])
        gmean = Tn.mean(0); gmean = gmean / gmean.sum()
        jsd_floor = np.array([jensenshannon(Tn[i], gmean, base=2) ** 2 for i in range(len(Tn))])
        top1 = (Pn.argmax(1) == Tn.argmax(1)).mean()
        top1_floor = (Tn.argmax(1) == gmean.argmax()).mean()
        srows.append(dict(config=cfg, mode=mode, n_spots=len(Tn), n_states=len(shared),
            jsd_mean=jsd.mean(), jsd_median=np.median(jsd),
            jsd_floor_mean=jsd_floor.mean(), jsd_ratio=jsd.mean() / jsd_floor.mean(),
            top1_acc=top1, top1_floor=top1_floor,
            rmse_overall=np.sqrt(((T - P) ** 2).mean()),
            median_umi=meta.get("median_umi"), note=f"truth_only={';'.join(only_t) or 'none'}"))

        cross = pd.DataFrame(index=shared, columns=shared, dtype=float)
        for a in shared:
            for b in shared:
                ta, pb = truth[a].values, pred[b].values
                cross.loc[a, b] = np.nan if (ta.std() == 0 or pb.std() == 0) else np.corrcoef(ta, pb)[0, 1]
        cross.to_csv(f"{OUTDIR}/leakage_{cfg}_{mode}.csv")

        share_stats = {}
        for pref in PREFIXES:
            gc = [c for c in shared if c.startswith(pref)]
            if len(gc) < 2:
                continue
            has = truth[gc].sum(1) > 0
            ts = truth.loc[has, gc].div(truth.loc[has, gc].sum(1), axis=0)
            ps = pred.loc[has, gc].div(pred.loc[has, gc].sum(1) + 1e-12, axis=0)
            for s in gc:
                t_, p_ = ts[s].values, ps[s].values
                share_stats[s] = dict(share_n=int(has.sum()),
                    share_r=np.nan if (t_.std() == 0 or p_.std() == 0) else np.corrcoef(t_, p_)[0, 1],
                    share_ccc=ccc(t_, p_), share_rmse=float(np.sqrt(((t_ - p_) ** 2).mean())),
                    share_rmse_floor=float(t_.std(ddof=0)))

        for s in shared:
            t_ = truth[s].values.astype(float); p_ = pred[s].values.astype(float)
            present = (t_ > 0).astype(int)
            r_ = np.nan if (t_.std() == 0 or p_.std() == 0) else np.corrcoef(t_, p_)[0, 1]
            rho = np.nan if (t_.std() == 0 or p_.std() == 0) else spearmanr(t_, p_).statistic
            slope = np.nan if t_.var(ddof=0) == 0 else ((t_ - t_.mean()) * (p_ - p_.mean())).mean() / t_.var(ddof=0)
            icpt = np.nan if np.isnan(slope) else p_.mean() - slope * t_.mean()
            rmse = float(np.sqrt(((t_ - p_) ** 2).mean())); rmse_floor = float(t_.std(ddof=0))
            auroc = auprc = np.nan
            if 0 < present.sum() < len(present):
                auroc = roc_auc_score(present, p_); auprc = average_precision_score(present, p_)
            absent = present == 0
            fp_w = float(p_[absent].mean()) if absent.sum() > 0 else np.nan
            row = dict(config=cfg, mode=mode, state=s, n_spots=len(t_),
                prevalence=float(present.mean()), mean_true=float(t_.mean()), mean_pred=float(p_.mean()),
                pearson_r=r_, spearman_rho=rho, lins_ccc=ccc(t_, p_),
                calib_slope=slope, calib_intercept=icpt,
                rmse=rmse, rmse_floor=rmse_floor, rmse_ratio=(rmse / rmse_floor if rmse_floor > 0 else np.nan),
                mae=float(np.abs(t_ - p_).mean()),
                auroc=auroc, auprc=auprc, auprc_floor=float(present.mean()),
                fp_weight_in_absent=fp_w, fp_ratio=(fp_w / t_.mean() if t_.mean() > 0 else np.nan),
                diag_is_rowmax=bool(cross.loc[s].idxmax() == s) if cross.loc[s].notna().any() else False,
                rowmax_partner=str(cross.loc[s].idxmax()) if cross.loc[s].notna().any() else "NA",
                rowmax_value=float(cross.loc[s].max()) if cross.loc[s].notna().any() else np.nan)
            row.update(share_stats.get(s, {}))
            rows.append(row)

pd.DataFrame(rows).to_csv(f"{OUTDIR}/metrics_per_state.csv", index=False)
pd.DataFrame(srows).to_csv(f"{OUTDIR}/metrics_per_spot.csv", index=False)
open(f"{OUTDIR}/score_log.txt", "w").write("\n".join(logs) + "\n")

# jsd_by_group
rows2 = []
for TAG, SUF in CONFIGS:
    for mode in MODES:
        fp = f"{RES}/resolv_{TAG}{SUF}_{mode}_pred.csv.gz"
        ft = f"{RES}/resolv_{TAG}{SUF}_{mode}_truth.csv.gz"
        if not os.path.exists(fp):
            continue
        pred = pd.read_csv(fp, index_col=0)
        truth = pd.read_csv(ft, index_col=0).reindex(index=pred.index).fillna(0.0)
        shared = [c for c in truth.columns if c in pred.columns]
        cfg = f"{TAG}{SUF or '_fine'}"

        def block(cols, label):
            if len(cols) < 2:
                return None
            T = truth[cols].values.astype(float); P = pred[cols].values.astype(float)
            keep = T.sum(1) > 0
            T, P = T[keep], P[keep]
            if len(T) < 5:
                return None
            Tn = T / np.clip(T.sum(1, keepdims=True), 1e-12, None)
            Pn = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
            j = np.array([jensenshannon(Tn[i], Pn[i], base=2) ** 2 for i in range(len(Tn))])
            g = Tn.mean(0); g = g / g.sum()
            jf = np.array([jensenshannon(Tn[i], g, base=2) ** 2 for i in range(len(Tn))])
            return dict(config=cfg, mode=mode, group=label, n_states=len(cols),
                n_spots_with_group=int(keep.sum()),
                jsd=j.mean(), jsd_floor=jf.mean(), jsd_ratio=j.mean() / jf.mean(),
                top1=float((Pn.argmax(1) == Tn.argmax(1)).mean()),
                top1_floor=float((Tn.argmax(1) == g.argmax()).mean()))

        groups = [("ALL", shared)]
        groups += [(p.rstrip("_"), [c for c in shared if c.startswith(p)]) for p in PREFIXES]
        groups += [("+".join(p.rstrip("_") for p in PREFIXES), [c for c in shared if c.startswith(PREFIXES)])]
        for label, cols in groups:
            b = block(cols, label)
            if b:
                rows2.append(b)

jsd_df = pd.DataFrame(rows2)
jsd_df.to_csv(f"{OUTDIR}/jsd_by_group.csv", index=False)

# decision table
m = pd.read_csv(f"{OUTDIR}/metrics_per_state.csv")
g = pd.read_csv(f"{OUTDIR}/jsd_by_group.csv")
m = m[m.state.str.startswith(PREFIXES)].copy()
rows3 = []
for (cfg, st), d in m.groupby(["config", "state"]):
    if set(d["mode"]) != {"overlap", "dominant"}:
        continue
    o = d[d["mode"] == "overlap"].iloc[0]; dm = d[d["mode"] == "dominant"].iloc[0]
    worst_r = min(o.pearson_r, dm.pearson_r)
    worst_ccc = min(o.lins_ccc, dm.lins_ccc)
    rowmax_both = bool(o.diag_is_rowmax) and bool(dm.diag_is_rowmax)
    rmse_ok = (o.rmse_ratio < 1) and (dm.rmse_ratio < 1)
    fp_ok = (o.fp_weight_in_absent < o.mean_true) and (dm.fp_weight_in_absent < dm.mean_true)
    auprc_ok = (o.auprc > o.auprc_floor) and (dm.auprc > dm.auprc_floor)
    grp = next(p for p in PREFIXES if st.startswith(p)).rstrip("_")
    gg = g[(g.config == cfg) & (g.group == grp)]
    jsd_ok = bool((gg.jsd_ratio < 1).all()) and len(gg) == 2
    rows3.append(dict(config=cfg, state=st, worst_pearson=worst_r, worst_ccc=worst_ccc,
        rowmax_both=rowmax_both, rmse_beats_floor=rmse_ok, fp_ok=fp_ok, auprc_ok=auprc_ok,
        group_jsd_beats_floor=jsd_ok,
        worst_share_r=min(o.share_r, dm.share_r) if pd.notna(o.share_r) and pd.notna(dm.share_r) else np.nan,
        PASS=rowmax_both and worst_r >= CUT and worst_ccc >= CUT and rmse_ok and fp_ok and jsd_ok,
        PASS_absolute_only=rowmax_both and worst_r >= CUT and worst_ccc >= CUT and rmse_ok and fp_ok))

v = pd.DataFrame(rows3).sort_values(["config", "state"])
v.to_csv(f"{OUTDIR}/decision_table.csv", index=False)
print(v.to_string())