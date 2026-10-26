"""Held-out synthetic-spot benchmark for spatial deconvolution (DestVI).

1. Hold out whole patients (or, if that leaves too few cells, half of each patient's cells).
2. Train the single-cell model (CondSCVI) on the remaining cells only.
3. Build synthetic spots by summing 2-10 held-out cells per spot, then binomially down-sampling
   to a spot depth drawn from N(VIS_MEAN, VIS_SD). Two spot regimes: "overlap" and "dominant".
4. Fit DestVI on the synthetic spots and save predicted vs true proportions.

Everything is configured through environment variables; see README.md. Scoring: score.py.
"""

import os, sys, time, json, gzip
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
import anndata as ad

ROOT = os.environ.get("DECONV_ROOT", ".")          # working directory for inputs, models, outputs
EXP  = f"{ROOT}/export"                          # input: one sub-folder per TAG (see README)
OUT  = f"{ROOT}/results"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{ROOT}/models", exist_ok=True)

def load_export(tag):
    d = f"{EXP}/{tag}"
    X = sp.csr_matrix(scipy.io.mmread(f"{d}/counts.mtx.gz").T)   # mtx is genes x cells
    genes = [l.strip() for l in open(f"{d}/genes.txt")]
    cells = [l.strip() for l in open(f"{d}/cells.txt")]
    meta  = pd.read_csv(f"{d}/meta.csv", index_col=0)
    a = ad.AnnData(X=X, obs=meta.loc[cells].copy(), var=pd.DataFrame(index=genes))
    a.layers["counts"] = a.X.copy()
    return a

def fit(model, max_epochs, patience=12, batch_size=1024, tag=""):
    """Train with early stopping; fall back to a fixed budget if the kwargs are unsupported."""
    t0 = time.time()
    kw = dict(max_epochs=max_epochs, accelerator="cpu", batch_size=batch_size,
              enable_progress_bar=False)
    try:
        model.train(train_size=0.9, early_stopping=True,
                    early_stopping_monitor="elbo_validation",
                    early_stopping_patience=patience, **kw)
        mode = "early_stopping"
    except TypeError as e:
        print("early-stopping kwargs rejected:", e, "-> fixed budget", flush=True)
        model.train(**kw)
        mode = "fixed"
    el = model.history["elbo_train"].iloc[:, 0]
    dt = time.time() - t0
    print(f"[{tag}] {mode}: {len(el)} epochs in {dt/60:.1f} min "
          f"({dt/max(1,len(el)):.1f} s/epoch), elbo {el.iloc[0]:.1f} -> {el.iloc[-1]:.1f}", flush=True)
    return {"mode": mode, "epochs_run": int(len(el)), "minutes": dt/60,
            "sec_per_epoch": dt/max(1, len(el)),
            "elbo_first": float(el.iloc[0]), "elbo_last": float(el.iloc[-1])}


def prepare_sc_model(m):
    """Work around an scvi-tools 1.5.1 bug in DestVI.from_rna_model.

    from_rna_model reads attr_dict["init_params_"]["kwargs"]["module_kwargs"]["prior"],
    but CondSCVI.__init__ only puts into module_kwargs what the caller passed, so the key
    is absent and it raises KeyError: 'prior'. VAEC's default prior is "normal" and the
    trained state dict has no prior_means/prior_log_std/prior_logits, confirming no
    mixture-of-Gaussians prior was ever built. We therefore write the prior the module
    actually used, which routes from_rna_model down the VampPrior branch - the published
    DestVI behaviour. The installed package is not modified.
    """
    k = m.init_params_.setdefault("kwargs", {}).setdefault("module_kwargs", {})
    actual = getattr(m.module, "prior", "normal")
    k.setdefault("prior", actual)
    print(f"[workaround] module_kwargs['prior'] = {k['prior']!r} "
          f"(state dict has MoG params: {'prior_means' in dict(m.module.named_parameters())})",
          flush=True)
    return m

# ---- scvi-tools 1.5.1 workaround #2 -------------------------------------------------
# DestVI.from_rna_model takes the VampPrior branch when the CondSCVI prior is not "mog".
# That branch calls CondSCVI.get_vamp_prior(), which returns numpy arrays, and then hands
# them to MRDeconv, which does register_buffer() on them -> TypeError: cannot assign
# 'numpy.ndarray' object to buffer 'mean_vprior'. We wrap the method so it returns float32
# tensors, and cache the result per model: from_rna_model recomputes it for every section
# and it runs KMeans over the whole reference each time. The package is unmodified.
def _install_vamp_patch(scvi, torch, np):
    _orig = scvi.model.CondSCVI.get_vamp_prior
    def get_vamp_prior(self, adata=None, p=10):
        cached = getattr(self, "_vamp_cache", None)
        if cached is not None and cached[0] == p:
            return cached[1]
        out = _orig(self, adata, p)
        out = {k: (v.detach() if torch.is_tensor(v)
                   else torch.as_tensor(np.asarray(v), dtype=torch.float32))
               for k, v in out.items()}
        self._vamp_cache = (p, out)
        return out
    scvi.model.CondSCVI.get_vamp_prior = get_vamp_prior

import torch, scvi
torch.set_num_threads(int(os.environ.get("NTHREADS", "10")))
scvi.settings.seed = 0
_install_vamp_patch(scvi, torch, np)
rng = np.random.default_rng(0)
TAG        = os.environ["TAG"]
STATE_COL  = os.environ.get("STATE_COL", "state")
SUF        = os.environ.get("SUF", "")
SC_EPOCHS  = int(os.environ.get("SC_EPOCHS", "300"))
ST_EPOCHS  = int(os.environ.get("ST_EPOCHS", "2000"))
VIS_MEAN   = float(os.environ["VIS_MEAN"])
VIS_SD     = float(os.environ["VIS_SD"])

a = load_export(TAG)
CAP = int(os.environ.get("RESOLV_CAP", "3000"))
_keep = np.concatenate([v if len(v) <= CAP else rng.choice(v, CAP, replace=False)
                        for v in a.obs.groupby(STATE_COL, observed=True).indices.values()])
a = a[np.sort(_keep)].copy()
print(f"resolvability runs on a capped subsample: {a.shape} (cap {CAP} per state)")
IN_SCOPE_PREFIX = tuple(os.environ.get("IN_SCOPE_PREFIX", "CAF_,DUCTAL_").split(","))   # states that get scored
states = sorted(a.obs[STATE_COL].unique())
focus  = [s for s in states if s.startswith(IN_SCOPE_PREFIX)]
print(f"[{TAG}] {a.shape}; {len(states)} states, {len(focus)} in scope")

# ---- 4a. patient split, held out ----
pats = np.array(sorted(a.obs["patient"].unique()))
n_test = max(1, len(pats)//5)
test_p = rng.choice(pats, size=n_test, replace=False)
is_test = a.obs["patient"].isin(test_p).values
print(f"patients {len(pats)} -> {n_test} held out; test cells {is_test.sum()}")
tab = pd.crosstab(a.obs[STATE_COL], is_test)
tab.to_csv(f"{OUT}/resolv_{TAG}{SUF}_split.csv")
missing = [s for s in focus if s not in tab.index or tab.loc[s].min() < 25]
if missing:
    print("states with <25 cells on one side of the patient split:", missing)
    print("falling back to a 50/50 within-patient split (weaker test) for ALL states")
    is_test = np.zeros(a.n_obs, bool)
    for (p, s), idx in a.obs.groupby(["patient",STATE_COL], observed=True).indices.items():
        if len(idx) >= 2:
            is_test[rng.choice(idx, size=len(idx)//2, replace=False)] = True
    split_kind = "within_patient_50_50"
else:
    split_kind = "held_out_patients"
print("split kind:", split_kind, " test cells:", int(is_test.sum()))

train = a[~is_test].copy(); test = a[is_test].copy()

# ---- 4b/4c. synthetic Visium spots built from the held-out cells ----
# procedure follows synthspot's generate_synthetic_visium: 2-10 cells per spot, counts summed,
# then binomially downsampled to a Visium-like depth. Regions give spots distinct compositions.
def make_synthetic(adata, mode, n_regions=5, spots_min=50, spots_max=200, seed=1):
    r = np.random.default_rng(seed)
    st = adata.obs[STATE_COL].values
    by_state = {s: np.where(st == s)[0] for s in np.unique(st)}
    by_state = {s: v for s, v in by_state.items() if len(v) > 0}
    keys = sorted(by_state)
    X = adata.layers["counts"]
    rows, truth, names = [], [], []
    for reg in range(n_regions):
        if mode == "dominant":
            w = r.dirichlet(np.full(len(keys), 0.25))
            dom = r.integers(len(keys)); w = w * 0.25; w[dom] += 0.75; w /= w.sum()
        else:
            w = r.dirichlet(np.full(len(keys), 1.0))
        n_spots = int(r.integers(spots_min, spots_max + 1))
        for sp_i in range(n_spots):
            k = int(r.integers(2, 11))
            picks = r.choice(len(keys), size=k, p=w)
            cells = [r.choice(by_state[keys[j]]) for j in picks]
            v = np.asarray(X[cells].sum(0)).ravel()
            target = max(500, int(r.normal(VIS_MEAN, VIS_SD)))
            tot = v.sum()
            if tot > target:
                p = target / tot
                v = r.binomial(v.astype(np.int64), p).astype(np.float32)
            rows.append(sp.csr_matrix(v))
            comp = np.zeros(len(keys))
            for j in picks: comp[j] += 1.0
            truth.append(comp / comp.sum())
            names.append(f"{mode}_r{reg}_s{sp_i}")
    Xs = sp.vstack(rows).tocsr()
    syn = ad.AnnData(X=Xs, obs=pd.DataFrame(index=names), var=pd.DataFrame(index=adata.var_names))
    syn.layers["counts"] = syn.X.copy()
    tr = pd.DataFrame(np.vstack(truth), index=names, columns=keys)
    return syn, tr

# ---- 4d. scLVM on the training cells only ----
_mp = f"{ROOT}/models/sclvm_resolvtrain_{TAG}{SUF}"
if os.path.exists(f"{_mp}/model.pt"):
    print(f"reusing cached held-out-patient model at {_mp}", flush=True)
    m_tr = scvi.model.CondSCVI.load(_mp, adata=train)
else:
    scvi.model.CondSCVI.setup_anndata(train, layer="counts", labels_key="cell_type",
                                      fine_labels_key=STATE_COL)
    m_tr = scvi.model.CondSCVI(train, n_latent=10, weight_obs=False)
    fit(m_tr, SC_EPOCHS, batch_size=int(os.environ.get("BATCH","1024")),
        tag=f"{TAG}-train-sclvm")
    m_tr.save(_mp, save_anndata=False, overwrite=True)

for mode in ["overlap", "dominant"]:
    syn, truth = make_synthetic(test, mode)
    print(f"\n[{mode}] {syn.shape} synthetic spots; median UMI {np.median(np.asarray(syn.X.sum(1))):.0f}")
    syn = syn[:, train.var_names].copy(); syn.layers["counts"] = syn.X.copy()
    scvi.model.DestVI.setup_anndata(syn, layer="counts")
    d = scvi.model.DestVI.from_rna_model(syn, prepare_sc_model(m_tr), augmentation=False)
    fit(d, ST_EPOCHS, batch_size=int(os.environ.get("BATCH","1024")), tag=f"{TAG}-{mode}-destvi")
    pred = d.get_fine_celltypes(m_tr)
    pred = pred.div(pred.sum(axis=1), axis=0)
    truth = truth.reindex(index=pred.index).fillna(0.0)
    pred.to_csv(f"{OUT}/resolv_{TAG}{SUF}_{mode}_pred.csv.gz")
    truth.to_csv(f"{OUT}/resolv_{TAG}{SUF}_{mode}_truth.csv.gz")

    cols = [s for s in focus if s in pred.columns and s in truth.columns]
    cross = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for t_ in cols:
        for p_ in cols:
            if truth[t_].std() == 0 or pred[p_].std() == 0: cross.loc[t_, p_] = np.nan
            else: cross.loc[t_, p_] = np.corrcoef(truth[t_], pred[p_])[0, 1]
    cross.to_csv(f"{OUT}/resolv_{TAG}{SUF}_{mode}_crosscorr.csv")
    
# ---- 4f. bookkeeping (scoring is done separately by score.py) ----
json.dump({"tag": TAG, "suffix": SUF, "state_col": STATE_COL, "split_kind": split_kind,
           "test_patients": list(map(str, test_p)), "n_states": len(states), "n_scored_states": len(focus)},
          open(f"{OUT}/resolv_{TAG}{SUF}_meta.json", "w"))
print("RESOLVABILITY DONE", TAG)
