"""
OW-ERF core pipeline (TABULAR PROTOTYPE, no graph encoder).

What this script does (all on SYNTHETIC data you control):
  1. Generates benign claims (incl. rare-severe legitimate ones) and 4 fraud mechanisms.
  2. Leave-one-mechanism-out: trains on benign + 3 known mechanisms, tests on all 4.
  3. Novelty score = fusion of Energy (classifier logits) and Mahalanobis distance
     via empirical CDFs.
  4. Split-conformal p-value on a disjoint calibration set; reject if p <= alpha.
  5. Explained-Rarity Filter: heteroscedastic expectation model trained on BENIGN only;
     escalate only if the conformal-rejected claim is also "unexplained".
  6. Compares against an IsolationForest at a matched alert rate.

LIMITATIONS (state these in the paper):
  - Synthetic data only; results depend on the generator.
  - No graph encoder / temporal edges here: mechanism D (collusion) is only
    represented via two tabular proxy features.
  - No episodic pseudo-novel training; energy/Mahalanobis are post-hoc.
  - Heteroscedastic model is a two-stage approximation (mean model + variance
    model on cross-fitted residuals), not a Gaussian-NLL neural net.
Run: python owerf_core.py
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(0)
FEATS = ["age", "severity", "comorb", "log_amount", "los", "items", "ref_conc", "claim_rate"]
MECH = {1: "upcoding", 2: "unbundling", 3: "phantom", 4: "collusion"}


def make_claims(n, rng, fraud_rate=0.10):
    """Rows: features + label (0 benign, 1..4 fraud mechanism) + rare flag."""
    age = rng.integers(18, 85, n)
    # severity is an INDEPENDENT proxy (e.g. from pre-claim history), not the claimed code
    severity = np.clip(rng.beta(1.5, 5, n), 0, 1)
    rare = rng.random(n) < 0.05                       # rare-severe LEGITIMATE claims
    severity[rare] = rng.uniform(0.85, 1.0, rare.sum())
    comorb = rng.poisson(1 + 3 * severity)
    sd_scale = 0.15 + 0.5 * severity                  # heteroscedastic noise
    log_amount = 10 + 1.8 * severity + 0.08 * comorb + rng.normal(0, sd_scale)
    los = np.maximum(1, 2 + 10 * severity + 0.3 * comorb + rng.normal(0, 0.5 + 2 * severity))
    items = np.maximum(1, rng.poisson(2 + 4 * severity))
    ref_conc = rng.beta(2, 8, n)
    claim_rate = rng.gamma(2, 1, n)
    label = np.zeros(n, int)

    fraud = rng.random(n) < fraud_rate
    label[fraud] = rng.integers(1, 5, fraud.sum())
    m1, m2, m3, m4 = (label == k for k in (1, 2, 3, 4))
    log_amount[m1] += rng.normal(1.0, 0.25, m1.sum())              # upcoding: amount up
    log_amount[m2] += rng.normal(0.6, 0.25, m2.sum())              # unbundling: amount+items up
    items[m2] += rng.integers(3, 8, m2.sum())
    los[m3] = rng.uniform(0.5, 1.5, m3.sum())                      # phantom: tiny LOS, high amount
    log_amount[m3] += rng.normal(0.9, 0.25, m3.sum())
    ref_conc[m4] = rng.beta(8, 2, m4.sum())                        # collusion: proxy features only
    claim_rate[m4] = rng.gamma(8, 1, m4.sum())
    log_amount[m4] += rng.normal(0.25, 0.2, m4.sum())

    X = np.column_stack([age, severity, comorb, log_amount, los, items, ref_conc, claim_rate])
    return X, label, rare


def energy(logits, T=1.0):
    return -T * logsumexp(logits / T, axis=1)


def maha(Z, mu, prec):
    d = Z - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)


def ecdf(ref):
    ref = np.sort(ref)
    return lambda x: np.searchsorted(ref, x, side="right") / len(ref)


class OWERF:
    def __init__(self, alpha=0.05, tau_q=0.95):
        self.alpha, self.tau_q = alpha, tau_q

    def fit(self, Xtr, ytr, Xcal, ycal):
        self.sc = StandardScaler().fit(Xtr)
        Ztr, Zcal = self.sc.transform(Xtr), self.sc.transform(Xcal)
        # --- closed-set classifier over known classes (benign + known mechanisms)
        self.clf = LogisticRegression(max_iter=2000).fit(Ztr, ytr)
        # --- class-conditional Gaussians for Mahalanobis (min over classes)
        self.cls_stats = []
        for c in self.clf.classes_:
            Zc = Ztr[ytr == c]
            mu = Zc.mean(0)
            cov = np.cov(Zc.T) + 1e-3 * np.eye(Zc.shape[1])
            self.cls_stats.append((mu, np.linalg.inv(cov)))
        e_tr, m_tr = self._raw(Ztr)
        self.F_e, self.F_m = ecdf(e_tr), ecdf(m_tr)
        # --- conformal calibration on DISJOINT inlier window
        self.cal_scores = np.sort(self._fused(Zcal))
        # --- Explained-Rarity Filter: trained on BENIGN only
        ben = ytr == 0
        Xb = Xtr[ben]
        inp = [1, 2, 0]                         # severity, comorb, age (exogenous inputs)
        self.inp = inp
        self.targets = [3, 4]                   # log_amount, los
        self.mean_m, self.var_m = {}, {}
        for j in self.targets:
            mm = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0)
            oof = cross_val_predict(mm, Xb[:, inp], Xb[:, j], cv=5)
            res2 = (Xb[:, j] - oof) ** 2
            self.mean_m[j] = mm.fit(Xb[:, inp], Xb[:, j])
            vm = GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=0)
            self.var_m[j] = vm.fit(Xb[:, inp], np.log(res2 + 1e-6))
        # tau calibrated on benign calibration claims (not hand-set)
        a_cal = self.unexplained(Xcal[ycal == 0])
        self.tau = np.quantile(a_cal, self.tau_q)
        return self

    def _raw(self, Z):
        e = energy(self.clf.decision_function(Z))
        m = np.min([maha(Z, mu, p) for mu, p in self.cls_stats], axis=0)
        return e, m

    def _fused(self, Z):
        e, m = self._raw(Z)
        return 0.5 * self.F_e(e) + 0.5 * self.F_m(m)

    def pvalues(self, X):
        s = self._fused(self.sc.transform(X))
        n = len(self.cal_scores)
        ge = n - np.searchsorted(self.cal_scores, s, side="left")
        return (1 + ge) / (n + 1)

    def unexplained(self, X):
        zs = []
        for j in self.targets:
            mu = self.mean_m[j].predict(X[:, self.inp])
            sd = np.sqrt(np.exp(self.var_m[j].predict(X[:, self.inp])))
            zs.append(np.maximum(0, (X[:, j] - mu) / sd))
        return np.max(zs, axis=0)

    def decide(self, X):
        rej = self.pvalues(X) <= self.alpha
        esc = rej & (self.unexplained(X) > self.tau)
        return rej, esc


def run(held_out, n=30000):
    X, y, rare = make_claims(n, RNG)
    known = [0] + [k for k in MECH if k != held_out]
    idx = RNG.permutation(n)
    tr, cal, te = idx[: int(.5 * n)], idx[int(.5 * n): int(.7 * n)], idx[int(.7 * n):]
    trk = tr[np.isin(y[tr], known)]
    calk = cal[np.isin(y[cal], known)]
    model = OWERF().fit(X[trk], y[trk], X[calk], y[calk])

    Xt, yt, rt = X[te], y[te], rare[te]
    rej, esc = model.decide(Xt)
    nov = yt == held_out
    ben_rare = (yt == 0) & rt
    ben_all = yt == 0

    # matched-alert-rate IsolationForest baseline (fit on known classes only)
    iso = IsolationForest(random_state=0).fit(model.sc.transform(X[trk]))
    sc_iso = -iso.score_samples(model.sc.transform(Xt))
    k = int(esc.sum())
    iso_flag = np.zeros(len(Xt), bool)
    if k > 0:
        iso_flag[np.argsort(-sc_iso)[:k]] = True

    return dict(
        held_out=MECH[held_out],
        recall_novel_conformal=rej[nov].mean(),
        recall_novel_erf=esc[nov].mean(),
        fpr_benign_conformal=rej[ben_all].mean(),
        fpr_benign_erf=esc[ben_all].mean(),
        fpr_rare_conformal=rej[ben_rare].mean(),
        fpr_rare_erf=esc[ben_rare].mean(),
        recall_novel_iforest_matched=iso_flag[nov].mean(),
        fpr_rare_iforest_matched=iso_flag[ben_rare].mean(),
    )


if __name__ == "__main__":
    rows = []
    for seed in range(5):
        RNG = np.random.default_rng(seed)
        for h in MECH:
            r = run(h)
            r["seed"] = seed
            rows.append(r)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["held_out"]].append(r)
    keys = [k for k in rows[0] if k not in ("held_out", "seed")]
    print(f"{'held-out':<11}" + "".join(f"{k[:17]:>19}" for k in keys))
    for h, rs in agg.items():
        print(f"{h:<11}" + "".join(
            f"{np.mean([r[k] for r in rs]):>13.3f}±{np.std([r[k] for r in rs]):.2f}" for k in keys))
