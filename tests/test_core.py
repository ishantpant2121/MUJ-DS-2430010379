import numpy as np
import owerf_core as oc


def _fit_model(held_out=4, n=12000, seed=0):
    rng = np.random.default_rng(seed)
    X, y, rare = oc.make_claims(n, rng)
    known = [0] + [k for k in oc.MECH if k != held_out]
    idx = rng.permutation(n)
    tr, cal, te = idx[: int(.5 * n)], idx[int(.5 * n): int(.7 * n)], idx[int(.7 * n):]
    trk = tr[np.isin(y[tr], known)]
    calk = cal[np.isin(y[cal], known)]
    model = oc.OWERF().fit(X[trk], y[trk], X[calk], y[calk])
    return model, X[te], y[te], rare[te], known


def test_generator_shapes_labels_and_rates():
    X, y, rare = oc.make_claims(5000, np.random.default_rng(1))
    assert X.shape == (5000, 8)
    assert set(np.unique(y)) <= {0, 1, 2, 3, 4}
    assert 0.07 < (y > 0).mean() < 0.13          # about 10% fraud
    assert 0.03 < rare.mean() < 0.07             # about 5% rare-severe


def test_generator_is_reproducible():
    a = oc.make_claims(1000, np.random.default_rng(3))
    b = oc.make_claims(1000, np.random.default_rng(3))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_energy_matches_manual_formula():
    logits = np.array([[1.0, 2.0, 3.0]])
    expected = -np.log(np.exp(1) + np.exp(2) + np.exp(3))
    assert np.isclose(oc.energy(logits)[0], expected)


def test_ecdf_is_monotone_and_bounded():
    f = oc.ecdf(np.random.default_rng(0).normal(size=1000))
    v = f(np.linspace(-4, 4, 50))
    assert np.all(np.diff(v) >= 0)
    assert v.min() >= 0 and v.max() <= 1


def test_pvalues_valid_and_conformal_rate_near_alpha():
    model, Xt, yt, rt, known = _fit_model()
    p = model.pvalues(Xt)
    assert np.all((p > 0) & (p <= 1))
    inliers = np.isin(yt, known)
    rate = (p[inliers] <= 0.05).mean()
    assert 0.02 < rate < 0.09                    # near the nominal 5%


def test_escalation_is_subset_of_rejection_and_filter_helps():
    model, Xt, yt, rt, known = _fit_model()
    rej, esc = model.decide(Xt)
    assert np.all(esc <= rej)                    # escalated implies rejected
    benign = yt == 0
    assert esc[benign].mean() < rej[benign].mean()


def test_inconsistency_ratio_nonnegative_and_tau_finite():
    model, Xt, yt, rt, known = _fit_model()
    a = model.unexplained(Xt)
    assert np.all(a >= 0)
    assert np.isfinite(model.tau)
