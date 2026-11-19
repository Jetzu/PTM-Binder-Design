"""CPU tests of ptm_rfd3.particles: resampling primitives, weight bookkeeping and the targets each scheme samples.

The end-to-end checks use a Gaussian random walk x_{t+1} = x_t + N(0, s^2) whose 'clean prediction' reward is x_t:
under the base law x_T ~ N(0, T s^2); the FK target q(x) exp(lam x_T) has mean lam T s^2; WE must keep the base law.
"""
import math

import pytest
import torch

from ptm_rfd3.particles import (BurialReward, ParticleEngine, ParticlePolicy, bin_ids, capped, cess, ess, systematic,
                                we_resample)


def test_systematic_counts_are_floor_or_ceil():
    g = torch.Generator().manual_seed(0)
    w = torch.tensor([0.5, 0.25, 0.125, 0.125, 0.0])
    for _ in range(50):
        c = torch.bincount(systematic(w, 8, g), minlength=5)
        assert c[4] == 0
        assert torch.all((c.double() >= torch.floor(8 * w.double())) & (c.double() <= torch.ceil(8 * w.double())))


def test_capped_bounds_copies():
    g = torch.Generator().manual_seed(1)
    w = torch.tensor([0.9, 0.05, 0.03, 0.02, 0, 0, 0, 0], dtype=torch.float64)
    q = capped(w, 2)
    assert abs(float(q.sum()) - 1) < 1e-12 and float(q.max()) <= 2 / 8 + 1e-12
    for _ in range(50):
        assert int(torch.bincount(systematic(q, 8, g)).max()) <= 2


def test_ess_and_cess():
    assert abs(ess(torch.zeros(10)) - 10) < 1e-9
    assert abs(ess(torch.tensor([0.0, -1e9, -1e9])) - 1) < 1e-6
    assert abs(cess(torch.zeros(5), torch.zeros(5)) - 5) < 1e-9


def test_we_resample_conserves_bin_weights():
    g = torch.Generator().manual_seed(2)
    for _ in range(20):
        lw = torch.randn(12, dtype=torch.float64)
        r = torch.randn(12)
        b = bin_ids(r, 4)
        par, nlw = we_resample(lw, b, g)
        W, Wn = torch.softmax(lw, 0), torch.exp(nlw)
        assert par.numel() == 12
        assert abs(float(Wn.sum()) - 1) < 1e-9
        for k in set(b.tolist()):
            assert abs(float(W[b == k].sum()) - float(Wn[b[par] == k].sum())) < 1e-9


def test_we_resample_unbiased_for_observables():
    g = torch.Generator().manual_seed(3)
    lw = torch.randn(10, dtype=torch.float64)
    f = torch.randn(10, dtype=torch.float64)
    b = bin_ids(f.float(), 3)
    exact = float((torch.softmax(lw, 0) * f).sum())
    est = []
    for _ in range(4000):
        par, nlw = we_resample(lw, b, g)
        est.append(float((torch.exp(nlw) * f[par]).sum()))
    est = torch.tensor(est)
    assert abs(float(est.mean()) - exact) < 4 * float(est.std()) / math.sqrt(len(est)) + 1e-9


def test_smc_weights_telescope_without_resampling():
    pol = ParticlePolicy(scheme="smc", n_particles=6, start=0.0, stop=1.0, ess_frac=0.0, lam_max=3.0)
    eng = ParticleEngine(pol)
    for i in range(10):
        assert eng.score(torch.randn(6), progress=(i + 1) / 10) is None
    r_final = torch.randn(6, dtype=torch.float64)
    out = eng.finish(r_final)
    expect = torch.softmax(3.0 * r_final, 0)
    assert torch.allclose(out["weights"], expect, atol=1e-9)


def _walk(policy, T=20, s=0.3, seed=0):
    torch.manual_seed(seed)
    K = policy.n_particles
    eng = ParticleEngine(policy)
    x = torch.zeros(K, dtype=torch.float64)
    for t in range(T):
        x = x + s * torch.randn(K, dtype=torch.float64)
        par = eng.score(x, progress=(t + 1) / T, step=t)
        if par is not None:
            x = x[par]
    out = eng.finish(x)
    return x, out


@pytest.mark.parametrize("scheme", ["smc", "twe"])
def test_tilted_schemes_sample_fk_target(scheme):
    lam, T, s = 1.0, 20, 0.3
    pol = ParticlePolicy(scheme=scheme, n_particles=4000, start=0.0, stop=1.0, lam_max=lam, ess_frac=0.5, we_bins=8,
                         we_every=2, seed=5)
    x, out = _walk(pol, T, s, seed=5)
    mean = float((out["weights"] * x).sum())
    assert abs(mean - lam * T * s * s) < 0.12, mean           # target mean 1.8, base mean 0


def test_we_keeps_base_law_and_enriches_tail():
    T, s = 20, 0.3
    sd = math.sqrt(T) * s
    pol = ParticlePolicy(scheme="we", n_particles=4000, start=0.0, stop=1.0, we_bins=8, we_every=1, seed=7)
    x, out = _walk(pol, T, s, seed=7)
    w = out["weights"]
    assert abs(float((w * x).sum())) < 0.1                                    # base mean 0
    p_tail = float(w[x > 2 * sd].sum())
    assert abs(p_tail - 0.02275) < 0.012, p_tail                              # P(Z > 2) under the base law
    assert float((x > 2 * sd).double().mean()) > 2 * 0.02275                  # but more WALKERS sit in the tail


def test_islands_never_exchange_and_none_is_inert():
    pol = ParticlePolicy(scheme="smc", n_particles=8, islands=2, start=0.0, stop=1.0, ess_frac=1.0, lam_max=5)
    eng = ParticleEngine(pol)
    for i in range(5):
        par = eng.score(torch.randn(8), progress=(i + 1) / 5)
        if par is not None:
            assert bool((par[:4] < 4).all()) and bool((par[4:] >= 4).all())
    eng0 = ParticleEngine(ParticlePolicy(scheme="none", n_particles=4))
    assert eng0.score(torch.randn(4), 0.6) is None
    assert torch.allclose(eng0.finish(torch.randn(4))["weights"], torch.full((4,), 0.25, dtype=torch.float64))


def test_window_and_cess_tempering():
    pol = ParticlePolicy(scheme="smc", n_particles=16, start=0.5, stop=0.8, cess_frac=0.9, lam_max=50, ess_frac=0.0)
    eng = ParticleEngine(pol)
    assert eng.score(torch.randn(16), 0.2) is None and eng.lam == 0.0           # outside the window: no tilt
    eng.score(torch.randn(16) * 3, 0.8)
    assert 0 < eng.lam < 50                                                    # adaptive tempering held lam back


def test_policy_from_env():
    p = ParticlePolicy.from_env({"PTMSMC_SCHEME": "twe", "PTMSMC_N_PARTICLES": "16", "PTMSMC_ISLANDS": "2",
                                 "PTMSMC_LAM_MAX": "4.5", "PTMSMC_TERMINAL_RESAMPLE": "true"})
    assert (p.scheme, p.n_particles, p.islands, p.lam_max, p.terminal_resample) == ("twe", 16, 2, 4.5, True)


def test_burial_reward_prefers_buried_ptm_and_is_invariant():
    torch.manual_seed(0)
    ptm_xyz = torch.zeros(1, 3)
    other_xyz = torch.tensor([[12.0, 0, 0], [15.0, 0, 0]])
    shell = torch.randn(30, 3)
    shell = 5.0 * shell / shell.norm(dim=-1, keepdim=True)                      # binder atoms 5 A around the PTM
    far = shell + torch.tensor([30.0, 0, 0])
    X = torch.stack([torch.cat([ptm_xyz, other_xyz, shell]), torch.cat([ptm_xyz, other_xyz, far])])
    ptm = torch.zeros(33, dtype=torch.bool); ptm[0] = True
    oth = torch.zeros(33, dtype=torch.bool); oth[1:3] = True
    bnd = torch.zeros(33, dtype=torch.bool); bnd[3:] = True
    br = BurialReward()
    r = br(X, ptm, oth, bnd)
    assert r[0] > 0.9 and r[1] < 0.1
    Rm = torch.linalg.qr(torch.randn(3, 3))[0]
    assert torch.allclose(br(X @ Rm.T + 3.0, ptm, oth, bnd), r, atol=1e-4)


def test_solid_burial_bounded_and_ordered():
    torch.manual_seed(1)
    ptm = torch.zeros(1, 3)
    shell = torch.randn(200, 3); shell = 5.0 * shell / shell.norm(dim=-1, keepdim=True)
    half = shell[shell[:, 0] > 0]
    X = [torch.cat([ptm, s]) for s in (shell, half, shell + 40.0)]
    n = max(len(x) for x in X)
    X = torch.stack([torch.cat([x, torch.full((n - len(x), 3), 500.0)]) for x in X])
    pm = torch.zeros(n, dtype=torch.bool); pm[0] = True
    om = torch.zeros(n, dtype=torch.bool)
    bm = ~pm
    br = BurialReward(mode="solid", clash_w=0.0)
    t = br.terms(X, pm, om, bm)
    b = t["bur_ptm"]
    assert b[0] > 0.9 and 0.3 < b[1] < 0.7 and b[2] == 0.0
    assert torch.allclose(t["reward"], b)


def test_fold_clusters_alignment_free():
    from ptm_rfd3.particles import fold_clusters
    torch.manual_seed(0)
    a = torch.cumsum(torch.randn(30, 3), 0) * 2
    b = torch.cumsum(torch.randn(30, 3), 0) * 2
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    anc = torch.zeros(2, 3)
    ca = torch.stack([a, a @ R.T + 5, b, a + 0.3 * torch.randn(30, 3)])
    an = torch.stack([anc, anc @ R.T + 5, anc, anc])
    cl = fold_clusters(ca, an, 3.0)
    assert cl[0] == cl[1] == cl[3] and cl[2] != cl[0]


def test_we_fold_bins_keep_every_fold():
    pol = ParticlePolicy(scheme="we", n_particles=8, start=0.0, stop=1.0, we_bin_by="fold", we_every=1)
    eng = ParticleEngine(pol)
    clusters = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2])
    par = eng.score(torch.randn(8), 0.5, clusters=clusters)
    counts = torch.bincount(clusters[par], minlength=3)
    assert counts.tolist() in ([2, 3, 3], [3, 3, 2], [3, 2, 3]) and abs(float(torch.softmax(eng.log_w, 0).sum()) - 1) < 1e-9


def test_smc_fold_dedup_spreads_offspring():
    pol = ParticlePolicy(scheme="smc", n_particles=9, start=0.0, stop=1.0, ess_frac=1.0, lam_max=0.0, fold_dedup=True)
    eng = ParticleEngine(pol)
    eng.log_w = torch.zeros(9, dtype=torch.float64)
    eng.log_w[0] = -0.5                                    # ESS below K (8.9 of 9): forces the resampling branch
    clusters = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 2])
    par = eng.score(torch.zeros(9), 0.5, clusters=clusters)
    counts = torch.bincount(clusters[par], minlength=3).tolist()
    assert all(2 <= c <= 4 for c in counts), counts      # ~3 each (systematic: floor/ceil per particle, +-1 per cluster)
    plain = ParticleEngine(ParticlePolicy(scheme="smc", n_particles=9, start=0.0, stop=1.0, ess_frac=1.0, lam_max=0.0))
    plain.log_w = eng.log_w.clone() * 0
    plain.log_w[0] = -0.5
    pc = torch.bincount(clusters[plain.score(torch.zeros(9), 0.5, clusters=clusters)], minlength=3).tolist()
    assert pc[0] >= 6, pc                                  # without dedup the big fold keeps ~7 of 9 slots


def test_fork_pairs_once():
    eng = ParticleEngine(ParticlePolicy(scheme="fork", n_particles=6, start=0.3, stop=1.0))
    assert eng.score(torch.randn(6), 0.2) is None
    par = eng.score(torch.randn(6), 0.4)
    assert par.tolist() == [0, 0, 2, 2, 4, 4] and eng.score(torch.randn(6), 0.5) is None


@pytest.mark.parametrize("dedup,mix", [(False, 0.0), (True, 1.0), (True, 0.5)])
def test_smc_fold_dedup_is_importance_corrected(dedup, mix):
    """Random walk with a 'fold' label fixed at the first step (two folds, 85/15 under the base law). With dedup the
    survivors are spread over both folds, yet the weighted estimate of the fold frequency and of the tilted mean must
    match plain SMC / the analytic target."""
    lam, T, s, K = 1.0, 20, 0.3, 4000
    torch.manual_seed(11)
    pol = ParticlePolicy(scheme="smc", n_particles=K, start=0.0, stop=1.0, lam_max=lam, ess_frac=0.5, seed=11,
                         fold_dedup=dedup, fold_mix=mix)
    eng = ParticleEngine(pol)
    x = torch.zeros(K, dtype=torch.float64)
    fold = (torch.rand(K) < 0.15).long()
    for t in range(T):
        x = x + s * torch.randn(K, dtype=torch.float64)
        par = eng.score(x, (t + 1) / T, step=t, clusters=fold)
        if par is not None:
            x, fold = x[par], fold[par]
    w = eng.finish(x)["weights"]
    assert abs(float((w * x).sum()) - lam * T * s * s) < 0.15
    assert abs(float(w[fold == 1].sum()) - 0.15) < 0.04
    if dedup:                                               # survivors spread: the minority fold holds many more slots
        assert float((fold == 1).double().mean()) > (0.4 if mix == 1.0 else 0.25)
