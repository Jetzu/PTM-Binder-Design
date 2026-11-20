"""Particle guidance shared by the RFD3 and RFD4 samplers: ESS-triggered SMC, weighted ensemble (WE) and tilted WE.

The generator is not touched: a sampler runs K particles as one batch, calls `ParticleEngine.score(reward, progress)` at
each step with the per-particle reward of the CLEAN prediction (x0 for RFD3, x1 for RFD4) and, when the engine returns
parent indices, reindexes its state along the batch axis. `finish(reward)` applies the terminal weight and returns the
per-particle output weights. Pure torch, CPU-side, no model imports (unit-testable).

Schemes (PN001 literature review, .agent-research/results/N001-*):
  smc  Feynman-Kac SMC with the tempered DIFFERENCE potential G = exp(lam_t r_t - lam_prev r_prev); the product telescopes,
       so the target is q(x) exp(lam_max r(x)) (Singhal et al. arXiv 2501.06848; tempering as in DAS, arXiv 2501.05803).
       lam_t follows lam_max * window_progress**lam_power, optionally lowered so the conditional ESS (Zhou, Johansen &
       Aston arXiv 1303.3123) of the increment stays >= cess_frac * K. Resampling fires when ESS < ess_frac * K
       (systematic; optional cap on offspring per parent, Douc et al. cs/0507025).
  we   weighted ensemble (Huber & Kim 1996; unbiased for any binning, Aristoff arXiv 1609.05887): NO tilt. Particles are
       binned on the reward (equal-width bins between the current min and max); every `we_every`-th scored step each occupied bin receives an equal share of the K slots and
       is resampled within itself proportionally to weight, each child carrying bin weight / bin count. Total weight is
       conserved exactly, so the weights stay those of the base sampler; the allocation spreads particles along the
       reward axis instead of tilting toward its top.
  twe  tilted WE: the smc weights (tempered difference potential) with the we allocation (bins decide copy counts).
  none / null: `none` only records rewards; `null=True` runs the chosen scheme on a randomly permuted reward each step
       (control: separates any gain of resampling itself from the reward signal).

Islands: K particles form `islands` contiguous groups that never exchange particles (each an independent sampler).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch

SCHEMES = ("none", "smc", "we", "twe", "fork")


@dataclass
class ParticlePolicy:
    scheme: str = "none"
    n_particles: int = 8
    islands: int = 1
    seed: int = 0
    # scoring window on the sampler's progress clock (0 = start, 1 = last step); resampling only inside it
    start: float = 0.55
    stop: float = 0.78
    every: int = 1                # score every n-th step inside the window
    # tilt (smc / twe)
    lam_max: float = 8.0
    lam_power: float = 1.0
    cess_frac: float = 0.0        # > 0: adaptive tempering, lam never raises the conditional ESS above this fraction
    ess_frac: float = 0.5         # smc: resample when ESS < ess_frac * particles per island
    max_offspring: int = 0        # smc: > 0 caps copies of any parent (water-filled weights)
    # weighted ensemble
    we_bins: int = 4
    we_every: int = 2             # WE resampling every n-th scored step
    # output
    terminal_resample: bool = False  # False: every finished particle returned once with its weight
    null: bool = False
    # anti-collapse (N002: late resampling copies a committed fold - WE siblings TM 0.92, 8 -> 2 fold clusters).
    # Particles are clustered at every resampling event by the predicted binder geometry (fold_clusters below).
    we_bin_by: str = "reward"     # WE bins: 'reward' (equal-width on the reward) or 'fold' (one bin per fold cluster:
                                  # every fold present keeps its share of slots; tilt still selects within a fold)
    fold_dedup: bool = False      # smc: offspring drawn from q = (1 - fold_mix) W + fold_mix * W_i / (C * M_fold(i)):
                                  # every fold cluster gets an equal share of the slots at fold_mix = 1 (WE-like allocation)
    fold_mix: float = 0.5         # 0 = plain W (no diversity bias) .. 1 = equal mass per fold cluster
    fold_correct: bool = True     # smc: each child carries log W_parent - log q_parent, so the weighted population still
                                  # targets q_RFD3 * exp(lam r) (importance-corrected); False = reset weights (biased, old)
    fold_thr: float = 2.0         # distance-matrix RMSD (A) below which two particles are the same fold + pose (N002 calibration: resampled siblings 0.3-1.3 A, distinct designs >= 2.7 A)
    chunk: int = 0                # > 0: the generator's denoiser is called on chunks of this many particles (memory); the
                                  # population (resampling) still spans all n_particles

    def __post_init__(self):
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        if self.n_particles < 1 or self.islands < 1 or self.n_particles % self.islands:
            raise ValueError("islands must divide n_particles")
        if not 0.0 <= self.start <= self.stop <= 1.0:
            raise ValueError("need 0 <= start <= stop <= 1")
        if not 0.0 <= self.fold_mix <= 1.0:
            raise ValueError("fold_mix must be in [0, 1]")
        if self.we_bin_by not in ("reward", "fold"):
            raise ValueError("we_bin_by must be reward or fold")
        if self.every < 1 or self.we_every < 1 or self.we_bins < 1:
            raise ValueError("every, we_every, we_bins must be >= 1")
        if not 0.0 <= self.ess_frac <= 1.0 or not 0.0 <= self.cess_frac <= 1.0:
            raise ValueError("ess_frac and cess_frac must be in [0, 1]")
        if self.max_offspring and not 1 <= self.max_offspring <= self.n_particles // self.islands:
            raise ValueError("max_offspring must be 0 or in [1, particles per island]")
        if self.chunk < 0:
            raise ValueError("chunk must be >= 0")
        if self.lam_max < 0 or self.lam_power <= 0:
            raise ValueError("lam_max >= 0 and lam_power > 0 required")

    @property
    def tilted(self) -> bool:
        return self.scheme in ("smc", "twe")

    @classmethod
    def from_env(cls, env: dict, prefix: str = "PTMSMC_") -> "ParticlePolicy":
        """Policy from environment variables PTMSMC_<FIELD> (e.g. PTMSMC_SCHEME=smc, PTMSMC_N_PARTICLES=16)."""
        kw = {}
        for f in cls.__dataclass_fields__.values():
            v = env.get(prefix + f.name.upper())
            if v is None or v == "":
                continue
            kw[f.name] = (v.lower() in ("1", "true", "yes")) if f.type in ("bool", bool) else type(f.default)(v)
        return cls(**kw)


# ----------------------------------------------------------------------------------------------- resampling primitives
def systematic(weights: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """n parent indices from normalised CPU weights by one systematic draw; a position on a cumulative boundary goes to
    the next particle, so zero-weight particles are never drawn and parent i gets floor or ceil of n w_i copies."""
    w = weights.double()
    if w.numel() == 0 or not bool(torch.isfinite(w).all()) or bool((w < 0).any()) or float(w.sum()) <= 0:
        raise ValueError("systematic needs finite non-negative weights with a positive sum")
    c = torch.cumsum(w / w.sum(), 0)
    c[-1] = 1.0
    pos = (torch.rand((), generator=gen, dtype=torch.float64) + torch.arange(n, dtype=torch.float64)) / n
    return torch.searchsorted(c, pos, right=True).clamp(max=w.numel() - 1)


def capped(weights: torch.Tensor, cap: int) -> torch.Tensor:
    """Water-filling of normalised weights at cap / K: systematic resampling of the result gives each parent <= cap copies
    in expectation-consistent proportions; offspring must carry w / q to stay unbiased."""
    K = weights.numel()
    w = weights.double() / weights.double().sum()
    m = cap / K
    q = w.clone()
    for _ in range(K):
        over = q > m + 1e-15
        if not bool(over.any()):
            break
        excess = float((q[over] - m).sum())
        q[over] = m
        free = q < m - 1e-15
        if not bool(free.any()) or float(q[free].sum()) <= 0:
            break
        q[free] = q[free] * (1.0 + excess / float(q[free].sum()))
    if float(q.sum()) < 1 - 1e-12:  # remaining mass onto zero-weight particles (all others at the cap)
        z = q < m
        q[z] += (1.0 - float(q.sum())) / int(z.sum())
    return q / q.sum()


def ess(log_w: torch.Tensor) -> float:
    w = torch.softmax(log_w.double(), 0)
    return float(1.0 / (w * w).sum())


def cess(log_w: torch.Tensor, log_g: torch.Tensor) -> float:
    """Conditional ESS of an incremental weight g given current weights (Zhou, Johansen & Aston 2016), in particles."""
    W = torch.softmax(log_w.double(), 0)
    lg = log_g.double() - log_g.double().max()
    g = torch.exp(lg)
    return float(W.numel() * (W * g).sum() ** 2 / (W * g * g).sum())


def bin_ids(r: torch.Tensor, nbins: int) -> torch.Tensor:
    """Bin index per particle on the reward axis (higher bin = higher reward): equal-width bins between the current
    minimum and maximum (WESTPA's minimal adaptive binning). Quantile bins would be useless here: they always hold equal
    counts, so an even allocation over occupied bins would never move a walker toward the sparse high-reward end
    (caught by tests/test_particles.py::test_we_keeps_base_law_and_enriches_tail)."""
    r = r.double()
    if nbins == 1 or float(r.max() - r.min()) <= 1e-12:
        return torch.zeros(r.numel(), dtype=torch.long)
    edges = torch.linspace(float(r.min()), float(r.max()), nbins + 1, dtype=torch.float64)[1:-1]
    return torch.bucketize(r, edges, right=True)


def we_resample(log_w: torch.Tensor, bins: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted-ensemble resampling of one island: K slots split evenly over the occupied bins (remainder to the highest
    bins), within each bin systematic draws proportional to weight, each child weight = bin weight / bin slots.
    Returns (parents, new log weights); the total weight of every bin, hence of the island, is unchanged."""
    K = log_w.numel()
    W = torch.softmax(log_w.double(), 0)
    occ = sorted(set(bins.tolist()))
    base, extra = divmod(K, len(occ))
    slots = {b: base + (1 if i >= len(occ) - extra else 0) for i, b in enumerate(occ)}
    parents, new_w = [], []
    for b in occ:
        idx = (bins == b).nonzero().flatten()
        wb = W[idx]
        tot = float(wb.sum())
        if tot <= 0:  # whole bin underflowed: keep its members uniformly (it carries no weight anyway)
            pick = idx[torch.arange(slots[b]) % idx.numel()]
            parents.append(pick)
            new_w.append(torch.full((slots[b],), 1e-300, dtype=torch.float64))
            continue
        parents.append(idx[systematic(wb, slots[b], gen)])
        new_w.append(torch.full((slots[b],), tot / slots[b], dtype=torch.float64))
    return torch.cat(parents), torch.log(torch.cat(new_w))


def fold_clusters(ca: torch.Tensor, anchor: torch.Tensor | None, thr: float) -> torch.Tensor:
    """Greedy clusters of particles by predicted binder geometry. ca [K,n,3] binder CA (same residues in every particle),
    anchor [K,m,3] target atoms (e.g. the PTM atoms) or None. Distance between particles k, l = RMS difference of their
    internal distance matrices (binder-binder + binder-anchor): alignment-free, captures fold AND binding pose.
    Returns [K] long cluster ids (0..C-1, cluster 0 holds particle 0)."""
    pts = ca if anchor is None else torch.cat([ca, anchor], 1)
    D = torch.cdist(pts.float(), pts.float())                                   # [K, N, N]
    n = ca.shape[1]
    D = D[:, :n].flatten(1)                                                     # binder rows: binder-binder + binder-anchor
    d = torch.cdist(D, D) / (D.shape[1] ** 0.5)                                 # [K, K] RMS difference
    K = ca.shape[0]
    cid = torch.full((K,), -1, dtype=torch.long)
    c = 0
    for i in range(K):
        if cid[i] >= 0:
            continue
        members = ((d[i].cpu() < thr) & (cid < 0))
        cid[members] = c
        cid[i] = c
        c += 1
    return cid


# ------------------------------------------------------------------------------------------------------------- engine
@dataclass
class ParticleEngine:
    policy: ParticlePolicy
    log_w: torch.Tensor = field(init=False)
    pot: torch.Tensor = field(init=False)       # lam_prev * r_prev per particle (tilted schemes)
    lam: float = field(init=False, default=0.0)
    n_scored: int = field(init=False, default=0)
    log: list = field(init=False, default_factory=list)

    def __post_init__(self):
        K = self.policy.n_particles
        self.log_w = torch.zeros(K, dtype=torch.float64)
        self.pot = torch.zeros(K, dtype=torch.float64)
        self.gen = torch.Generator().manual_seed(self.policy.seed)
        self.ancestry = torch.arange(K)  # original particle (lineage root) of every current slot

    # -- helpers
    def _slices(self):
        g = self.policy.n_particles // self.policy.islands
        return [slice(i * g, (i + 1) * g) for i in range(self.policy.islands)]

    def window(self, progress: float) -> bool:
        """Inside the scoring window. Scheme 'none' scores (records) there too but never reweights or resamples."""
        p = self.policy
        return p.start <= progress <= p.stop

    def wants_score(self, step_in_window: int) -> bool:
        return step_in_window % self.policy.every == 0

    def _lam_sched(self, progress: float) -> float:
        p = self.policy
        span = max(p.stop - p.start, 1e-12)
        frac = min(max((progress - p.start) / span, 0.0), 1.0)
        return p.lam_max * frac ** p.lam_power

    def _choose_lam(self, r: torch.Tensor, target: float) -> float:
        """Largest lam in [self.lam, target] whose increment keeps CESS >= cess_frac in every island (grid + bisection;
        CESS is not monotone in lam in general)."""
        p = self.policy
        if not p.cess_frac or target <= self.lam:
            return max(target, self.lam)

        def ok(lam):
            inc = lam * r - self.pot
            return all(cess(self.log_w[s], inc[s]) >= p.cess_frac * (s.stop - s.start) for s in self._slices())

        if ok(target):
            return target
        grid = torch.linspace(self.lam, target, 33, dtype=torch.float64).tolist()
        feas = [i for i, v in enumerate(grid) if ok(v)]
        if not feas:
            return self.lam
        lo, hi = grid[feas[-1]], grid[feas[-1] + 1]
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo

    def _tilt(self, r: torch.Tensor, lam: float):
        new_pot = lam * r
        self.log_w = self.log_w + (new_pot - self.pot)
        self.pot = new_pot
        self.lam = lam

    def _apply(self, parents: torch.Tensor, new_log_w: torch.Tensor):
        self.pot = self.pot[parents]
        self.log_w = new_log_w
        self.ancestry = self.ancestry[parents]

    def _smc_resample(self, clusters: torch.Tensor | None = None):
        """ESS-triggered systematic resampling per island; returns parents or None.
        Offspring are drawn from a proposal q (W, optionally divided by fold-cluster size [fold_dedup] and water-filled at
        max_offspring); whenever q != W each child carries log W_parent - log q_parent (fold_correct, default), which keeps
        the weighted population an unbiased estimate of the tilted target (resampling with an arbitrary proposal +
        importance correction). The diversity bias then lives only in WHICH particles survive, not in the target."""
        p = self.policy
        if not any(ess(self.log_w[s]) < p.ess_frac * (s.stop - s.start) for s in self._slices()):
            return None
        parents = torch.arange(p.n_particles)
        new_lw = self.log_w.clone()
        for s in self._slices():
            n = s.stop - s.start
            if ess(self.log_w[s]) >= p.ess_frac * n:
                continue
            W = torch.softmax(self.log_w[s], 0)
            q = W
            if p.fold_dedup and clusters is not None:
                cl = torch.unique(clusters[s], return_inverse=True)[1]
                M = torch.zeros(int(cl.max()) + 1, dtype=torch.float64).index_add_(0, cl, W)   # weight per fold
                eq = W / (M[cl].clamp_min(1e-300) * len(M))                                   # equal mass per fold
                q = (1.0 - p.fold_mix) * W + p.fold_mix * eq
                q = q / q.sum()
            if p.max_offspring:
                q = capped(q, p.max_offspring)
            loc = systematic(q, n, self.gen)
            if q is W or (p.fold_dedup and not p.fold_correct and not p.max_offspring):
                new_lw[s] = 0.0                                   # plain resampling (or the biased legacy dedup)
            else:
                lw = torch.log(W[loc].clamp_min(1e-300)) - torch.log(q[loc].clamp_min(1e-300))
                new_lw[s] = lw - lw.max()
            parents[s] = loc + s.start
        return parents, new_lw

    def _we_resample(self, r: torch.Tensor, clusters: torch.Tensor | None = None):
        p = self.policy
        parents = torch.empty(p.n_particles, dtype=torch.long)
        new_lw = torch.empty(p.n_particles, dtype=torch.float64)
        for s in self._slices():
            if p.we_bin_by == "fold":
                if clusters is None:
                    raise ValueError("we_bin_by='fold' needs per-particle fold clusters")
                b = torch.unique(clusters[s], return_inverse=True)[1]
            else:
                b = bin_ids(r[s], p.we_bins)
            loc, lw = we_resample(self.log_w[s], b, self.gen)
            parents[s] = loc + s.start
            new_lw[s] = lw - lw.max()  # per-island constant shift: relative weights unchanged
        return parents, new_lw

    # -- public API
    def score(self, reward: torch.Tensor, progress: float, step: int | None = None,
              clusters: torch.Tensor | None = None) -> torch.Tensor | None:
        """Weight update with the per-particle reward at this step; returns parent indices if resampled, else None.
        Call only for steps inside `window(progress)` (others are no-ops apart from logging)."""
        p = self.policy
        r = reward.detach().double().cpu().flatten()
        if r.numel() != p.n_particles:
            raise ValueError(f"expected {p.n_particles} rewards, got {r.numel()}")
        if not bool(torch.isfinite(r).all()):
            raise ValueError("non-finite reward")
        r_used = r[torch.randperm(r.numel(), generator=self.gen)] if p.null else r
        entry = dict(step=step, progress=float(progress), reward=r.tolist())
        parents = None
        if self.window(progress):
            self.n_scored += 1
            if p.tilted:
                lam = self._choose_lam(r_used, self._lam_sched(progress))
                self._tilt(r_used, lam)
                entry["lam"] = lam
            entry["ess"] = [ess(self.log_w[s]) for s in self._slices()]
            out = None
            if clusters is not None:
                entry["fold_clusters"] = [len(set(clusters[s].tolist())) for s in self._slices()]
            if p.scheme == "smc":
                out = self._smc_resample(clusters)
            elif p.scheme in ("we", "twe") and (self.n_scored - 1) % p.we_every == 0:
                out = self._we_resample(r_used, clusters)
            elif p.scheme == "fork" and self.n_scored == 1:
                # diagnostic: duplicate every even slot into the next odd slot ONCE (pairs of siblings); weights untouched
                par = torch.arange(p.n_particles)
                par = par - (par % 2)
                out = (par, self.log_w.clone())
            if out is not None:
                parents, new_lw = out
                self._apply(parents, new_lw)
                entry["parents"] = parents.tolist()
                entry["lineages"] = [len(set(self.ancestry[s].tolist())) for s in self._slices()]
        entry["log_w"] = self.log_w.tolist()
        self.log.append(entry)
        return parents

    def finish(self, reward: torch.Tensor) -> dict:
        """Terminal weight on the FINAL structures: tilted schemes add lam_max r_final - pot (the product of all
        increments telescopes to exp(lam_max r_final)); WE keeps its weights. Optional terminal resampling.
        Returns dict(parents, weights [K] normalised per island, log_w, ancestry, lam, log)."""
        p = self.policy
        r = reward.detach().double().cpu().flatten()
        if p.null:
            r = r[torch.randperm(r.numel(), generator=self.gen)]
        if p.tilted:
            self._tilt(r, p.lam_max)
        parents = torch.arange(p.n_particles)
        if p.terminal_resample and p.scheme != "none":
            for s in self._slices():
                n = s.stop - s.start
                parents[s] = systematic(torch.softmax(self.log_w[s], 0), n, self.gen) + s.start
            self._apply(parents, torch.zeros(p.n_particles, dtype=torch.float64))
        weights = torch.empty(p.n_particles, dtype=torch.float64)
        for s in self._slices():
            weights[s] = torch.softmax(self.log_w[s], 0)
        self.log.append(dict(step="final", reward=reward.detach().double().cpu().flatten().tolist(), lam=self.lam,
                             ess=[ess(self.log_w[s]) for s in self._slices()], weights=weights.tolist(),
                             parents=parents.tolist(), lineages=[len(set(self.ancestry[s].tolist())) for s in self._slices()]))
        return dict(parents=parents, weights=weights, log_w=self.log_w.clone(), ancestry=self.ancestry.clone(),
                    lam=self.lam, policy=asdict(p), log=self.log)


# ------------------------------------------------------------------------------------------------------- burial reward
@dataclass
class BurialReward:
    """PTM-specific burial on clean-predicted coordinates (Angstrom), SE(3)-invariant, no gradients needed.

    n_P = mean over PTM atoms of sum over binder atoms of s(d), s(d) = 1 / (1 + ((d - d0)+ / r0)^6) (soft neighbour count);
    n_Q = the same over the other target atoms. reward = min(n_P / n_sat, 1) - alpha * max(0, n_Q - n_P) / n_sat
    - clash_w * (# binder-target atom pairs closer than clash_d) / n_P_atoms. Rationale (N001): steer the binder onto the
    modification, not onto the whole peptide (Bauer et al. PMC12621789 drop designs with > 12 peptide H-bonds)."""
    n_sat: float = 12.0
    r0: float = 6.0
    d0: float = 2.0
    alpha: float = 0.5
    clash_d: float = 2.5
    clash_w: float = 0.1
    # mode 'count': soft neighbour counts (above). mode 'solid': solid-angle burial = fraction of n_dir directions from each
    # atom that are blocked by a binder atom within ray_len along the ray and ray_r of it (bounded in [0, 1]; does not
    # saturate the way counts do once the binder sits on the site - N002: counts 43-77 for every patch-hotspot design).
    mode: str = "count"
    n_dir: int = 64
    ray_len: float = 10.0
    ray_r: float = 2.5

    @staticmethod
    def _fib_dirs(n: int, device) -> torch.Tensor:
        i = torch.arange(n, dtype=torch.float32, device=device) + 0.5
        phi = torch.acos(1 - 2 * i / n)
        th = math.pi * (1 + 5 ** 0.5) * i
        return torch.stack([torch.cos(th) * torch.sin(phi), torch.sin(th) * torch.sin(phi), torch.cos(phi)], -1)

    def solid_burial(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a [K,m,3], b [K,n,3] -> [K,m] fraction of directions from each a atom blocked by b atoms."""
        u = self._fib_dirs(self.n_dir, a.device)                              # [R,3]
        v = b[:, None] - a[:, :, None]                                        # [K,m,n,3]
        t = v @ u.T                                                           # [K,m,n,R] projection on each ray
        perp2 = (v * v).sum(-1, keepdim=True) - t * t
        hit = (t > 0) & (t < self.ray_len) & (perp2 < self.ray_r ** 2)
        return hit.any(2).float().mean(-1)

    def soft_count(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a [K,m,3], b [K,n,3] -> [K,m] soft count of b atoms around each a atom."""
        d = torch.cdist(a, b)
        x = (d - self.d0).clamp_min(0.0) / self.r0
        return (1.0 / (1.0 + x ** 6)).sum(-1)

    def terms(self, X: torch.Tensor, ptm: torch.Tensor, other: torch.Tensor, binder: torch.Tensor) -> dict:
        """X [K,L,3] (Angstrom), boolean masks [L] (shared by all particles). Returns per-particle tensors."""
        X = X.detach().float()
        P, Q, B = X[:, ptm], X[:, other], X[:, binder]
        tgt = torch.cat([P, Q], 1)
        clash = (torch.cdist(tgt, B) < self.clash_d).sum((-2, -1)).float()
        n_p = self.soft_count(P, B).mean(-1)
        n_q = self.soft_count(Q, B).mean(-1) if Q.shape[1] else torch.zeros_like(n_p)
        out = dict(n_ptm=n_p, n_other=n_q, clash=clash)
        if self.mode == "solid":
            b_p = self.solid_burial(P, B).mean(-1)
            b_q = self.solid_burial(Q, B).mean(-1) if Q.shape[1] else torch.zeros_like(b_p)
            reward = b_p - self.alpha * (b_q - b_p).clamp_min(0.0)
            out.update(bur_ptm=b_p, bur_other=b_q)
        elif self.mode == "count":
            reward = (n_p / self.n_sat).clamp(max=1.0) - self.alpha * (n_q - n_p).clamp_min(0.0) / self.n_sat
        else:
            raise ValueError(f"unknown burial mode {self.mode!r}")
        out["reward"] = reward - self.clash_w * clash / max(int(ptm.sum()), 1)
        return out

    def __call__(self, X, ptm, other, binder) -> torch.Tensor:
        return self.terms(X, ptm, other, binder)["reward"]
