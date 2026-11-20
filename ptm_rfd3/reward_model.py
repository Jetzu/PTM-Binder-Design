"""Multi-component, fully tunable reward for particle guidance (RFD3 and RFD4), configured from ONE file.

    PTMSMC_CONFIG=my.yaml   (YAML or JSON; every key optional)

    policy:   {scheme: smc, lam_max: 40, cess_frac: 0.5, start: 0.6, fold_dedup: true, fold_mix: 0.5, ...}
              (any ptm_rfd3.particles.ParticlePolicy field)
    geometry: {mode: solid, n_dir: 64, ray_len: 10, ray_r: 2.5, r0: 6, d0: 2, n_sat: 12, clash_d: 2.5}
              (ptm_rfd3.particles.BurialReward fields: the shared geometry of the burial / contact / clash terms)
    physics:  {a0: 15, r_cut: 12, unsat_burial: 0.5, hb_d: 3.5, pol_cb_d: 7.5, charge_offset: 2.4}
              (ptm_rfd3.ptm_physics.PhysicsTerms fields)
    terms:    - {name: ptm_burial, weight: 1.0}
              - {name: electrostatics, weight: 0.01,
                 schedule: {start: 0.7, stop: 1.0, shape: linear, floor: 0.0, hold: true},
                 normalize: {mode: scale, scale: 10.0, clip: [-3, 3]}}

Every term is "higher is better" (penalties are returned negative). The reward at sampler progress p (0 = first step,
1 = final structure) is   R(x, p) = sum_k  weight_k * m_k(p) * norm_k(term_k(x)),
with the per-term time schedule m_k(p):
    0 before `start`; inside [start, stop]: floor + (1 - floor) * shape(f), f = (p - start) / (stop - start), shape one of
    const (1), linear (f), cosine (0.5 - 0.5 cos(pi f)), power (f ** power), step (1 at f >= 1 else 0);
    after `stop`: the value at stop if `hold` (default) else 0.
The FINAL structures are scored at p = 1, so the SMC / tilted-WE TARGET is q_generator(x) * exp(lam_max * R(x, 1)): a term
whose schedule is 0 at p = 1 shapes the path (a twisting heuristic that the importance weights correct for) but not the
target. `zscore` normalisation is per step across particles (scale-free; use it only for path terms: at p = 1 it makes the
target population-dependent).

Environment overrides (win over the file): PTMSMC_<POLICY FIELD>, PTMSMC_<GEOMETRY FIELD>, PTMSMC_<PHYSICS FIELD>,
PTMSMC_W_<TERM> (weight; adds the term if absent), PTMSMC_TERMS="ptm_burial:1,clash:0.1" (replaces the term list).
Legacy (no config file): PTMSMC_ALPHA -> peptide_overburial, PTMSMC_CLASH_W -> clash, PTMSMC_PHYSICS=1 -> solvation /
electrostatics / unsat_polar with PTMSMC_W_SOLV / _W_ELEC / _W_UNSAT. The default reproduces the pre-config reward exactly
(tests/test_reward_model.py).

Terms (P = PTM atoms = atoms the modification adds, Q = other target atoms, B = binder atoms):
  ptm_burial          mean solid-angle burial of P by B (mode solid) or min(n_P / n_sat, 1) soft contacts (mode count)
  ptm_contacts        min(n_P / n_sat, 1), soft neighbour count of B around P (always the count form)
  peptide_overburial  -max(0, burial(Q) - burial(P))  (binding the peptide more than the modification)
  contact_share       n_P / (n_P + n_Q): fraction of the binder's target contacts that land on the modification
  clash               -(# target-binder atom pairs closer than clash_d) / |P|
  solvation           -dG_solv of P (Eisenberg-McLachlan, ptm_physics)
  electrostatics      -E_elec between P charges and the binder's expected side-chain charges (ptm_physics)
  unsat_polar         -(buried polar P atoms without a partner) / |polar P| (ptm_physics)
  compactness         -max(0, Rg(binder CA) / (2.2 N^0.38) - 1)
  ptm_close_contact   -(sum over binder atoms closer than close_d to a P atom of (close_d - d) / close_d) / |P|:
                      steric overlap WITH the modification (RFD3: all predicted binder atoms incl. side-chain slots;
                      RFD4: CA, close_d 4.0). N006: 56 % of designs fail the pipeline gate on a binder atom <= 2.4 A from P.
  charge_partner      per charged P atom (|q| >= 0.3): sum_j p_j(complementary) * ramp(d(charge centre_j, a)), capped at
                      1, averaged over charged atoms; complementary = R/K (+0.5 H) for anions, D/E + 0.5 W/Y/F (aromatic
                      cage, cation-pi) for cations; 0 for neutral PTMs. params: r_on (full credit, 5.0 A), ramp (1.5 A).
  interface_contacts  fraction of ALL target atoms (P and Q: the whole peptide) with a binder atom within d_on (soft, +ramp):
                      rewards forming an interface with the peptide at all (RFD3 default d_on 6 A on N/CA/C/O/CB; RFD4 8 A
                      on CA + virtual CB). params: d_on, ramp (1.5 A).
  peptide_burial      mean solid-angle burial of ALL target atoms by the binder (enclosing the peptide, not just touching it)
Every term takes optional `params: {...}` in the config (passed to the term function).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch

from .particles import BurialReward
from .ptm_physics import PhysicsTerms, PTMChem, virtual_cb

SHAPES = ("const", "linear", "cosine", "power", "step")
NORMS = ("none", "scale", "zscore")


@dataclass
class Schedule:
    start: float = 0.0
    stop: float = 1.0
    shape: str = "const"
    power: float = 2.0
    floor: float = 0.0
    hold: bool = True

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"schedule shape must be one of {SHAPES}")
        if not 0.0 <= self.start <= self.stop <= 1.0:
            raise ValueError("schedule needs 0 <= start <= stop <= 1")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("schedule floor must be in [0, 1]")

    def __call__(self, p: float) -> float:
        if p < self.start:
            return 0.0
        if p > self.stop:
            if not self.hold:
                return 0.0
            p = self.stop
        f = 1.0 if self.stop <= self.start else (p - self.start) / (self.stop - self.start)
        s = {"const": 1.0, "linear": f, "cosine": 0.5 - 0.5 * math.cos(math.pi * f), "power": f ** self.power,
             "step": 1.0 if f >= 1.0 else 0.0}[self.shape]
        return self.floor + (1.0 - self.floor) * s


@dataclass
class Normalize:
    mode: str = "none"
    scale: float = 1.0
    clip: tuple | None = None

    def __post_init__(self):
        if self.mode not in NORMS:
            raise ValueError(f"normalize mode must be one of {NORMS}")
        if self.scale <= 0:
            raise ValueError("normalize scale must be > 0")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "scale":
            x = x / self.scale
        elif self.mode == "zscore":
            sd = x.std()
            x = (x - x.mean()) / sd if float(sd) > 1e-12 else torch.zeros_like(x)
        if self.clip is not None:
            x = x.clamp(float(self.clip[0]), float(self.clip[1]))
        return x


@dataclass
class TermSpec:
    name: str
    weight: float = 1.0
    schedule: Schedule = field(default_factory=Schedule)
    normalize: Normalize = field(default_factory=Normalize)
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "TermSpec":
        d = dict(d)
        if d.get("name") not in TERMS:
            raise ValueError(f"unknown reward term {d.get('name')!r}; known: {sorted(TERMS)}")
        sch = d.pop("schedule", None) or {}
        nrm = d.pop("normalize", None) or {}
        if "clip" in nrm and nrm["clip"] is not None:
            nrm["clip"] = tuple(nrm["clip"])
        return cls(schedule=Schedule(**sch), normalize=Normalize(**nrm), **d)


class Ctx:
    """Lazily computed shared intermediates for one batch of particles."""

    def __init__(self, model, X, ptm, other, binder, chem, ca_idx, cb_idx, bbpol_idx, probs, binder_all=None):
        self.m, self.X = model, X.detach().float()
        self.ptm, self.other, self.binder = ptm, other, binder
        self.binder_all = binder if binder_all is None else binder_all
        self.chem, self.ca_idx, self.cb_idx, self.bbpol_idx, self.probs = chem, ca_idx, cb_idx, bbpol_idx, probs
        self._c = {}

    def get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    @property
    def P(self):
        return self.X[:, self.ptm]

    @property
    def Q(self):
        return self.X[:, self.other]

    @property
    def B(self):
        return self.X[:, self.binder]

    def bur_p_atoms(self):
        return self.get("bpa", lambda: self.m.geometry.solid_burial(self.P, self.B))

    def bur_p(self):
        return self.get("bp", lambda: self.bur_p_atoms().mean(-1))

    def bur_q(self):
        return self.get("bq", lambda: self.m.geometry.solid_burial(self.Q, self.B).mean(-1) if self.Q.shape[1]
                        else torch.zeros_like(self.bur_p()))

    def n_p(self):
        return self.get("np", lambda: self.m.geometry.soft_count(self.P, self.B).mean(-1))

    def n_q(self):
        return self.get("nq", lambda: self.m.geometry.soft_count(self.Q, self.B).mean(-1) if self.Q.shape[1]
                        else torch.zeros_like(self.n_p()))

    def clash(self):
        def f():
            tgt = torch.cat([self.P, self.Q], 1)
            return (torch.cdist(tgt, self.B) < self.m.geometry.clash_d).sum((-2, -1)).float()
        return self.get("clash", f)

    def physics(self):
        def f():
            if self.chem is None or self.ca_idx is None:
                return None
            ca = self.X[:, self.ca_idx]
            cb = self.X[:, self.cb_idx] if self.cb_idx is not None else virtual_cb(ca)
            bbp = self.X[:, self.bbpol_idx] if self.bbpol_idx is not None else None
            return self.m.physics.compute(self.P, self.bur_p_atoms(), self.chem, ca, cb, self.probs, bbp)
        return self.get("phys", f)


def _phys(key, sign=-1.0):
    def term(c: Ctx, **_):
        ph = c.physics()
        return None if ph is None else sign * ph[key]
    return term


def _burial(c: Ctx, **_):
    return c.bur_p() if c.m.geometry.mode == "solid" else (c.n_p() / c.m.geometry.n_sat).clamp(max=1.0)


def _overburial(c: Ctx, **_):
    if c.m.geometry.mode == "solid":
        return -(c.bur_q() - c.bur_p()).clamp_min(0.0)
    return -(c.n_q() - c.n_p()).clamp_min(0.0) / c.m.geometry.n_sat


def _close_contact(c: Ctx, close_d: float = 2.6):
    d = torch.cdist(c.P, c.X[:, c.binder_all])
    return -((close_d - d).clamp_min(0.0) / close_d).sum((-2, -1)) / max(int(c.ptm.sum()), 1)


def _charge_partner(c: Ctx, r_on: float = 5.0, ramp: float = 1.5, offset: float = 2.4):
    if c.chem is None or c.ca_idx is None or c.probs is None:
        return None
    from .ptm_physics import AA20
    q = c.chem.charge.to(c.X.device)
    charged = q.abs() >= 0.3
    if not bool(charged.any()):
        return torch.zeros(c.X.shape[0], device=c.X.device)
    idx = lambda letters, w: sum(w * c.probs[..., AA20.index(a)] for a in letters)
    p_cat = idx("RK", 1.0) + idx("H", 0.5)                               # partners for anionic groups
    p_an = idx("DE", 1.0) + idx("WYF", 0.5)                              # partners for cationic groups (incl. cation-pi)
    ca = c.X[:, c.ca_idx]
    cb = c.X[:, c.cb_idx] if c.cb_idx is not None else virtual_cb(ca)
    u = cb - ca
    site = cb + offset * u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    Pc = c.P[:, charged]
    w = ((r_on + ramp - torch.cdist(Pc, site)) / ramp).clamp(0.0, 1.0)  # [K, mc, nres]
    comp = torch.where(q[charged][None, :, None] < 0, p_cat[:, None, :], p_an[:, None, :])
    return (w * comp).sum(-1).clamp(max=1.0).mean(-1)


def _interface_contacts(c: Ctx, d_on: float = 6.0, ramp: float = 1.5):
    tgt = torch.cat([c.P, c.Q], 1)
    dmin = torch.cdist(tgt, c.B).min(-1).values                       # [K, n_target]
    return ((d_on + ramp - dmin) / ramp).clamp(0.0, 1.0).mean(-1)


def _peptide_burial(c: Ctx, **_):
    tgt = torch.cat([c.P, c.Q], 1)
    return c.get("bt", lambda: c.m.geometry.solid_burial(tgt, c.B).mean(-1))


def _compact(c: Ctx, **_):
    if c.ca_idx is None:
        return None
    ca = c.X[:, c.ca_idx]
    rg = ((ca - ca.mean(1, keepdim=True)) ** 2).sum(-1).mean(-1).sqrt()
    return -(rg / (2.2 * ca.shape[1] ** 0.38) - 1.0).clamp_min(0.0)


TERMS = {
    "ptm_burial": _burial,
    "ptm_contacts": lambda c, **_: (c.n_p() / c.m.geometry.n_sat).clamp(max=1.0),
    "peptide_overburial": _overburial,
    "contact_share": lambda c, **_: c.n_p() / (c.n_p() + c.n_q()).clamp_min(1e-6),
    "clash": lambda c, **_: -c.clash() / max(int(c.ptm.sum()), 1),
    "solvation": _phys("dG_solv"),
    "electrostatics": _phys("E_elec"),
    "unsat_polar": _phys("unsat"),
    "compactness": _compact,
    "ptm_close_contact": _close_contact,
    "charge_partner": _charge_partner,
    "interface_contacts": _interface_contacts,
    "peptide_burial": _peptide_burial,
}
LEGACY_W = {"SOLV": "solvation", "ELEC": "electrostatics", "UNSAT": "unsat_polar"}


@dataclass
class RewardModel:
    terms: list
    geometry: BurialReward = field(default_factory=BurialReward)
    physics: PhysicsTerms = field(default_factory=PhysicsTerms)

    def evaluate(self, X, ptm, other, binder, chem: PTMChem | None = None, ca_idx=None, cb_idx=None, bbpol_idx=None,
                 probs=None, progress: float = 1.0, binder_all=None) -> dict:
        """Per-particle tensors: 'reward' (the weighted, scheduled, normalised sum at `progress`), every term's raw value
        under its name, the diagnostics n_ptm / n_other / clash / bur_ptm / bur_other."""
        c = Ctx(self, X, ptm, other, binder, chem, ca_idx, cb_idx, bbpol_idx, probs, binder_all)
        total = torch.zeros(X.shape[0], device=X.device)
        out = {}
        for t in self.terms:
            v = TERMS[t.name](c, **t.params)
            if v is None:                                   # term needs data this batch does not have (e.g. chemistry)
                continue
            out[t.name] = v
            w = t.weight * t.schedule(progress)
            if w:
                total = total + w * t.normalize(v)
        out.update(n_ptm=c.n_p(), n_other=c.n_q(), clash_count=c.clash())
        if self.geometry.mode == "solid":
            out.update(bur_ptm=c.bur_p(), bur_other=c.bur_q())
        out["reward"] = total
        return out

    # same call signature as the older BurialReward / PTMReward (hooks call .terms(..., progress=p))
    def terms_at(self, *a, **kw):
        return self.evaluate(*a, **kw)


def _cast(cls, d: dict) -> dict:
    types = {f.name: type(f.default) for f in fields(cls)}
    return {k: (types[k](v) if k in types and not isinstance(v, types[k]) else v) for k, v in d.items() if k in types}


def load(env=None, generator: str = "rfd3") -> tuple[dict, RewardModel]:
    """(policy dict as PTMSMC_* strings for ParticlePolicy.from_env, RewardModel) from PTMSMC_CONFIG + env overrides."""
    env = dict(os.environ if env is None else env)
    cfg = {}
    if env.get("PTMSMC_CONFIG"):
        txt = Path(env["PTMSMC_CONFIG"]).read_text()
        if env["PTMSMC_CONFIG"].endswith((".yaml", ".yml")):
            import yaml
            cfg = yaml.safe_load(txt) or {}
        else:
            cfg = json.loads(txt)
    unknown = set(cfg) - {"policy", "geometry", "physics", "terms"}
    if unknown:
        raise ValueError(f"PTMSMC_CONFIG: unknown sections {sorted(unknown)}")
    # RFD4: binder = CA + virtual CB (no side chains while sampling) -> coarser geometry, solid-angle burial by default
    geo = dict(mode="solid", r0=8.0, d0=3.8, n_sat=6.0, clash_d=3.0, ray_r=3.0) if generator == "rfd4" else {}
    geo.update(cfg.get("geometry") or {})
    for f in fields(BurialReward):
        v = env.get("PTMSMC_" + f.name.upper())
        if v not in (None, ""):
            geo[f.name] = v
    phys = dict(cfg.get("physics") or {})
    for f in fields(PhysicsTerms):
        v = env.get("PTMSMC_" + f.name.upper())
        if v not in (None, "") and not f.name.startswith("w_"):
            phys[f.name] = v
    geometry = BurialReward(**_cast(BurialReward, geo))
    physics = PhysicsTerms(**_cast(PhysicsTerms, phys))
    if env.get("PTMSMC_TERMS"):
        specs = [TermSpec.from_dict({"name": k.strip(), "weight": float(w)})
                 for k, w in (x.split(":") for x in env["PTMSMC_TERMS"].split(",") if x.strip())]
    elif cfg.get("terms"):
        specs = [TermSpec.from_dict(d) for d in cfg["terms"]]
    else:                                                    # legacy default == the pre-config reward
        specs = [TermSpec("ptm_burial", 1.0), TermSpec("peptide_overburial", geometry.alpha),
                 TermSpec("clash", geometry.clash_w)]
        if env.get("PTMSMC_PHYSICS", "0") in ("1", "true", "yes"):
            specs += [TermSpec("solvation", physics.w_solv), TermSpec("electrostatics", physics.w_elec),
                      TermSpec("unsat_polar", physics.w_unsat)]
    if generator == "rfd4":                                    # CA-only binder: closest CA to a PTM atom ~4 A
        for sp in specs:
            if sp.name == "ptm_close_contact":
                sp.params.setdefault("close_d", 4.0)
            if sp.name == "interface_contacts":
                sp.params.setdefault("d_on", 8.0)
    by = {s.name: s for s in specs}
    for k, v in env.items():                                 # PTMSMC_W_<TERM> weight overrides
        if k.startswith("PTMSMC_W_") and v not in (None, ""):
            name = LEGACY_W.get(k[len("PTMSMC_W_"):], k[len("PTMSMC_W_"):].lower())
            if name in by:
                by[name].weight = float(v)
            elif name in TERMS:
                specs.append(TermSpec(name, float(v)))
                by[name] = specs[-1]
            else:
                raise ValueError(f"{k}: unknown reward term {name!r}; known: {sorted(TERMS)}")
    pol = {("PTMSMC_" + k.upper()): str(v) for k, v in (cfg.get("policy") or {}).items()}
    return pol, RewardModel(specs, geometry, physics)


def describe(model: RewardModel) -> list[dict]:
    """JSON-able summary of the active terms (written into the run logs)."""
    return [dict(name=t.name, weight=t.weight, schedule=vars(t.schedule), normalize=dict(vars(t.normalize)),
                 params=dict(t.params)) for t in model.terms]
