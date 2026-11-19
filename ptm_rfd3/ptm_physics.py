"""Cheap, PTM-general physics terms on top of PTM burial (milliseconds per step; no co-folding, no learned judge).

Everything is evaluated on the atoms the modification ADDS (ptm_rfd3.rfd3_inputs.detect_sites: atoms not in the parent
residue + the anchor), so each term measures what the modified target has and the unmodified one lacks.

Chemistry (static, per example) from the PTM residue's own heavy-atom bond graph - no residue-specific tables:
  pH-7 group charges   phosphate/phosphonate (P with >= 3 terminal O): -2 on a monoester, -1 per remaining P-O-
                        otherwise, spread over its terminal O; sulfate/sulfonate (S with 3 terminal O): -1; carboxylate
                        (C with 2 terminal O): -1; sp3 amine / alkyl-ammonium N (no carbonyl / ring neighbour): +1;
                        guanidinium / amidine C (3 N or 2 N + C, no O): +1 over its N. Everything else neutral
                        (amides, acetyl, urea of citrulline, ethers, sugars).
  polarity             N, O polar; C, S apolar.
  solvation parameter  Eisenberg & McLachlan 1986 atomic solvation parameters (cal/mol/A^2): C 16, N/O -6, S 21,
                        charged O- -24, charged N+ -50.

Terms (per particle; coordinates in Angstrom):
  burial     solid-angle burial b_a of each added atom by binder atoms (ptm_rfd3.particles.BurialReward.solid_burial)
  dG_solv    = - sum_a sigma_a * A0 * b_a / 1000   (kcal/mol; burying apolar atoms favourable, charged ones costly)
  E_elec     = 332 * sum_{a,j} q_a q_j / (4 r^2)  (distance-dependent dielectric eps = 4r, r >= 3 A, smooth cut 12 A);
               binder charges q_j = p_j(R) + p_j(K) - p_j(D) - p_j(E) + 0.1 p_j(H) from the per-step sequence
               probabilities, placed 2.4 A beyond CB along CA->CB (a side-chain charge centre)
  unsat      buried (b_a > 0.5) polar added atoms without a partner: binder backbone N/O within 3.5 A, or polar side
             chains (p of STNQDEKRHYW) whose CB lies within ~6-7.5 A; counted per PTM atom
  reward     = burial_mean - w_solv dG_solv - w_elec E_elec - w_unsat unsat - clash term (burial primary, K006)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

AA20 = "ARNDCQEGHILKMFPSTWYV"           # RFD3 indices 0..19, RFD4 indices 1..20 (alphabetical three-letter order)
_CHARGE20 = torch.tensor([{"R": 1.0, "K": 1.0, "D": -1.0, "E": -1.0, "H": 0.1}.get(a, 0.0) for a in AA20])
_POLAR20 = torch.tensor([1.0 if a in "STNQDEKRHYW" else 0.0 for a in AA20])
ASP = {"C": 16.0, "N": -6.0, "O": -6.0, "S": 21.0, "O-": -24.0, "N+": -50.0}


def group_charges(elements: list[str], adj: list[set[int]], in_ring: list[bool] | None = None) -> np.ndarray:
    """pH-7 formal charges from a heavy-atom graph (see module docstring). elements upper-case, adj = neighbour sets."""
    n = len(elements)
    el = [e.upper() for e in elements]
    ring = in_ring or [False] * n
    q = np.zeros(n)
    term_o = lambda i: [j for j in adj[i] if el[j] == "O" and len(adj[j]) == 1]
    for i in range(n):
        if el[i] == "P":
            to = term_o(i)
            if len(to) >= 2:
                nb_o = [j for j in adj[i] if el[j] == "O"]
                bridging = len(nb_o) - len(to)
                charge = -2.0 if (len(to) >= 3 and bridging <= 1) else -1.0
                for j in to:
                    q[j] += charge / len(to)
        elif el[i] == "S":
            to = term_o(i)
            if len(to) >= 3:
                for j in to:
                    q[j] += -1.0 / len(to)
        elif el[i] == "C":
            to = term_o(i)
            nn = [j for j in adj[i] if el[j] == "N"]
            if len(to) == 2:
                for j in to:
                    q[j] += -0.5
            elif not [j for j in adj[i] if el[j] == "O"] and (len(nn) == 3 or (len(nn) == 2 and len(adj[i]) == 3)) \
                    and not ring[i]:
                for j in nn:
                    q[j] += 1.0 / len(nn)
    for i in range(n):
        if el[i] != "N" or ring[i] or q[i] != 0:
            continue
        nbs = adj[i]
        if any(el[j] != "C" for j in nbs):
            continue
        carbonyl = any(any(el[k] == "O" and len(adj[k]) == 1 for k in adj[j]) for j in nbs)
        guan = any(sum(el[k] == "N" for k in adj[j]) >= 2 for j in nbs)
        if not carbonyl and not guan:
            q[i] = 1.0
    return q


def ring_flags(adj: list[set[int]]) -> list[bool]:
    """Atom i is in a ring iff for some bond (i, j), j is still reachable from i without that bond."""
    n = len(adj)
    ring = [False] * n
    for i in range(n):
        for j in adj[i]:
            seen, stack = {i}, [k for k in adj[i] if k != j]
            seen.update(stack)
            while stack:
                v = stack.pop()
                if v == j:
                    ring[i] = True
                    break
                for w in adj[v]:
                    if w not in seen:
                        seen.add(w); stack.append(w)
            if ring[i]:
                break
    return ring


def residue_graph(atom_array, idx: np.ndarray) -> tuple[list[str], list[set[int]], list[bool]]:
    """Elements, neighbour sets and ring flags of the atoms `idx` of a biotite AtomArray WITH bonds (graph restricted
    to those atoms). Falls back to biotite's CCD (connect_via_residue_names) when the array carries no bonds."""
    from biotite import structure as struc

    sub = atom_array[idx]
    bonds = sub.bonds if sub.bonds is not None and sub.bonds.get_bond_count() else struc.connect_via_residue_names(sub)
    n = len(sub)
    adj = [set() for _ in range(n)]
    for a, b, *_ in bonds.as_array():
        if a < n and b < n and sub.element[a] != "H" and sub.element[b] != "H":
            adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    ring = ring_flags(adj)
    return [str(e).upper() for e in sub.element], adj, ring


@dataclass
class PTMChem:
    """Static chemistry of the PTM atoms (in the order of the PTM mask)."""
    charge: torch.Tensor     # [m]
    polar: torch.Tensor      # [m] bool
    sigma: torch.Tensor      # [m] cal/mol/A^2

    @classmethod
    def from_lists(cls, elements: list[str], charges) -> "PTMChem":
        q = torch.as_tensor(np.asarray(charges, dtype=float), dtype=torch.float32)
        el = [e.upper() for e in elements]
        sig = []
        for e, c in zip(el, q.tolist()):
            key = e + ("-" if c <= -0.3 else "+" if c >= 0.3 else "")
            sig.append(ASP.get(key, ASP.get(e, 0.0)))
        return cls(charge=q, polar=torch.tensor([e in ("N", "O") for e in el]), sigma=torch.tensor(sig, dtype=torch.float32))

    def to(self, device):
        return PTMChem(self.charge.to(device), self.polar.to(device), self.sigma.to(device))


def ptm_chem_for(atom_array, ptm_idx: np.ndarray) -> PTMChem:
    """Chemistry of the PTM atoms `ptm_idx` (indices into atom_array), computed on each modified residue's full graph."""
    import biotite.structure as struc  # noqa: F401

    chain = np.asarray(atom_array.chain_id).astype(str)
    resid = np.asarray(atom_array.res_id)
    charges = np.zeros(len(ptm_idx))
    pos = {int(i): k for k, i in enumerate(ptm_idx)}
    for c, r in sorted({(chain[i], int(resid[i])) for i in ptm_idx}):
        res_idx = np.where((chain == c) & (resid == r) & (np.asarray(atom_array.element).astype(str) != "H"))[0]
        el, adj, ring = residue_graph(atom_array, res_idx)
        q = group_charges(el, adj, ring)
        for k, i in enumerate(res_idx):
            if int(i) in pos:
                charges[pos[int(i)]] = q[k]
    return PTMChem.from_lists([str(e) for e in np.asarray(atom_array.element)[ptm_idx]], charges)


def virtual_cb(ca: torch.Tensor) -> torch.Tensor:
    """[K, n, 3] CA trace (one chain, in order) -> approximate CB 1.53 A from CA along the bisector pointing away from the
    neighbours (ends use their single neighbour)."""
    prev = torch.cat([ca[:, :1], ca[:, :-1]], 1)
    nxt = torch.cat([ca[:, 1:], ca[:, -1:]], 1)
    d = (ca - prev) + (ca - nxt)
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return ca + 1.53 * d


@dataclass
class PhysicsTerms:
    w_solv: float = 0.05
    w_elec: float = 0.01
    w_unsat: float = 0.2
    a0: float = 15.0          # exposed area per added atom (A^2) used with the solvation parameters
    r_cut: float = 12.0
    unsat_burial: float = 0.5
    hb_d: float = 3.5
    pol_cb_d: float = 7.5      # a polar side chain counts fully with CB <= 6 A, fading out at 7.5 A (Arg/Lys/Glu reach 5-7 A)
    charge_offset: float = 2.4

    def compute(self, P: torch.Tensor, burial: torch.Tensor, chem: PTMChem, ca: torch.Tensor, cb: torch.Tensor,
                probs: torch.Tensor | None, bb_polar: torch.Tensor | None) -> dict:
        """P [K,m,3] PTM atoms; burial [K,m]; ca, cb [K,nres,3] binder residues; probs [K,nres,20] or None;
        bb_polar [K,p,3] binder backbone N/O (or None). Returns per-particle tensors."""
        chem = chem.to(P.device)
        dG = -(chem.sigma[None] * self.a0 * burial).sum(-1) / 1000.0
        if probs is not None:
            q = (probs * _CHARGE20.to(probs)).sum(-1)                     # [K, nres]
            ppol = (probs * _POLAR20.to(probs)).sum(-1)
        else:
            q = torch.zeros(ca.shape[:2], device=P.device)
            ppol = torch.full(ca.shape[:2], 0.5, device=P.device)
        u = cb - ca
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        site = cb + self.charge_offset * u                                    # side-chain charge centre
        r = torch.cdist(P, site).clamp_min(3.0)                               # [K, m, nres]
        sw = torch.clamp((self.r_cut - r) / 2.0, 0.0, 1.0)                   # smooth cut-off over the last 2 A
        E = 332.0 * (chem.charge[None, :, None] * q[:, None, :] / (4.0 * r * r) * sw).sum((-2, -1))
        dcb = torch.cdist(P, cb)
        partner = (ppol[:, None, :] * torch.clamp((self.pol_cb_d - dcb) / 1.5, 0.0, 1.0)).sum(-1)
        if bb_polar is not None and bb_polar.shape[1]:
            partner = partner + (torch.cdist(P, bb_polar) < self.hb_d).float().sum(-1)
        need = chem.polar[None].float() * (burial > self.unsat_burial).float()
        unsat = (need * (1.0 - partner.clamp(max=1.0))).sum(-1) / max(int(chem.polar.sum()), 1)
        total = -self.w_solv * dG - self.w_elec * E - self.w_unsat * unsat
        return dict(dG_solv=dG, E_elec=E, unsat=unsat, physics=total)


@dataclass
class PTMReward:
    """PTM burial (primary, ptm_rfd3.particles.BurialReward) + optional physics terms (PhysicsTerms) in one reward."""
    burial: object
    physics: PhysicsTerms | None = None

    def terms(self, X, ptm, other, binder, chem: PTMChem | None = None, ca_idx=None, cb_idx=None, bbpol_idx=None,
              probs=None) -> dict:
        """X [K,L,3] Angstrom; boolean masks [L]; ca_idx/cb_idx/bbpol_idx index tensors into L (cb_idx None -> virtual CB
        from the CA trace); probs [K, nres, 20] residue probabilities of the binder residues (ca_idx order)."""
        t = self.burial.terms(X, ptm, other, binder)
        if self.physics is not None and chem is not None and ca_idx is not None:
            X = X.detach().float()
            P = X[:, ptm]
            b = self.burial.solid_burial(P, X[:, binder])
            ca = X[:, ca_idx]
            cb = X[:, cb_idx] if cb_idx is not None else virtual_cb(ca)
            bbp = X[:, bbpol_idx] if bbpol_idx is not None else None
            ph = self.physics.compute(P, b, chem, ca, cb, probs, bbp)
            t.update(ph)
            t["reward"] = t["reward"] + ph["physics"]
        return t


def physics_from_env(env) -> PhysicsTerms | None:
    """PTMSMC_PHYSICS=1 switches the physics terms on; PTMSMC_W_SOLV / PTMSMC_W_ELEC / PTMSMC_W_UNSAT / ... override."""
    if env.get("PTMSMC_PHYSICS", "0") not in ("1", "true", "yes"):
        return None
    from dataclasses import fields
    kw = {}
    for f in fields(PhysicsTerms):
        v = env.get("PTMSMC_" + f.name.upper())
        if v not in (None, ""):
            kw[f.name] = type(f.default)(v)
    return PhysicsTerms(**kw)
