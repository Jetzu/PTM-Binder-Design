"""RDKit-based construction of PTMs on an experimental/predicted target structure.

Two construction modes share one engine:

* replace  - the residue becomes a modified amino acid (SEP, TPO, PTR, ALY, M3L, CIR, HYP, ...).
             Parent atoms already in the structure stay EXACTLY where they are; only the new atoms
             (and any missing parent atoms) are built.
* adduct   - a covalently bound ligand residue (NAG on Asn ND2, palmitoyl, ...) is attached to an
             anchor atom of an existing residue; the anchor residue stays fixed.

Engine: build a chemically complete RDKit molecule (explicit H, CCD topology), pin the heavy atoms
that exist in the target to their coordinates, sample many ETKDG conformers with those pins as a
coordinate map, relax each with MMFF (UFF fallback) holding the pins fixed, then rank by
clash/H-bond/salt-bridge score against the surrounding target and keep a diverse top-K ensemble.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from biotite import structure as struc
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDistGeom, rdFMCS
from rdkit.Chem import rdForceFieldHelpers as FFH
from rdkit.Geometry import Point3D

from . import ccd
from .scoring import DESC_KEYS as DESC_KEYS_
from .scoring import EXT_KEYS, ext_descriptors
from .scoring import Environment, coulomb_and_cationpi, descriptors, score_conformer, total_score
from .structio import ResKey, new_atom_array, next_free_chain_id, residue_mask

RDLogger.DisableLog("rdApp.*")


TORS_LIB: dict | None = None     # {code: [rotor entries]} from torsion_lib.learn (train split only); None = staggered default
WANT_EXT = False         # attach shell descriptors (EXT_KEYS) too (needed to TRAIN ranker v4; inference reads the model's keys)
USE_RANKER = True        # learned conformer ranker (ptm_rfd3/ranker.py, data/ranker.json) for codes it was trained on
WANT_DESC = False        # set True to attach learned-ranker descriptors to every candidate's `terms`


class BuildError(RuntimeError):
    pass


# --------------------------------------------------------------------------- geometry helpers
def kabsch(P: np.ndarray, Q: np.ndarray):
    """R, t minimising |R P + t - Q| (rows are points)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - R @ Pc


def _dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# exo-anomeric effect: the aglycon sits gauche to the ring oxygen. phi = (X, anchor, C1, O5) clusters at
# ~ -95 (beta-D: N-GlcNAc), +75 (alpha-D: GalNAc), -75 (alpha-L: Fuc) in the crystal glycan benchmark (N002).
_PHI_PRIOR = {("D", "beta"): -95.0, ("D", "alpha"): 75.0, ("L", "alpha"): -75.0, ("L", "beta"): 95.0}
PHI_SIGMA = 25.0


def _anomeric_prior(ligand: str) -> float | None:
    t = ccd.get_comp(ligand)["type"].lower()
    m = re.match(r"([dl])-saccharide, (alpha|beta) linking", t)
    return _PHI_PRIOR.get((m.group(1).upper(), m.group(2))) if m else None


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


# --------------------------------------------------------------------------- results
@dataclass
class Candidate:
    xyz: np.ndarray            # (n_atoms, 3) all heavy output atoms, target frame
    e_rel: float
    terms: dict
    score: float
    method: str = "etkdg"


@dataclass
class Built:
    """One constructed modification with its ranked ensemble."""
    kind: str                       # 'replace' | 'adduct'
    key: ResKey                     # anchor / modified residue
    mod_code: str                   # SEP / NAG / ...
    parent: str                     # parent residue name
    atom_names: list[str]           # output atoms (heavy, non-leaving) of the new residue
    elements: list[str]
    new_mask: np.ndarray            # True = atom built here, False = copied from target
    bonds: list[tuple[int, int, int]]   # intra-residue bonds (i, j, order) over atom_names
    candidates: list[Candidate]
    pinned: list[str] = field(default_factory=list)
    anchor_atom: str | None = None  # adduct: anchor atom name in the parent residue
    ligand_atom: str | None = None  # adduct: attach atom in the ligand
    ligand_key: ResKey | None = None
    note: str = ""
    charges: list[int] = field(default_factory=list)       # CCD formal charges of atom_names (kept in the output)

    def best(self) -> Candidate:
        return self.candidates[0]


# --------------------------------------------------------------------------- engine
def _physio_charges(mol: Chem.Mol, keep: list[int], code: str) -> np.ndarray:
    """Heuristic pH-7 partial charges on the newly built heavy atoms (phosphate/sulfate O, cations)."""
    q = np.zeros(len(keep))
    pos = {i: k for k, i in enumerate(keep)}
    for i in keep:
        a = mol.GetAtomWithIdx(i)
        k = pos[i]
        q[k] = a.GetFormalCharge()
        if a.GetSymbol() == "O":
            heavy = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
            if len(heavy) == 1 and heavy[0].GetSymbol() in ("P", "S"):
                term_o = [n for n in heavy[0].GetNeighbors() if n.GetSymbol() == "O" and
                          len([x for x in n.GetNeighbors() if x.GetAtomicNum() > 1]) == 1]
                tot = -2.0 if heavy[0].GetSymbol() == "P" else -1.0
                q[k] = tot / max(len(term_o), 1)
    nm = {mol.GetAtomWithIdx(i).GetProp("name"): pos[i] for i in keep}
    centers = [nm["NZ"]] if code in ("M3L", "MLY", "MLZ", "LYS") and "NZ" in nm else []
    centers += [pos[i] for i in keep if mol.GetAtomWithIdx(i).GetSymbol() == "N"
                and mol.GetAtomWithIdx(i).GetFormalCharge() == 1 and pos[i] not in centers]
    inv = {v: k for k, v in pos.items()}
    for c in centers:                       # spread +1 over the ammonium N (0.2) and its heavy neighbours
        nb = [n.GetIdx() for n in mol.GetAtomWithIdx(inv[c]).GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() in pos]
        q[c] = 0.2
        for n_ in nb:
            q[pos[n_]] = 0.8 / max(len(nb), 1)
    if code in ("DA2", "2MR", "ARG"):
        for n_ in ("NE", "NH1", "NH2"):
            if n_ in nm:
                q[nm[n_]] = 0.33
    return q


def _ranker_needs_ext() -> bool:
    if not USE_RANKER:
        return False
    from . import ranker as _rk
    rk = _rk.load()
    return rk is not None and any(k.startswith("e_all_") for k in rk.keys)


def _anionic_atoms(mol: Chem.Mol, keep: list[int]) -> np.ndarray:
    """Terminal O on P/S(VI) (phosphate, sulfate, phosphonate) -> physiological anions."""
    flags = []
    for i in keep:
        a = mol.GetAtomWithIdx(i)
        anion = False
        if a.GetSymbol() == "O":
            heavy = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
            if len(heavy) == 1 and heavy[0].GetSymbol() in ("P", "S"):
                anion = True
        flags.append(anion)
    return np.array(flags, bool)


def _embed(mol: Chem.Mol, pin_idx: list[int], pin_xyz: np.ndarray, n_confs: int, seed: int, n_threads: int):
    """ETKDG with pinned atoms; cascade of fallbacks. Returns (mol_with_confs, method)."""
    cmap = {int(i): Point3D(*map(float, p)) for i, p in zip(pin_idx, pin_xyz)}
    attempts = (
        dict(),
        dict(useRandomCoords=True),
        dict(useRandomCoords=True, ignoreSmoothingFailures=True),
        dict(useRandomCoords=True, ignoreSmoothingFailures=True, useExpTorsionAnglePrefs=False,
             useBasicKnowledge=False),
    )
    for k, kw in enumerate(attempts):
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = seed + 17 * k
        ps.numThreads = n_threads
        ps.pruneRmsThresh = -1.0
        ps.SetCoordMap(cmap)
        for name, val in kw.items():
            setattr(ps, name, val)
        cids = list(rdDistGeom.EmbedMultipleConfs(m, n_confs, ps))
        if cids:
            return m, ("etkdg" if k == 0 else f"etkdg-fallback{k}")
    return None, "failed"


def _relax(mol: Chem.Mol, cid: int, pin_idx: list[int], max_iters: int = 600) -> float:
    """MMFF (UFF fallback) minimisation with the pinned atoms frozen; returns energy (kcal/mol)."""
    props = FFH.MMFFGetMoleculeProperties(mol)
    if props is not None:
        ff = FFH.MMFFGetMoleculeForceField(mol, props, confId=cid)
    else:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
    if ff is None:
        return 0.0
    for i in pin_idx:
        ff.AddFixedPoint(int(i))
    ff.Initialize()
    for _ in range(4):
        if ff.Minimize(maxIts=max_iters) == 0:
            break
    return float(ff.CalcEnergy())


def _components_without(mol: Chem.Mol, i: int, j: int) -> tuple[set[int], set[int]]:
    """Atom sets on the i-side and j-side after cutting the (acyclic) bond i-j."""
    def reach(start, banned):
        seen, stack = {start}, [start]
        while stack:
            a = stack.pop()
            for n in mol.GetAtomWithIdx(a).GetNeighbors():
                k = n.GetIdx()
                if k in seen or (a, k) in banned:
                    continue
                seen.add(k)
                stack.append(k)
        return seen
    banned = {(i, j), (j, i)}
    return reach(i, banned), reach(j, banned)


def _is_carbonyl_c(mol: Chem.Mol, a) -> bool:
    if a.GetSymbol() != "C":
        return False
    for n in a.GetNeighbors():
        if n.GetSymbol() in ("O", "S", "N"):
            b = mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx())
            if b.GetBondTypeAsDouble() == 2 and n.GetDegree() == 1:
                return True
    return False


def _rotors(mol: Chem.Mol, pin_set: set[int]) -> list[tuple[int, int, int, int, bool]]:
    """Rotatable bonds whose movable side holds no pinned atom: (a, b, c, d, sp3sp3)."""
    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        hi = [n.GetIdx() for n in ai.GetNeighbors() if n.GetIdx() != j and n.GetAtomicNum() > 1]
        hj = [n.GetIdx() for n in aj.GetNeighbors() if n.GetIdx() != i and n.GetAtomicNum() > 1]
        if not hi or not hj:
            continue                                   # terminal OH / NH2 / CH3: heavy atoms unaffected
        if (_is_carbonyl_c(mol, ai) and aj.GetSymbol() in ("N", "O", "S")) or \
           (_is_carbonyl_c(mol, aj) and ai.GetSymbol() in ("N", "O", "S")):
            continue                                   # amide / ester / thioester stay planar-trans
        side_i, side_j = _components_without(mol, i, j)
        # the two bond atoms sit on the rotation axis and never move, so pins there are harmless
        pin_i, pin_j = bool((side_i - {i}) & pin_set), bool((side_j - {j}) & pin_set)
        if pin_i and pin_j:
            continue
        if pin_i or (not pin_j and len(side_j) <= len(side_i)):
            bb, cc, a_, d_ = i, j, hi[0], hj[0]        # j side moves
        else:
            bb, cc, a_, d_ = j, i, hj[0], hi[0]
        sp3 = all(x.GetHybridization() == Chem.HybridizationType.SP3 for x in (ai, aj))
        out.append((a_, bb, cc, d_, sp3))
    return out


def _sample(mol, pin_idx, pin_xyz, n_confs, seed, n_threads, env=None, new_idx=None, priors=None, tors=None):
    """Torsion-space sampling around a pin-constrained seed. Returns [(coords (N,3), e_rel, method)].

    ETKDG with a coordinate map reproduces ~one geometry for the free part, so diversity is made
    here: random torsions on every rotor whose movable side holds no pinned atom, a cheap
    environment-clash pre-filter, then MMFF relaxation with the pins frozen.
    """
    from rdkit.Chem import rdMolTransforms as T
    rng = np.random.default_rng(seed)
    pin_set = set(int(i) for i in pin_idx)
    m, method = _embed(mol, list(pin_idx), pin_xyz, 3, seed, n_threads)
    if m is None:
        if not mol.GetNumConformers():
            raise BuildError("conformer generation failed (no embedding, no ideal coordinates)")
        m = Chem.Mol(mol)
        method = "ideal-align"
        keep0 = m.GetConformer(0)
        X = np.array(keep0.GetPositions())
        R, t = kabsch(X[list(pin_idx)], pin_xyz)
        for i, p in enumerate(X @ R.T + t):
            keep0.SetAtomPosition(i, Point3D(*p))
    seeds = []
    for conf in m.GetConformers():
        X = np.array(conf.GetPositions())
        R, t = kabsch(X[list(pin_idx)], pin_xyz)
        for i, p in enumerate(X @ R.T + t):
            conf.SetAtomPosition(i, Point3D(*p))
        for i, p in zip(pin_idx, pin_xyz):            # pins sit EXACTLY on the target coordinates
            conf.SetAtomPosition(int(i), Point3D(*map(float, p)))
        seeds.append(conf.GetId())
    priors = list(priors or [])
    nm_of = {a.GetIdx(): a.GetProp("name") for a in m.GetAtoms()}
    idx_of = {v: k for k, v in nm_of.items()}
    prior_bonds = {frozenset((b_, c_)) for (_a, b_, c_, _d, _mu, _sd) in priors}
    rotors = [r for r in _rotors(m, pin_set) if frozenset((r[1], r[2])) not in prior_bonds]
    heavy_new = [i for i in (new_idx if new_idx is not None else range(m.GetNumAtoms()))
                 if m.GetAtomWithIdx(i).GetAtomicNum() > 1]
    el_new = np.array([m.GetAtomWithIdx(i).GetSymbol().upper() for i in heavy_new])
    out = []
    n_try = 0
    max_try = n_confs * 6
    while len(out) < n_confs and n_try < max_try:
        n_try += 1
        base = m.GetConformer(seeds[n_try % len(seeds)])
        trial = Chem.Mol(m, confId=base.GetId()) if False else Chem.Mol(m)
        # work on a fresh single-conformer copy
        c = Chem.Conformer(base)
        trial.RemoveAllConformers()
        cid = trial.AddConformer(c, assignId=True)
        conf = trial.GetConformer(cid)
        for (a_, b_, c_, d_, sp3) in rotors:
            ang = None
            if tors:
                ent = tors.get((nm_of[b_], nm_of[c_]))
                if ent is not None and ent.get("p") is not None and all(n in idx_of for n in ent["names"]):
                    from .torsion_lib import sample_angle
                    ang = sample_angle(ent, rng)
                    if ang is not None:
                        i4 = [idx_of[n] for n in ent["names"]]
                        T.SetDihedralDeg(conf, i4[0], i4[1], i4[2], i4[3], float(ang))
                        continue
            ang = (rng.choice([60.0, 180.0, 300.0]) + rng.normal(0, 18.0)) if sp3 else rng.uniform(-180, 180)
            T.SetDihedralDeg(conf, a_, b_, c_, d_, float(ang))
        for (a_, b_, c_, d_, mu, sd) in priors:     # stereoelectronic prior (exo-anomeric effect) with 15% uniform
            ang = rng.normal(mu, sd) if rng.random() < 0.85 else rng.uniform(-180, 180)
            T.SetDihedralDeg(conf, a_, b_, c_, d_, float(ang))
        if env is not None and env.tree is not None and heavy_new:
            xyz = np.array(conf.GetPositions())[heavy_new]
            sc = score_conformer(xyz, el_new, None, env)
            if sc["clash"] > 4.0 and n_try < max_try - n_confs:        # hard-clash pre-filter
                continue
        e = _relax(trial, cid, list(pin_idx), max_iters=300)
        out.append((np.array(trial.GetConformer(cid).GetPositions()), e, "torsion-" + method))
    if not out:
        raise BuildError("torsion sampling produced no conformer")
    e0 = min(e for _, e, _ in out)
    return [(x, e - e0, mth) for x, e, mth in out]


def _select_diverse(cands: list[Candidate], new_sel: np.ndarray, top_k: int, min_rmsd: float) -> list[Candidate]:
    cands = sorted(cands, key=lambda c: c.score)
    if min_rmsd <= 0:                      # keep the whole pool (dataset building)
        return cands[:top_k]
    chosen: list[Candidate] = []
    for c in cands:
        if all(rmsd(c.xyz[new_sel], d.xyz[new_sel]) >= min_rmsd for d in chosen) if new_sel.any() else not chosen:
            chosen.append(c)
        if len(chosen) >= top_k:
            break
    return chosen


def _rank(coords_all, keep_idx, mol, pin_set, env, n_keep_top, min_rmsd, code="", priors=None, site=None,
          want_desc=False, tors=None):
    """Score conformers over the OUTPUT atoms (keep_idx) and select a diverse top-K."""
    el = np.array([mol.GetAtomWithIdx(i).GetSymbol().upper() for i in keep_idx])
    anion = _anionic_atoms(mol, keep_idx)
    qall = _physio_charges(mol, keep_idx, code)
    new_sel = np.array([i not in pin_set for i in keep_idx])
    cands = []
    for X, e, method in coords_all:
        xyz = X[keep_idx]
        terms = score_conformer(xyz[new_sel], el[new_sel], anion[new_sel], env, qall[new_sel])
        if want_desc:
            nm = {mol.GetAtomWithIdx(i).GetProp("name"): k for k, i in enumerate(keep_idx)}
            own_N = xyz[nm["N"]] if "N" in nm else None
            own_CB = xyz[nm["CB"]] if "CB" in nm else None
            terms.update(descriptors(xyz[new_sel], el[new_sel], qall[new_sel], anion[new_sel], env, own_N, own_CB, site))
            if WANT_EXT or _ranker_needs_ext():
                terms.update(ext_descriptors(xyz[new_sel], anion[new_sel], env))
        if tors:
            from .torsion_lib import neg_logp, torsion_deg
            nmx = {mol.GetAtomWithIdx(i).GetProp("name"): k for k, i in enumerate(keep_idx)}
            tl, nt = 0.0, 0
            for ent in tors.values():
                if all(n in nmx for n in ent["names"]):
                    ang = torsion_deg(*[xyz[nmx[n]] for n in ent["names"]])
                    if nt < 3:                           # raw torsion (sin/cos) of the first rotors as ranker features
                        terms[f"t{nt}_sin"] = float(np.sin(np.radians(ang)))
                        terms[f"t{nt}_cos"] = float(np.cos(np.radians(ang)))
                    if ent.get("p") is not None:
                        tl += neg_logp(ent, ang)
                    nt += 1
            terms["tlib"] = float(tl)
            terms["tlib_n"] = float(nt)
        if priors:
            terms["tors"] = float(sum(min(((_wrap(_dihedral(X[a_], X[b_], X[c_], X[d_]) - mu)) / sd) ** 2, 9.0)
                                      for (a_, b_, c_, d_, mu, sd) in priors))
        cands.append(Candidate(xyz=xyz, e_rel=e, terms=terms, score=total_score(terms, e), method=method))
    if USE_RANKER and want_desc and code:
        from . import ranker as _rk
        rk = _rk.load()
        if rk is not None and rk.knows(code):
            F = np.array([[(c.e_rel if k == "e_rel" else c.score if k == "base_score" else c.terms.get(k, 0.0))
                           for k in rk.keys] for c in cands], float)
            pred = rk.predict(rk.features(F, code))
            for c, v in zip(cands, pred):
                c.terms["heur_score"] = c.score
                c.score = float(-v)
    return _select_diverse(cands, new_sel, n_keep_top, min_rmsd)


def _mcs_name_map(parent_code: str, mod_code: str) -> dict[str, str]:
    """parent atom name -> modified-comp atom name by name match, then MCS for renamed atoms."""
    pm = Chem.RemoveHs(Chem.Mol(ccd.comp_mol(parent_code)))
    mm = Chem.RemoveHs(Chem.Mol(ccd.comp_mol(mod_code)))
    pn, mn = ccd.names(pm), ccd.names(mm)
    direct = {n: n for n in pn if n in mn
              and pm.GetAtomWithIdx(pn.index(n)).GetSymbol() == mm.GetAtomWithIdx(mn.index(n)).GetSymbol()}
    if len(direct) >= 0.8 * pm.GetNumAtoms():
        return direct
    try:
        res = rdFMCS.FindMCS([pm, mm], timeout=5, ringMatchesRingOnly=True, completeRingsOnly=False,
                             atomCompare=rdFMCS.AtomCompare.CompareElements,
                             bondCompare=rdFMCS.BondCompare.CompareAny)
        patt = Chem.MolFromSmarts(res.smartsString)
        pa, ma = pm.GetSubstructMatch(patt), mm.GetSubstructMatch(patt)
        mapped = {pn[i]: mn[j] for i, j in zip(pa, ma)}
    except Exception:
        mapped = {}
    mapped.update(direct)        # name match wins
    # injective?
    seen, out = set(), {}
    for k, v in mapped.items():
        if v not in seen:
            out[k] = v
            seen.add(v)
    return out


def _environment(arr: struc.AtomArray, exclude: np.ndarray) -> Environment:
    m = ~exclude & ~np.isin(arr.element, ["H", "D"])
    return Environment(coord=np.asarray(arr.coord[m], float), element=np.char.upper(arr.element[m]),
                       atom_name=arr.atom_name[m], res_name=arr.res_name[m],
                       chain_id=arr.chain_id[m], res_id=arr.res_id[m])


def _heavy_bonds(mol: Chem.Mol, keep_idx: list[int]) -> list[tuple[int, int, int]]:
    """Heavy-atom bonds with KEKULE orders (aromatic 1.5 would truncate to 1 and lose the ring pattern)."""
    k = Chem.Mol(mol)
    try:
        Chem.Kekulize(k, clearAromaticFlags=True)
    except Exception:                                               # noqa: BLE001
        k = mol
    pos = {a: n for n, a in enumerate(keep_idx)}
    out = []
    for b in k.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in pos and j in pos:
            order = b.GetBondTypeAsDouble()
            out.append((pos[i], pos[j], 2 if order == 2.0 else 3 if order == 3.0 else 1))
    return out


def _formal_charges(mol: Chem.Mol, keep_idx: list[int]) -> list[int]:
    return [int(mol.GetAtomWithIdx(i).GetFormalCharge()) for i in keep_idx]


def _check_single_conformer(res: struc.AtomArray, key: ResKey) -> None:
    names = list(res.atom_name)
    if len(set(names)) != len(names):
        raise BuildError(f"{key}: duplicate atom names (alternate locations?) - select one conformer first")


# --------------------------------------------------------------------------- replace mode
def build_replacement(
    arr: struc.AtomArray,
    key: ResKey,
    mod_code: str,
    n_confs: int = 300,
    top_k: int = 5,
    seed: int = 0,
    n_threads: int = 1,
    min_rmsd: float = 0.6,
) -> Built:
    mod_code = mod_code.upper()
    rmask = residue_mask(arr, key)
    if not rmask.any():
        raise BuildError(f"residue {key} not found")
    res = arr[rmask]
    _check_single_conformer(res, key)
    parent_in_struct = str(res.res_name[0]).upper()
    parent = ccd.parent_of(mod_code)
    if parent_in_struct == mod_code:
        raise BuildError(f"{key} is already {mod_code}")
    if parent is not None and parent_in_struct != parent:
        # allow chaining only if the structure already holds another modification of the same parent
        if not (ccd.has_comp(parent_in_struct) and ccd.parent_of(parent_in_struct) == parent):
            raise BuildError(f"{key} is {parent_in_struct}; {mod_code} is a modification of {parent}")
    mol = Chem.Mol(ccd.comp_mol(mod_code))
    names = ccd.names(mol)
    nidx = {n: i for i, n in enumerate(names)}

    # which target atoms pin which mod atoms
    nm = _mcs_name_map(parent_in_struct, mod_code) if ccd.has_comp(parent_in_struct) else {}
    pin_idx, pin_xyz, pinned = [], [], []
    for a_name, xyz in zip(res.atom_name, res.coord):
        tgt = nm.get(a_name, a_name)
        if tgt in nidx and mol.GetAtomWithIdx(nidx[tgt]).GetAtomicNum() > 1:
            if nidx[tgt] not in pin_idx:
                pin_idx.append(nidx[tgt])
                pin_xyz.append(xyz)
                pinned.append(tgt)
    if len(pin_idx) < 4:
        raise BuildError(f"{key}: only {len(pin_idx)} atoms map onto {mod_code}; need >= 4 "
                         f"(have {list(res.atom_name)})")
    pin_xyz = np.array(pin_xyz, float)

    # output atoms: heavy and not flagged leaving (unless the structure itself has that atom, e.g. OXT)
    have = set(res.atom_name)
    keep_idx = [i for i, a in enumerate(mol.GetAtoms())
                if a.GetAtomicNum() > 1 and (not a.GetIntProp("leave") or names[i] in have)]
    pin_set = set(pin_idx)
    env = _environment(arr, rmask)
    new_idx = [i for i in range(mol.GetNumAtoms()) if i not in pin_set]
    tors = None
    if TORS_LIB and mod_code in TORS_LIB:
        tors = {(e["names"][1], e["names"][2]): e for e in TORS_LIB[mod_code]}
    coords_all = _sample(mol, pin_idx, pin_xyz, n_confs, seed, n_threads, env=env, new_idx=new_idx, tors=tors)
    cands = _rank(coords_all, keep_idx, mol, pin_set, env, top_k, min_rmsd, mod_code, site=(key.chain, key.res_id),
                  want_desc=(WANT_DESC or USE_RANKER), tors=tors)
    # exact pinned coordinates (they were fixed, but enforce numerically)
    for c in cands:
        for k_out, i in enumerate(keep_idx):
            if i in pin_set:
                c.xyz[k_out] = pin_xyz[pin_idx.index(i)]
    return Built(
        kind="replace", key=key, mod_code=mod_code, parent=parent_in_struct,
        atom_names=[names[i] for i in keep_idx],
        elements=[mol.GetAtomWithIdx(i).GetSymbol() for i in keep_idx],
        new_mask=np.array([i not in pin_set for i in keep_idx]),
        bonds=_heavy_bonds(mol, keep_idx), candidates=cands, pinned=pinned,
        charges=_formal_charges(mol, keep_idx),
    )


# --------------------------------------------------------------------------- adduct mode
@dataclass(frozen=True)
class AdductSpec:
    ligand: str                 # CCD code of the covalently attached group
    anchor_atom: str            # atom of the anchor residue that forms the bond
    ligand_atom: str            # atom of the ligand that forms the bond
    leaving: tuple[str, ...] = ()   # ligand atoms lost on bond formation (e.g. anomeric O1)
    order: int = 1


def build_adduct(
    arr: struc.AtomArray,
    key: ResKey,
    spec: AdductSpec,
    n_confs: int = 300,
    top_k: int = 5,
    seed: int = 0,
    n_threads: int = 1,
    min_rmsd: float = 0.8,
    ligand_chain: str | None = None,
    ligand_res_id: int | None = None,
) -> Built:
    rmask = residue_mask(arr, key)
    if not rmask.any():
        raise BuildError(f"residue {key} not found")
    res = arr[rmask]
    _check_single_conformer(res, key)
    if ligand_chain is not None and ligand_res_id is not None:
        if residue_mask(arr, ResKey(ligand_chain, int(ligand_res_id))).any():
            raise BuildError(f"ligand identity {ligand_chain}{ligand_res_id} already exists in the structure")
    anchor_name = str(res.res_name[0]).upper()
    if not ccd.has_comp(anchor_name):
        raise BuildError(f"anchor residue type {anchor_name} not in CCD subset")
    if spec.anchor_atom not in set(res.atom_name):
        raise BuildError(f"{key} ({anchor_name}) has no atom {spec.anchor_atom}")

    parent = Chem.Mol(ccd.comp_mol(anchor_name, drop_leaving=True))
    lig = Chem.Mol(ccd.comp_mol(spec.ligand, drop_atoms=tuple(spec.leaving)))
    pn, ln = ccd.names(parent), ccd.names(lig)
    if spec.ligand_atom not in ln:
        raise BuildError(f"ligand {spec.ligand} has no atom {spec.ligand_atom}")
    a_i = pn.index(spec.anchor_atom)
    l_i = ln.index(spec.ligand_atom)
    # position of the leaving group in the ligand's ideal frame (used to set stereo of the attach atom)
    full_lig = ccd.comp_mol(spec.ligand)
    full_names = ccd.names(full_lig)
    leave_xyz = None
    if spec.leaving:
        leave_xyz = np.array(full_lig.GetConformer(0).GetPositions())[full_names.index(spec.leaving[0])]

    pxyz = np.array(parent.GetConformer(0).GetPositions())
    lxyz = np.array(lig.GetConformer(0).GetPositions())
    shift = (leave_xyz if leave_xyz is not None else lxyz[l_i] + np.array([1.4, 0, 0])) - pxyz[a_i]
    comb = Chem.CombineMols(parent, lig)
    n_par = parent.GetNumAtoms()
    rw = Chem.RWMol(comb)
    rw.AddBond(a_i, n_par + l_i, {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE}[spec.order])
    # virtual conformer: ligand at its ideal pose, parent translated so the anchor sits where the
    # leaving group was -> stereo tags (anomeric C1 etc.) are inherited from the CCD geometry
    conf = Chem.Conformer(rw.GetNumAtoms())
    for i, p in enumerate(np.vstack([pxyz + shift, lxyz])):
        conf.SetAtomPosition(i, Point3D(*p))
    rw.RemoveAllConformers()
    rw.AddConformer(conf, assignId=True)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.AssignStereochemistryFrom3D(mol)
    # an H on each end of the new bond is displaced when that end would otherwise be over-valent
    drop = []
    for end in (a_i, n_par + l_i):
        at = mol.GetAtomWithIdx(end)
        allowed = max(Chem.GetPeriodicTable().GetValenceList(at.GetAtomicNum())) + (
            1 if (at.GetSymbol() == "N" and at.GetFormalCharge() == 1) else 0)
        if at.GetExplicitValence() > allowed:
            hs = [n.GetIdx() for n in at.GetNeighbors() if n.GetAtomicNum() == 1 and n.GetIdx() not in drop]
            if not hs:
                raise BuildError(f"{spec.ligand}@{spec.anchor_atom}: over-valent attach atom without H to displace")
            drop.append(hs[0])
    rwm = Chem.RWMol(mol)
    for idx in sorted(drop, reverse=True):
        rwm.RemoveAtom(idx)
    mol = rwm.GetMol()
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    mol.UpdatePropertyCache(strict=False)
    mol.RemoveAllConformers()                      # embed from scratch; chirality tags persist
    names_all = ccd.names(Chem.Mol(parent)) + ccd.names(Chem.Mol(lig))
    src = ["P"] * n_par + ["L"] * lig.GetNumAtoms()
    keep_mask = [k not in set(drop) for k in range(len(names_all))]
    names_all = [n for n, k in zip(names_all, keep_mask) if k]
    src = [t for t, k in zip(src, keep_mask) if k]

    # pins: anchor-residue heavy atoms present in the structure (leaving atoms of the parent excluded)
    nidx = {n: i for i, n in enumerate(names_all) if src[i] == "P"}
    pin_idx, pin_xyz, pinned = [], [], []
    for a_name, xyz in zip(res.atom_name, res.coord):
        if a_name in nidx and mol.GetAtomWithIdx(nidx[a_name]).GetAtomicNum() > 1:
            pin_idx.append(nidx[a_name]); pin_xyz.append(xyz); pinned.append(a_name)
    if len(pin_idx) < 4:
        raise BuildError(f"{key}: too few anchor-residue atoms to pin ({len(pin_idx)})")
    pin_xyz = np.array(pin_xyz, float)
    pin_set = set(pin_idx)

    keep_idx = [i for i in range(mol.GetNumAtoms()) if src[i] == "L" and mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    for i, nm in enumerate(names_all):               # tag atoms with names for downstream helpers
        mol.GetAtomWithIdx(i).SetProp("name", nm)
    env = _environment(arr, rmask)
    new_idx = [i for i in range(mol.GetNumAtoms()) if i not in pin_set]
    priors = []
    mu_phi = _anomeric_prior(spec.ligand)
    if mu_phi is not None:
        anc = next(i for i, nm in enumerate(names_all) if src[i] == "P" and nm == spec.anchor_atom)
        lat = next(i for i, nm in enumerate(names_all) if src[i] == "L" and nm == spec.ligand_atom)
        xs = [n.GetIdx() for n in mol.GetAtomWithIdx(anc).GetNeighbors()
              if n.GetAtomicNum() > 1 and n.GetIdx() != lat and src[n.GetIdx()] == "P"]
        ro = [n.GetIdx() for n in mol.GetAtomWithIdx(lat).GetNeighbors() if n.GetSymbol() == "O" and n.IsInRing()]
        if xs and ro:
            priors = [(xs[0], anc, lat, ro[0], mu_phi, PHI_SIGMA)]
    coords_all = _sample(mol, pin_idx, pin_xyz, n_confs, seed, n_threads, env=env, new_idx=new_idx, priors=priors)
    cands = _rank(coords_all, keep_idx, mol, pin_set, env, top_k, min_rmsd, spec.ligand, priors=priors)
    lig_names = [names_all[i] for i in keep_idx]
    new_key = ResKey(ligand_chain or next_free_chain_id(arr), ligand_res_id or 1, "")
    return Built(
        kind="adduct", key=key, mod_code=spec.ligand, parent=anchor_name,
        atom_names=lig_names,
        elements=[mol.GetAtomWithIdx(i).GetSymbol() for i in keep_idx],
        new_mask=np.ones(len(keep_idx), bool),
        bonds=_heavy_bonds(mol, keep_idx), candidates=cands, pinned=pinned,
        charges=_formal_charges(mol, keep_idx),
        anchor_atom=spec.anchor_atom, ligand_atom=spec.ligand_atom, ligand_key=new_key,
    )


# --------------------------------------------------------------------------- splice into structure
def apply_built(arr: struc.AtomArray, built: Built, k: int = 0) -> struc.AtomArray:
    """Return a new AtomArray with candidate k of `built` spliced in (bond list maintained)."""
    cand = built.candidates[min(k, len(built.candidates) - 1)]
    n = len(built.atom_names)
    cats = arr.get_annotation_categories()
    if not {"b_factor", "occupancy", "charge"} <= set(cats):         # keep every annotation through concatenate()
        arr = arr.copy()
        for cat, dt, dv in (("b_factor", float, 0.0), ("occupancy", float, 1.0), ("charge", int, 0)):
            if cat not in cats:
                arr.add_annotation(cat, dt)
                arr.set_annotation(cat, np.full(arr.array_length(), dv, dtype=dt))
    new = new_atom_array(n)
    _carry = {}
    if built.kind == "replace":
        _old = arr[residue_mask(arr, built.key)]
        _carry = {c: _old.get_annotation(c)[0] for c in arr.get_annotation_categories()}
    for cat in arr.get_annotation_categories():           # annotations the builder does not know (provenance, ...)
        if cat not in new.get_annotation_categories():
            dt = arr.get_annotation(cat).dtype
            new.add_annotation(cat, dt)
            fill = _carry.get(cat, "" if dt.kind in "US" else 0)
            new.set_annotation(cat, np.full(n, fill, dtype=dt))
    new.coord = cand.xyz.astype(np.float32)
    new.atom_name = np.array(built.atom_names)
    new.element = np.array(built.elements)
    new.hetero = np.ones(n, bool)
    if len(built.charges) == n:
        new.charge = np.array(built.charges, dtype=int)
    bf = float(np.mean(arr.b_factor)) if "b_factor" in arr.get_annotation_categories() else 0.0

    if built.kind == "replace":
        rmask = residue_mask(arr, built.key)
        old = arr[rmask]
        for cat in ("chain_id", "res_id", "ins_code"):
            getattr(new, cat)[:] = getattr(old, cat)[0]
        new.res_name[:] = built.mod_code
        new.b_factor[:] = float(np.mean(old.b_factor)) if "b_factor" in old.get_annotation_categories() else bf
        idx_old = np.where(rmask)[0]
        start, stop = idx_old[0], idx_old[-1] + 1
        if stop - start != len(idx_old):
            raise BuildError("residue atoms not contiguous in the input array")
        out = struc.concatenate([arr[:start], new, arr[stop:]])
        # bonds: keep old bonds not touching the residue; re-map bonds touching it by atom name
        old_b = arr.bonds.as_array()
        bl = struc.BondList(out.array_length())
        shift_map = np.arange(arr.array_length())
        shift_map[stop:] += n - (stop - start)
        old_name_to_new = {nm: start + i for i, nm in enumerate(built.atom_names)}
        for i, j, t in old_b:
            ii, jj = int(i), int(j)
            ini, inj = start <= ii < stop, start <= jj < stop
            if ini and inj:
                continue
            if ini:
                nn = old_name_to_new.get(arr.atom_name[ii])
                if nn is None: continue
                bl.add_bond(nn, int(shift_map[jj]), int(t))
            elif inj:
                nn = old_name_to_new.get(arr.atom_name[jj])
                if nn is None: continue
                bl.add_bond(int(shift_map[ii]), nn, int(t))
            else:
                bl.add_bond(int(shift_map[ii]), int(shift_map[jj]), int(t))
        for i, j, o in built.bonds:
            bl.add_bond(start + i, start + j, struc.BondType.SINGLE if o == 1 else struc.BondType.DOUBLE if o == 2 else struc.BondType.TRIPLE)
        out.bonds = bl
        return out

    # adduct: append ligand residue as its own chain, covalent bond to the anchor atom
    # (explicit-H input: the H the new bond displaces must go, otherwise the anchor is over-valent)
    if np.isin(arr.element, ["H", "D"]).any():
        hm = residue_mask(arr, built.key) & np.isin(arr.element, ["H", "D"])
        am = residue_mask(arr, built.key) & (arr.atom_name == built.anchor_atom)
        if hm.any() and am.sum() == 1:
            axyz = arr.coord[am][0]
            lxyz = cand.xyz[built.atom_names.index(built.ligand_atom)]
            hidx = np.where(hm & (np.linalg.norm(arr.coord - axyz, axis=1) < 1.6))[0]   # S-H is 1.34 A
            if len(hidx):
                v = (lxyz - axyz) / (np.linalg.norm(lxyz - axyz) + 1e-9)
                drop = hidx[int(np.argmax([np.dot((arr.coord[h] - axyz), v) for h in hidx]))]
                keep = np.ones(arr.array_length(), bool)
                keep[drop] = False
                arr = arr[keep]
    lk = built.ligand_key
    new.chain_id[:] = lk.chain
    new.res_id[:] = lk.res_id
    new.ins_code[:] = ""
    new.res_name[:] = built.mod_code
    new.b_factor[:] = bf
    out = struc.concatenate([arr, new])
    bl = struc.BondList(out.array_length())
    for i, j, t in arr.bonds.as_array():
        bl.add_bond(int(i), int(j), int(t))
    base = arr.array_length()
    for i, j, o in built.bonds:
        bl.add_bond(base + i, base + j, struc.BondType.SINGLE if o == 1 else struc.BondType.DOUBLE if o == 2 else struc.BondType.TRIPLE)
    amask = residue_mask(arr, built.key) & (arr.atom_name == built.anchor_atom)
    if amask.sum() != 1:
        raise BuildError(f"anchor atom {built.anchor_atom} not unique in {built.key}")
    bl.add_bond(int(np.where(amask)[0][0]), base + built.atom_names.index(built.ligand_atom), struc.BondType.SINGLE)
    out.bonds = bl
    return out


# --------------------------------------------------------------------------- helpers: revert / remove
def _template_bonds(code: str, names: list[str]) -> list[tuple[int, int, int]]:
    """Intra-residue heavy bonds of CCD component `code` among the given atom names (indices into names)."""
    pos = {n: i for i, n in enumerate(names)}
    out = []
    for n1, n2, order, _a in ccd.get_comp(code)["bonds"]:
        if n1 in pos and n2 in pos:
            out.append((pos[n1], pos[n2], {"DOUB": 2, "TRIP": 3}.get(order, 1)))
    return out


def revert_residue(arr: struc.AtomArray, key: ResKey) -> struc.AtomArray:
    """Turn a modified residue back into its parent: atoms the parent also has are kept (renamed to the
    parent's names when the CCD renames them, e.g. CIR C3..N8 -> ARG CB..NH), all others are dropped, and the
    residue's internal bonds are re-derived from the PARENT template (a ring such as PCA must not survive).

    Used for (a) unmodified controls of a modified target and (b) native-recovery benchmarks.
    """
    rmask = residue_mask(arr, key)
    res = arr[rmask]
    code = str(res.res_name[0]).upper()
    parent = ccd.parent_of(code)
    if parent is None:
        raise BuildError(f"{code} has no parent residue in CCD")
    inv = {m: p for p, m in _mcs_name_map(parent, code).items()}
    parent_heavy = set(ccd.heavy_names(parent))
    keep_local, new_names = [], []
    for k, n in enumerate(res.atom_name):
        pn = inv.get(n, n if n in parent_heavy and n not in inv.values() else None)
        if n == "OXT":
            pn = "OXT"
        if pn is not None and pn not in new_names:
            keep_local.append(k)
            new_names.append(pn)
    keep = np.ones(arr.array_length(), bool)
    idx_res = np.where(rmask)[0]
    drop = np.ones(len(idx_res), bool)
    drop[keep_local] = False
    keep[idx_res[drop]] = False
    out = arr[keep]
    om = residue_mask(out, key)
    oi = np.where(om)[0]
    # names follow the original atom order of the kept atoms
    out.atom_name[oi] = np.array(new_names)
    out.res_name[om] = parent
    out.hetero[om] = False
    # rebuild intra-residue bonds from the parent template, keep inter-residue bonds
    if out.bonds is not None:
        old = out.bonds.as_array()
        bl = struc.BondList(out.array_length())
        ois = set(int(i) for i in oi)
        for i, j, t in old:
            i, j = int(i), int(j)
            if i in ois and j in ois:
                continue
            bl.add_bond(i, j, int(t))
        for i, j, o in _template_bonds(parent, new_names):
            bl.add_bond(int(oi[i]), int(oi[j]), {1: struc.BondType.SINGLE, 2: struc.BondType.DOUBLE,
                                                  3: struc.BondType.TRIPLE}[o])
        out.bonds = bl
    return out


def symmetry_perms(code: str, atom_names: list[str], fixed_names: set[str], max_perms: int = 512) -> list[np.ndarray]:
    """Graph automorphisms of component `code` restricted to `atom_names`, as index permutations that
    leave `fixed_names` (pinned atoms) in place. Used for symmetry-aware RMSD.

    Bond ORDERS are ignored on purpose: the three terminal oxygens of a phosphate/sulfate (and the two of a
    carboxylate/carbamate) are chemically equivalent at physiological pH although the CCD draws one P=O."""
    mol = Chem.RemoveHs(Chem.Mol(ccd.comp_mol(code)))
    nm = ccd.names(mol)
    pos = {n: i for i, n in enumerate(atom_names)}
    sel = [i for i, n in enumerate(nm) if n in pos]
    sub = Chem.RWMol(mol)
    for i in sorted(set(range(mol.GetNumAtoms())) - set(sel), reverse=True):
        sub.RemoveAtom(i)
    sub = sub.GetMol()
    flat = Chem.RWMol(sub)                               # element graph only: all bonds single, no aromaticity
    for bd in flat.GetBonds():
        bd.SetBondType(Chem.BondType.SINGLE)
        bd.SetIsAromatic(False)
    for at in flat.GetAtoms():
        at.SetIsAromatic(False)
        at.SetFormalCharge(0)
    flat = flat.GetMol()
    sub_names = [nm[i] for i in sel]
    perms = []
    for m in flat.GetSubstructMatches(flat, uniquify=False, useChirality=False, maxMatches=max_perms):
        perm = np.arange(len(atom_names))
        ok = True
        for q, t in enumerate(m):
            a, bb = sub_names[q], sub_names[t]
            if a in fixed_names and a != bb:
                ok = False
                break
            perm[pos[a]] = pos[bb]
        if ok:
            perms.append(perm)
    return perms or [np.arange(len(atom_names))]


def sym_rmsd(xyz: np.ndarray, ref_xyz: np.ndarray, idx: list[int], perms: list[np.ndarray]) -> float:
    """min over symmetry perms of RMSD between xyz[perm][idx] and ref_xyz[idx]."""
    best = np.inf
    for p in perms:
        d = xyz[p][idx] - ref_xyz[idx]
        best = min(best, float(np.sqrt((d ** 2).sum(-1).mean())))
    return best
