"""Score AF3 co-folds of binder + PTM target: interface confidence, PTM engagement, design agreement.

CPU side (biotite + numpy). Reads the tar written by scripts/af3/af3_batch.py.
Output-format assumptions (AF3 / AlphaFast): `<name>_summary_confidences.json` holds iptm, ptm,
ranking_score, chain_pair_iptm (chain order = input order), has_clash, fraction_disordered;
`<name>_confidences.json` holds atom_plddts, atom_chain_ids, token_chain_ids, token_res_ids, pae.
"""
from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass

import numpy as np
from biotite import structure as struc
from biotite.structure.io import pdbx

POLAR = {"N", "O"}


@dataclass
class PTMAtomRef:
    chain: str       # AF3 chain id
    pos: int         # 1-based residue/component position in that AF3 chain
    atom: str


def read_tar(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if not m.isfile() or "/" not in m.name:
                continue
            nm, fn = m.name.split("/", 1)
            d = out.setdefault(nm, {})
            data = tf.extractfile(m).read()
            if fn.endswith("_model.cif"):
                d["cif"] = data.decode()
            elif fn.endswith("_summary_confidences.json"):
                d["summary"] = json.loads(data)
            elif fn.endswith("_confidences.json"):
                d["conf"] = json.loads(data)
    return out


def read_model(cif_text: str) -> struc.AtomArray:
    f = pdbx.CIFFile.read(io.StringIO(cif_text))
    arr = pdbx.get_structure(f, model=1, extra_fields=["b_factor"], use_author_fields=False)
    return arr[arr.element != "H"]


def _chain_order(arr: struc.AtomArray) -> list[str]:
    seen, order = set(), []
    for c in arr.chain_id:
        if c not in seen:
            seen.add(c)
            order.append(str(c))
    return order


def iptm_binder_target(summary: dict, chain_order: list[str], binder: str, targets: list[str]) -> float:
    cp = np.asarray(summary.get("chain_pair_iptm", []), float)
    if cp.ndim != 2 or binder not in chain_order:
        return float("nan")
    b = chain_order.index(binder)
    vals = []
    for t in targets:
        if t in chain_order:
            j = chain_order.index(t)
            vals += [cp[b, j], cp[j, b]]
    return float(np.nanmax(vals)) if vals else float("nan")


def ptm_engagement(arr: struc.AtomArray, binder: str, ptm_atoms: list[PTMAtomRef], cutoff: float = 4.0,
                   res_cutoff: float = 4.5) -> dict:
    """How strongly the binder engages the modification atoms in a predicted model."""
    bmask = (arr.chain_id == binder)
    bx = arr.coord[bmask]
    out = {"ptm_min_dist": float("nan"), "ptm_atoms_contacted": 0, "ptm_atoms_total": len(ptm_atoms),
           "ptm_frac_contacted": 0.0, "ptm_binder_residues": 0, "ptm_polar_pairs": 0, "ptm_atom_contacts": 0,
           "ptm_n_sites": len({(r.chain, r.pos) for r in ptm_atoms}), "ptm_site_frac_max": 0.0}
    if not len(bx) or not ptm_atoms:
        return out
    res_hit = set()
    dmin_all, n_contacted, polar, n_pairs = [], 0, 0, 0
    site_tot: dict[tuple, int] = {}                                      # required atoms per site: counted BEFORE the lookup, a missing atom is not a contact
    for ref in ptm_atoms:
        site_tot[(ref.chain, ref.pos)] = site_tot.get((ref.chain, ref.pos), 0) + 1
    site_hit: dict[tuple, int] = {k: 0 for k in site_tot}
    b_el = arr.element[bmask]
    b_res = arr.res_id[bmask]
    for ref in ptm_atoms:
        m = (arr.chain_id == ref.chain) & (arr.res_id == ref.pos) & (arr.atom_name == ref.atom)
        if not m.any():
            continue
        x = arr.coord[m][0]
        d = np.linalg.norm(bx - x, axis=1)
        dmin_all.append(float(d.min()))
        close = d <= cutoff
        n_pairs += int(close.sum())
        n_contacted += int(close.any())
        site_hit[(ref.chain, ref.pos)] += int(close.any())              # one site = one modified residue / one attached group (own chain)
        res_hit.update(int(r) for r in b_res[d <= res_cutoff])
        el = str(arr.element[m][0]).upper()
        if el in POLAR:
            polar += int(((d <= 3.4) & np.isin(np.char.upper(b_el), list(POLAR))).sum())
    if dmin_all:
        out.update(ptm_min_dist=min(dmin_all), ptm_atoms_contacted=n_contacted,
                   ptm_frac_contacted=n_contacted / max(len(ptm_atoms), 1), ptm_binder_residues=len(res_hit),
                   ptm_polar_pairs=polar, ptm_atom_contacts=n_pairs,
                   ptm_site_frac_max=max(site_hit[k] / site_tot[k] for k in site_tot))
    return out


def _kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - R @ Pc


def design_agreement(pred: struc.AtomArray, design: struc.AtomArray, binder_pred: str, binder_design: str,
                     target_map: dict[str, str]) -> dict:
    """Binder CA RMSD to the RFD3 design: fold (binder-superposed) and pose (target-superposed)."""
    def ca(a, c):
        m = (a.chain_id == c) & (a.atom_name == "CA")
        return a.coord[m]
    pb, db = ca(pred, binder_pred), ca(design, binder_design)
    out = {"binder_rmsd_fold": float("nan"), "binder_rmsd_pose": float("nan"), "n_binder_ca": int(len(pb))}
    if len(pb) != len(db) or not len(pb):
        return out
    R, t = _kabsch(pb, db)
    out["binder_rmsd_fold"] = float(np.sqrt((((pb @ R.T + t) - db) ** 2).sum(-1).mean()))
    pt = [ca(pred, c) for c in target_map]
    dt = [ca(design, d) for d in target_map.values()]
    if all(len(a) == len(b) and len(a) for a, b in zip(pt, dt)):
        PT, DT = np.vstack(pt), np.vstack(dt)
        R, t = _kabsch(PT, DT)
        out["binder_rmsd_pose"] = float(np.sqrt((((pb @ R.T + t) - db) ** 2).sum(-1).mean()))
    return out


def plddt_mean(conf: dict, chain: str) -> float:
    ch = np.asarray(conf.get("atom_chain_ids", []))
    pl = np.asarray(conf.get("atom_plddts", []), float)
    return float(pl[ch == chain].mean()) if len(ch) == len(pl) and (ch == chain).any() else float("nan")


def interface_pae(conf: dict, binder: str, targets: list[str]) -> float:
    """Mean PAE over (binder token, target token) pairs, both directions."""
    pae = np.asarray(conf.get("pae", []), float)
    tc = np.asarray(conf.get("token_chain_ids", []))
    if pae.ndim != 2 or len(tc) != pae.shape[0]:
        return float("nan")
    b = tc == binder
    t = np.isin(tc, targets)
    if not b.any() or not t.any():
        return float("nan")
    return float(0.5 * (pae[np.ix_(b, t)].mean() + pae[np.ix_(t, b)].mean()))


def tm_residue_weighted(conf: dict, binder: str, targets: list[str], n_ref: int | None = None,
                        exclude: tuple[str, int] | None = None) -> float:
    """Interface TM-score-like confidence with ONE weight per residue (comparable between a modified and an unmodified target).

    AF3 tokenises a modified residue per atom (phospho-Tyr = 16 tokens, Tyr = 1), so its ipTM gives the modification up to 16x
    the weight of an ordinary residue and uses a different length normalisation. Here the tokens of each (chain, residue) are
    averaged first, d0 uses `n_ref` residues (default: residues of binder + targets, identical for both arms), and the pair
    score 1/(1+(PAE/d0)^2) is averaged over residues, max over the aligned residue, best of the two directions.
    `exclude=(chain, res_id)` leaves one residue (e.g. the modified one) out of the average."""
    pae = np.asarray(conf.get("pae", []), float)
    tc = np.asarray(conf.get("token_chain_ids", []))
    tr = np.asarray(conf.get("token_res_ids", []))
    if pae.ndim != 2 or len(tc) != pae.shape[0] or len(tr) != len(tc):
        return float("nan")
    b_idx = np.where(tc == binder)[0]
    groups: dict[tuple[str, int], np.ndarray] = {}
    for ch in targets:
        for r in sorted(set(tr[tc == ch].tolist())):
            if exclude is not None and (ch, int(r)) == (exclude[0], int(exclude[1])):
                continue
            groups[(ch, int(r))] = np.where((tc == ch) & (tr == r))[0]
    if not len(b_idx) or not groups:
        return float("nan")
    n_res = n_ref or (len(set(tr[b_idx].tolist())) + sum(len(set(tr[tc == ch].tolist())) for ch in targets))
    d0 = 1.24 * (max(n_res, 19) - 15) ** (1.0 / 3.0) - 1.8
    f = lambda x: 1.0 / (1.0 + (x / d0) ** 2)
    bt = np.stack([pae[np.ix_(b_idx, ix)].mean(1) for ix in groups.values()], 1)          # binder token x target residue
    tb = np.stack([pae[np.ix_(ix, b_idx)].mean(0) for ix in groups.values()], 0)          # target residue x binder token
    return float(max(f(bt).mean(1).max(), f(tb).mean(1).max()))


def score_entry(entry: dict, binder: str, targets: list[str], ptm_atoms: list[PTMAtomRef],
                design: struc.AtomArray | None = None, design_binder: str = "A",
                design_target_map: dict[str, str] | None = None) -> dict:
    arr = read_model(entry["cif"])
    order = _chain_order(arr)
    s, c = entry["summary"], entry["conf"]
    res = {
        "iptm": float(s.get("iptm", np.nan)), "ptm": float(s.get("ptm", np.nan)),
        "ranking_score": float(s.get("ranking_score", np.nan)),
        "iptm_binder_target": iptm_binder_target(s, order, binder, targets),
        "has_clash": bool(s.get("has_clash", False)), "fraction_disordered": float(s.get("fraction_disordered", np.nan)),
        "plddt_binder": plddt_mean(c, binder), "ipae_binder_target": interface_pae(c, binder, targets),
        "tm_res": tm_residue_weighted(c, binder, targets),
    }
    res.update(ptm_engagement(arr, binder, ptm_atoms))
    if design is not None:
        res.update(design_agreement(arr, design, binder, design_binder, design_target_map or {}))
    return res
