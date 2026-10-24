"""High level: PTM specs + target structure -> modified target ensemble, unmodified control, report."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from biotite import structure as struc

from . import ccd
from . import structio as sio
from .builder import (AdductSpec, Built, BuildError, apply_built, build_adduct, build_replacement,
                      revert_residue)
from .registry import parse_spec


@dataclass
class PTMRequest:
    key: sio.ResKey
    kind: str                         # 'replace' | 'adduct'
    payload: object                   # CCD code | AdductSpec
    parent: str | None = None
    text: str = ""


def parse_requests(specs: list[str]) -> list[PTMRequest]:
    out = []
    for t in specs:
        res, (kind, payload, parent) = parse_spec(t)
        out.append(PTMRequest(sio.ResKey.parse(res), kind, payload, parent, t))
    return out


def _is_resolved(arr: struc.AtomArray, req: PTMRequest) -> bool:
    """True if the structure already carries this modification (nothing to build)."""
    m = sio.residue_mask(arr, req.key)
    if not m.any():
        raise BuildError(f"residue {req.key} not found in structure")
    name = str(arr.res_name[m][0]).upper()
    if req.kind == "replace":
        if name != req.payload:
            return False
        # a residue NAME is not proof the modification is there: all heavy atoms of the CCD component
        # (minus leaving atoms) must be present, otherwise it is rebuilt
        need = {a["name"] for a in ccd.get_comp(name)["atoms"] if a["el"] != "H" and not a["leave"]}
        return need <= set(map(str, arr.atom_name[m]))
    # adduct: look for a covalently bonded ligand of that type on the anchor atom
    anchor = np.where(m & (arr.atom_name == req.payload.anchor_atom))[0]
    if len(anchor) != 1 or arr.bonds is None:
        return False
    nbrs, _ = arr.bonds.get_bonds(int(anchor[0]))
    return any(arr.res_name[j] == req.payload.ligand for j in nbrs)


@dataclass
class TargetBuild:
    unmodified: struc.AtomArray                 # control: every requested site as parent residue
    variants: list[struc.AtomArray]             # modified targets, variant k = rank-k conformer everywhere
    built: list[Built]
    resolved: list[PTMRequest]                  # PTMs that were already present in the input
    report: dict


def build_modified_target(
    arr: struc.AtomArray,
    requests: list[PTMRequest],
    n_confs: int = 300,
    top_k: int = 5,
    seed: int = 0,
    n_threads: int = 1,
) -> TargetBuild:
    cur = arr
    notes: list[str] = []
    built_list: list[Built] = []
    resolved: list[PTMRequest] = []
    todo: list[PTMRequest] = []
    for r in requests:
        if _is_resolved(cur, r):
            resolved.append(r)
        else:
            todo.append(r)

    # control = parent residues everywhere requested (also reverts resolved replace-type PTMs)
    ctrl = arr
    for r in requests:
        if r.kind == "replace":
            m = sio.residue_mask(ctrl, r.key)
            if str(ctrl.res_name[m][0]).upper() == r.payload:
                ctrl = revert_residue(ctrl, r.key)
        else:
            if _is_resolved(ctrl, r):
                ctrl = _strip_adduct(ctrl, r)

    for i, r in enumerate(todo):
        if r.kind == "replace":
            m_ = sio.residue_mask(cur, r.key)
            if str(cur.res_name[m_][0]).upper() == r.payload:      # named like the PTM but atoms missing
                cur = revert_residue(cur, r.key)
                notes.append(f"{r.text}: incomplete {r.payload} residue reverted to its parent atoms and rebuilt")
            b = build_replacement(cur, r.key, r.payload, n_confs=n_confs, top_k=top_k, seed=seed + i,
                                  n_threads=n_threads)
        else:
            b = build_adduct(cur, r.key, r.payload, n_confs=n_confs, top_k=top_k, seed=seed + i,
                             n_threads=n_threads)
        built_list.append(b)
        cur = apply_built(cur, b, 0)          # later PTMs see earlier ones as environment

    n_var = max([len(b.candidates) for b in built_list], default=1)
    variants = []
    for k in range(n_var):
        v = arr
        for n_, (b, r) in enumerate(zip(built_list, todo)):
            if n_ == 0 or len(built_list) == 1:
                v = apply_built(v, b, k)                    # variant k = rank-k conformer of the first PTM ...
            else:                                           # ... later PTMs are re-ranked IN that environment
                if r.kind == "replace":
                    m_ = sio.residue_mask(v, r.key)
                    if str(v.res_name[m_][0]).upper() == r.payload:
                        v = revert_residue(v, r.key)
                    b2 = build_replacement(v, r.key, r.payload, n_confs=n_confs, top_k=1, seed=seed + n_,
                                           n_threads=n_threads)
                else:
                    b2 = build_adduct(v, r.key, r.payload, n_confs=n_confs, top_k=1, seed=seed + n_,
                                      n_threads=n_threads)
                v = apply_built(v, b2, 0)
        variants.append(v)
    if not built_list:
        variants = [arr]

    report = {
        "requested": [r.text for r in requests],
        "already_resolved": [r.text for r in resolved],
        "notes": notes,
        "built": [
            {
                "residue": str(b.key), "kind": b.kind, "ccd": b.mod_code, "parent": b.parent,
                "new_atoms": [n for n, m in zip(b.atom_names, b.new_mask) if m],
                "pinned_atoms": b.pinned,
                "ligand_key": str(b.ligand_key) if b.ligand_key else None,
                "candidates": [
                    {"rank": k, "score": round(c.score, 3), "e_rel_kcal": round(c.e_rel, 2),
                     "method": c.method, **{t: (round(v, 3) if isinstance(v, float) else v) for t, v in c.terms.items()}}
                    for k, c in enumerate(b.candidates)
                ],
            }
            for b in built_list
        ],
    }
    return TargetBuild(unmodified=ctrl, variants=variants, built=built_list, resolved=resolved, report=report)


def _strip_adduct(arr: struc.AtomArray, r: PTMRequest) -> struc.AtomArray:
    """Remove the covalent ligand residue(s) of type r.payload.ligand attached at the anchor atom."""
    m = sio.residue_mask(arr, r.key)
    anchor = np.where(m & (arr.atom_name == r.payload.anchor_atom))[0]
    nbrs, _ = arr.bonds.get_bonds(int(anchor[0]))
    drop = np.zeros(arr.array_length(), bool)
    for j in nbrs:
        if arr.res_name[j] == r.payload.ligand:
            drop |= (arr.chain_id == arr.chain_id[j]) & (arr.res_id == arr.res_id[j]) & (arr.res_name == arr.res_name[j])
    return arr[~drop]


def write_outputs(tb: TargetBuild, out_dir: str | Path, name: str, fmts=("cif", "pdb")) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for fmt in fmts:
        w = sio.write_cif if fmt == "cif" else sio.write_pdb
        p = out / f"{name}_unmod.{fmt}"
        w(tb.unmodified, p)
        files[f"unmod.{fmt}"] = str(p)
        for k, v in enumerate(tb.variants):
            p = out / f"{name}_ptm_k{k}.{fmt}"
            w(v, p)
            files[f"ptm_k{k}.{fmt}"] = str(p)
    (out / f"{name}_ptm_report.json").write_text(json.dumps(tb.report, indent=1))
    files["report"] = str(out / f"{name}_ptm_report.json")
    return files
