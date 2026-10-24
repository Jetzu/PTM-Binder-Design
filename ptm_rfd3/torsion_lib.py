"""Empirical torsion library for PTM side chains, learned from crystal structures (train split only).

Rotors = rotatable bonds of the modified residue whose movable side holds no parent (pinned) atom, defined once
per CCD code from the CCD graph (builder._rotors), named by 4 atom names. Native torsions are measured on crystal
PTMs; each rotor gets a smoothed circular histogram (log-density) used (a) to SAMPLE torsions in the builder
and (b) as a ranking feature (sum of -log p).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem

from . import builder as B
from . import ccd

LIB_PATH = Path(__file__).with_name("data") / "torsion_lib.json"
NBINS = 24                      # 15 degree bins
SMOOTH_SIGMA_BINS = 1.0


@lru_cache(maxsize=64)
def rotor_defs(code: str) -> list[dict]:
    """Rotors of CCD component `code` with parent-atom pins: [{names:(a,b,c,d), sp3:bool}]."""
    mol = Chem.Mol(ccd.comp_mol(code))
    parent = ccd.parent_of(code)
    nm = B._mcs_name_map(parent, code) if parent else {}
    mapped = set(nm.values())
    names = ccd.names(mol)
    pins = {i for i, n in enumerate(names) if n in mapped and mol.GetAtomWithIdx(i).GetAtomicNum() > 1}
    out = []
    for a, b, c, d, sp3 in B._rotors(mol, pins):
        out.append({"names": [names[a], names[b], names[c], names[d]], "sp3": bool(sp3)})
    return out


def torsion_deg(p0, p1, p2, p3) -> float:
    return B._dihedral(np.asarray(p0, float), np.asarray(p1, float), np.asarray(p2, float), np.asarray(p3, float))


def _smooth_hist(angles: list[float]) -> list[float]:
    h = np.zeros(NBINS)
    for a in angles:
        h[int(((a + 180.0) % 360.0) / (360.0 / NBINS)) % NBINS] += 1.0
    # circular gaussian smoothing + uniform floor (pseudo-count) so unseen angles are rare, not impossible
    ker = np.exp(-0.5 * (np.arange(-4, 5) / SMOOTH_SIGMA_BINS) ** 2)
    ker /= ker.sum()
    sm = np.zeros(NBINS)
    for k, w in zip(range(-4, 5), ker):
        sm += w * np.roll(h, k)
    p = sm + 0.15 * max(len(angles), 1) / NBINS * 0.1 + 1e-3
    return (p / p.sum()).tolist()


def learn(rows_with_native: list[tuple[str, "object"]]) -> dict:
    """rows_with_native: [(code, native residue AtomArray)] -> {code: [{names, sp3, p:[NBINS], n}]}"""
    acc: dict[str, dict[int, list[float]]] = {}
    for code, res in rows_with_native:
        xyz = {str(n): x for n, x in zip(res.atom_name, res.coord)}
        for i, r in enumerate(rotor_defs(code)):
            if all(n in xyz for n in r["names"]):
                acc.setdefault(code, {}).setdefault(i, []).append(torsion_deg(*[xyz[n] for n in r["names"]]))
    lib = {}
    for code, d in acc.items():
        lib[code] = []
        for i, r in enumerate(rotor_defs(code)):
            ang = d.get(i, [])
            lib[code].append({"names": r["names"], "sp3": r["sp3"], "n": len(ang),
                              "p": _smooth_hist(ang) if len(ang) >= 8 else None})
    return lib


def save(lib: dict, path: Path = LIB_PATH) -> None:
    path.write_text(json.dumps(lib))


@lru_cache(maxsize=1)
def load(path: str = str(LIB_PATH)) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def sample_angle(entry: dict, rng: np.random.Generator, w_lib: float = 0.8) -> float | None:
    """Draw from the empirical histogram (w_lib) else uniform; None if the rotor has no library."""
    if entry.get("p") is None:
        return None
    if rng.random() > w_lib:
        return float(rng.uniform(-180, 180))
    p = np.asarray(entry["p"])
    k = int(rng.choice(NBINS, p=p))
    return float(-180.0 + (k + rng.random()) * 360.0 / NBINS)


def neg_logp(entry: dict, angle_deg: float) -> float:
    if entry.get("p") is None:
        return 0.0
    k = int(((angle_deg + 180.0) % 360.0) / (360.0 / NBINS)) % NBINS
    return float(-np.log(entry["p"][k] * NBINS))        # 0 at uniform density, >0 for unlikely torsions
