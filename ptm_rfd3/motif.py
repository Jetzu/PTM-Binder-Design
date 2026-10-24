"""PTM / covalent-ligand motifs for GENERATED chains: a short peptide with the modification built in, to be used as a fixed,
indexed motif inside a newly diffused chain (`ptm_motifs` spec key, see rfd3_run).

Why a peptide and not a single residue: RFD3 cannot place an atomised (modified / ligand-bearing) residue by itself
(`unindex` fails for non-standard residues and ligands) and a lone fixed residue is not connected to the residues generated
around it (gaps of 2-22 A in the N011 test). A 3-residue segment with real peptide geometry on both sides is connected
by the model (peptide bonds 1.2-1.4 A, covalent link intact, 12-15 of 16 designs). The position of the motif in the new chain
follows from the contig length ranges, e.g. "40-60,{motif},40-60".
"""
from __future__ import annotations

import logging

import numpy as np
from biotite import structure as struc

from . import ccd
from . import structio as sio
from . import target as T
from .af3_inputs import AA3

log = logging.getLogger("ptm_rfd3.motif")
AA1 = {v: k for k, v in {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
                         "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
                         "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}.items()}
ONE2THREE = {v: k for k, v in AA1.items()}
PHIPSI = {"helix": (-57.0, -47.0), "strand": (-120.0, 130.0), "ppii": (-75.0, 145.0)}
_B = dict(N_CA=1.458, CA_C=1.525, C_O=1.231, C_N=1.329)
_A = dict(N_CA_C=111.2, CA_C_N=116.2, C_N_CA=121.7, CA_C_O=120.8)


def _place(a, b, c, bond, angle_deg, torsion_deg):
    """NeRF: atom d bonded to c, with angle(b,c,d) and torsion(a,b,c,d)."""
    ang, tor = np.radians(angle_deg), np.radians(torsion_deg)
    bc = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d = np.array([-bond * np.cos(ang), bond * np.sin(ang) * np.cos(tor), bond * np.sin(ang) * np.sin(tor)])
    return c + d[0] * bc + d[1] * m + d[2] * n


def _cb(n, ca, c):
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def _kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    U, _s, Vt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - R @ Pc


def backbone_peptide(resnames: list[str], ss: str = "helix", chain: str = "A") -> struc.AtomArray:
    """Ideal-geometry peptide of standard residues (backbone from NeRF with the given phi/psi, side chains from the CCD
    ideal coordinates fitted onto N-CA-C), heavy atoms only, residue ids 1..n, peptide bonds set."""
    phi, psi = PHIPSI[ss]
    N, CA, C, O = [], [], [], []
    for i in range(len(resnames)):
        if i == 0:
            n0 = np.zeros(3)
            ca0 = np.array([_B["N_CA"], 0.0, 0.0])
            ang = np.radians(_A["N_CA_C"])
            c0 = ca0 + _B["CA_C"] * np.array([-np.cos(ang), np.sin(ang), 0.0])
            N.append(n0), CA.append(ca0), C.append(c0)
        else:
            n_ = _place(N[-1], CA[-1], C[-1], _B["C_N"], _A["CA_C_N"], psi)
            ca_ = _place(CA[-1], C[-1], n_, _B["N_CA"], _A["C_N_CA"], 180.0)
            c_ = _place(C[-1], n_, ca_, _B["CA_C"], _A["N_CA_C"], phi)
            N.append(n_), CA.append(ca_), C.append(c_)
        O.append(_place(N[-1], CA[-1], C[-1], _B["C_O"], _A["CA_C_O"], psi + 180.0))
    atoms = []                                                  # (res_idx, name, element, xyz)
    for i, code in enumerate(resnames):
        atoms += [(i, "N", "N", N[i]), (i, "CA", "C", CA[i]), (i, "C", "C", C[i]), (i, "O", "O", O[i])]
        if code == "GLY":
            continue
        mol = ccd.comp_mol(code, drop_leaving=True)
        nm = ccd.names(mol)
        xyz = ccd.ideal_coords(mol)
        ref = {n: xyz[k] for k, n in enumerate(nm)}
        R, t = _kabsch(np.array([ref["N"], ref["CA"], ref["C"]]), np.array([N[i], CA[i], C[i]]))
        for k, n in enumerate(nm):
            if n in ("N", "CA", "C", "O", "OXT") or mol.GetAtomWithIdx(k).GetAtomicNum() == 1:
                continue
            atoms.append((i, n, mol.GetAtomWithIdx(k).GetSymbol(), R @ xyz[k] + t))
    a = struc.AtomArray(len(atoms))
    for j, (i, n, el, x) in enumerate(atoms):
        a.coord[j] = x
        a.atom_name[j], a.element[j], a.res_name[j] = n, el, resnames[i]
        a.res_id[j], a.chain_id[j] = i + 1, chain
    a.hetero[:] = False
    a.bonds = struc.connect_via_residue_names(a)
    return a


def motif_definition(mod: str, seq: str | None = None, pos: int | None = None) -> tuple[list[str], int]:
    """(3-letter residue names of the peptide, 1-based index of the modified residue). Default peptide: A-<parent>-A."""
    req = T.parse_requests([f"A1={mod}"])[0]
    if req.kind == "replace":
        parents = [req.parent or ccd.parent_of(str(req.payload))]
    else:
        parents = list(req.parent) if isinstance(req.parent, (tuple, list)) else [req.parent]
    parents = [p for p in parents if p]
    if not parents and seq is None:
        raise ValueError(f"cannot derive the parent residue of {mod!r}; pass `seq` and `pos` (e.g. 'ACA', 2)")
    seq = seq or ("A" + AA1[parents[0]] + "A")
    names = [ONE2THREE[c] for c in seq.upper()]
    if pos is None:
        pos = next((i for i, n in enumerate(names, 1) if not parents or n in parents), None)
    if pos is None or (parents and names[pos - 1] not in parents):
        raise ValueError(f"motif sequence {seq!r} has no {'/'.join(parents)} at position {pos} for {mod!r}")
    return names, int(pos)


def build_motif(mod: str, seq: str | None = None, pos: int | None = None, ss: str = "helix", n_confs: int = 300,
                top_k: int = 1, seed: int = 0, chain: str = "A"):
    """Peptide with the modification built on residue `pos`. Returns (TargetBuild, modified residue key text)."""
    names, p = motif_definition(mod, seq, pos)
    arr = backbone_peptide(names, ss=ss, chain=chain)
    tb = T.build_modified_target(arr, T.parse_requests([f"{chain}{p}={mod}"]), n_confs=n_confs, top_k=top_k, seed=seed)
    d = steric_min_distance(arr)
    if d < 2.4:
        log.warning("ptm_rfd3.motif: peptide %s (%s) has a non-bonded heavy-atom contact of %.2f A in the unmodified backbone template; "
                    "pick another `seq` or `ss`", "".join(names), ss, d)
    return tb, f"{chain}{p}"


def steric_min_distance(arr: struc.AtomArray) -> float:
    """Smallest heavy-atom distance between residues, ignoring the backbone atoms of sequence neighbours (1-2 / 1-3 / 1-4 pairs)."""
    best = 9.9
    bb = {"N", "CA", "C", "O"}
    for i in np.unique(arr.res_id):
        for j in np.unique(arr.res_id):
            if j <= i:
                continue
            a, b = arr[arr.res_id == i], arr[arr.res_id == j]
            if j - i == 1:
                a, b = a[~np.isin(a.atom_name, list(bb))], b[~np.isin(b.atom_name, list(bb))]
                a2 = arr[(arr.res_id == i) & np.isin(arr.atom_name, list(bb))]
                d = np.linalg.norm(a.coord[:, None] - b.coord[None], axis=-1).min() if len(a) and len(b) else 9.9
                best = min(best, float(d))
                continue
            best = min(best, float(np.linalg.norm(a.coord[:, None] - b.coord[None], axis=-1).min()))
    return best
