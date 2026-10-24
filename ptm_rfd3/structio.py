"""Structure I/O on biotite AtomArrays (PDB / mmCIF in, mmCIF / PDB out)."""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from biotite import structure as struc
from biotite.structure.io import pdb, pdbx

EXTRA = ["b_factor", "occupancy", "charge"]


@dataclass(frozen=True)
class ResKey:
    chain: str
    res_id: int
    ins: str = ""

    @classmethod
    def parse(cls, text: str) -> "ResKey":
        """'A45' / 'A:45' / 'A45B' (insertion code) -> ResKey."""
        t = text.replace(":", "").strip()
        chain, rest = t[0], t[1:]
        digits = ""
        ins = ""
        for k, ch in enumerate(rest):
            if ch.isdigit() or (ch == "-" and k == 0):
                digits += ch
            else:
                ins = rest[k:]
                break
        return cls(chain, int(digits), ins)

    def __str__(self) -> str:
        return f"{self.chain}{self.res_id}{self.ins}"


def read_structure(path: str | Path, model: int = 1, drop_h: bool = True, drop_water: bool = True) -> struc.AtomArray:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    stem = path.name[:-3] if path.suffix == ".gz" else path.name
    if stem.lower().endswith((".cif", ".mmcif")):
        with opener(path, "rt") as fh:
            f = pdbx.CIFFile.read(fh)
        arr = pdbx.get_structure(f, model=model, extra_fields=EXTRA, use_author_fields=True,
                                 include_bonds=True)
    else:
        with opener(path, "rt") as fh:
            f = pdb.PDBFile.read(fh)
        arr = pdb.get_structure(f, model=model, extra_fields=EXTRA, include_bonds=True)
    # first altloc handled by biotite (altloc='first' default)
    mask = np.ones(arr.array_length(), bool)
    if drop_h:
        mask &= arr.element != "H"
        mask &= arr.element != "D"
    if drop_water:
        mask &= ~np.isin(arr.res_name, ["HOH", "WAT", "DOD"])
    return arr[mask]


def residue_mask(arr: struc.AtomArray, key: ResKey) -> np.ndarray:
    m = (arr.chain_id == key.chain) & (arr.res_id == key.res_id)
    if key.ins:
        m &= arr.ins_code == key.ins
    elif "ins_code" in arr.get_annotation_categories():
        m &= np.isin(arr.ins_code, ["", " "])
    return m


def residue_slice(arr: struc.AtomArray, key: ResKey) -> struc.AtomArray:
    m = residue_mask(arr, key)
    if not m.any():
        raise KeyError(f"residue {key} not found")
    return arr[m]


def write_cif(arr: struc.AtomArray, path: str | Path, data_name: str = "ptm_rfd3") -> None:
    f = pdbx.CIFFile()
    pdbx.set_structure(f, arr, data_block=data_name)
    f.write(str(path))


def write_pdb(arr: struc.AtomArray, path: str | Path) -> None:
    f = pdb.PDBFile()
    pdb.set_structure(f, arr)
    f.write(str(path))


def new_atom_array(n: int) -> struc.AtomArray:
    """AtomArray with the annotations we always set."""
    arr = struc.AtomArray(n)
    for cat, dt in (("b_factor", float), ("occupancy", float), ("charge", int)):
        arr.add_annotation(cat, dt)
    arr.occupancy[:] = 1.0
    return arr


def next_free_chain_id(arr: struc.AtomArray) -> str:
    used = set(np.unique(arr.chain_id))
    for c in "ZYXWVUTSRQPONMLKJIHGFEDCBA":
        if c not in used:
            return c
    for c in "abcdefghijklmnopqrstuvwxyz0123456789":
        if c not in used:
            return c
    raise RuntimeError("no free chain id")
