"""CCD subset access and CCD -> RDKit conversion (atom names carried as the `name` atom prop)."""
from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

DATA = Path(__file__).with_name("data") / "ccd_subset.json.gz"
_BOND = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
    "QUAD": Chem.BondType.QUADRUPLE,
}


@lru_cache(maxsize=1)
def _subset() -> dict:
    with gzip.open(DATA, "rt") as fh:
        return json.load(fh)


def has_comp(code: str) -> bool:
    return code.upper() in _subset()


def get_comp(code: str) -> dict:
    try:
        return _subset()[code.upper()]
    except KeyError as exc:
        raise KeyError(
            f"CCD component {code!r} not in ptm_rfd3/data/ccd_subset.json.gz; extend "
            "ADDUCT_CODES in data/build_ccd_subset.py and rerun it") from exc


def parent_of(code: str) -> str | None:
    p = get_comp(code)["parent"].upper()
    return None if p == "?" else p


@lru_cache(maxsize=512)
def comp_mol(code: str, drop_leaving: bool = False, drop_atoms: tuple[str, ...] = ()) -> Chem.Mol:
    """RDKit mol (explicit hydrogens, ideal coords as conformer 0) of a CCD component.

    Each atom carries prop `name` (CCD atom_id) and int props `leave` (CCD leaving-atom flag:
    OXT, HXT, ...) and `bb` (backbone flag). Leaving atoms are KEPT by default so the force field
    sees a chemically complete (neutral, capped) amino acid; callers drop them from the output.
    """
    rec = get_comp(code)
    drop = set(drop_atoms)
    # dropping an atom also drops the hydrogens it carries (no orphan H fragments)
    nbrs: dict[str, list[str]] = {}
    for n1, n2, _o, _a in rec["bonds"]:
        nbrs.setdefault(n1, []).append(n2)
        nbrs.setdefault(n2, []).append(n1)
    elem = {a["name"]: a["el"] for a in rec["atoms"]}
    for d_atom in list(drop):
        drop.update(n for n in nbrs.get(d_atom, []) if elem.get(n) == "H")
    keep = [a for a in rec["atoms"] if a["name"] not in drop and not (drop_leaving and a["leave"])]
    idx = {a["name"]: i for i, a in enumerate(keep)}
    rw = Chem.RWMol()
    for a in keep:
        at = Chem.Atom(a["el"])
        at.SetFormalCharge(a["chg"])
        at.SetNoImplicit(True)
        at.SetProp("name", a["name"])
        at.SetIntProp("leave", int(a["leave"]))
        at.SetIntProp("bb", int(a["bb"]))
        rw.AddAtom(at)
    for n1, n2, order, arom in rec["bonds"]:
        if n1 in idx and n2 in idx and order in _BOND:
            rw.AddBond(idx[n1], idx[n2], _BOND[order])
    mol = rw.GetMol()
    if all(a["xyz"] is not None for a in keep):
        conf = Chem.Conformer(len(keep))
        for i, a in enumerate(keep):
            conf.SetAtomPosition(i, Point3D(*a["xyz"]))
        conf.Set3D(True)
        mol.AddConformer(conf, assignId=True)
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        mol,
        Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        catchErrors=True,
    )
    if mol.GetNumConformers():
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def atom_index(mol: Chem.Mol, name: str) -> int:
    for a in mol.GetAtoms():
        if a.GetProp("name") == name:
            return a.GetIdx()
    raise KeyError(name)


def names(mol: Chem.Mol) -> list[str]:
    return [a.GetProp("name") for a in mol.GetAtoms()]


def ideal_coords(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(mol.GetConformer(0).GetPositions(), dtype=float)


def heavy_names(code: str) -> list[str]:
    return [a["name"] for a in get_comp(code)["atoms"] if a["el"] != "H" and not a["leave"]]


# --------------------------------------------------------------------------------------------------------------------
# Custom covalently attached groups defined by SMILES (not in the wwPDB CCD)
# --------------------------------------------------------------------------------------------------------------------
_CUSTOM: dict[str, dict] = {}          # code -> {"smiles", "attach_atom", "record"}
_ORDER = {1: "SING", 2: "DOUB", 3: "TRIP"}


def is_custom(code: str) -> bool:
    return code.upper() in _CUSTOM


@lru_cache(maxsize=1)
def _wwpdb_codes() -> frozenset:
    """Every component id of the full wwPDB CCD (as bundled with biotite): RFD3 / AF3 look these up and would use the REAL component."""
    from biotite.structure import info
    return frozenset(info.all_residues())


@lru_cache(maxsize=1)
def free_codes() -> list[str]:
    """3-character residue codes that are not wwPDB CCD components (about 2,500 are left)."""
    import itertools
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    taken = _wwpdb_codes()
    return ["".join(t) for t in itertools.product(digits, repeat=3) if "".join(t) not in taken and t[0].isalpha()]


def _default_code(smiles: str) -> str:
    import hashlib
    free = free_codes()
    return free[int(hashlib.sha1(smiles.encode()).hexdigest(), 16) % len(free)]


def register_smiles(smiles: str, name: str | None = None, seed: int = 0) -> tuple[str, str]:
    """Define a covalently attached group from SMILES and make it available like a CCD component.

    `*` marks the attachment point: the atom of the modified residue (e.g. Lys NZ) that the group binds to. The heavy atom
    next to `*` is the group's linkage atom. Example, N6-biotinyl: `*C(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12`.
    Atoms are named <element><n> in SMILES order (C1, C2, ..., N1, O1, ...; hydrogens H1, H2, ...). `name` is the 3-character
    residue code written to the structure files (default: derived from the SMILES). Returns (code, linkage atom name)."""
    import re
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse SMILES {smiles!r}")
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        raise ValueError(f"SMILES {smiles!r} needs exactly one attachment point '*' (the residue atom the group binds to); found {len(dummies)}")
    dummy = mol.GetAtomWithIdx(dummies[0])
    nbrs = list(dummy.GetNeighbors())
    if len(nbrs) != 1 or mol.GetBondBetweenAtoms(dummy.GetIdx(), nbrs[0].GetIdx()).GetBondTypeAsDouble() != 1.0:
        raise ValueError(f"the attachment point '*' of {smiles!r} must be bonded to exactly one atom by a single bond")
    attach_idx = nbrs[0].GetIdx()
    nh = nbrs[0].GetTotalNumHs()
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(dummies[0])
    attach_idx -= int(attach_idx > dummies[0])
    at = rw.GetAtomWithIdx(attach_idx)
    at.SetNoImplicit(True)
    at.SetNumExplicitHs(nh)
    at.SetNumRadicalElectrons(1)                                   # the open valence is the bond to the residue
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=seed + 7, useRandomCoords=True) != 0:
        raise ValueError(f"could not generate 3D coordinates for {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:                                               # noqa: BLE001  (no MMFF parameters: keep the embedded geometry)
        pass
    code = (name or _default_code(smiles)).upper()
    if not re.fullmatch(r"[A-Z0-9]{3}", code):
        raise ValueError(f"residue code {code!r} must be 3 letters/digits")
    if (code in _subset() or code in _wwpdb_codes()) and code not in _CUSTOM:
        raise ValueError(f"residue code {code!r} is already a wwPDB CCD component (RFD3 and AF3 would use the real one); choose another `name`, "
                         f"e.g. one of {', '.join(free_codes()[:5])} (or omit it)")
    if code in _CUSTOM and _CUSTOM[code]["smiles"] != smiles:
        raise ValueError(f"residue code {code!r} already defines another group ({_CUSTOM[code]['smiles']}); give each SMILES its own `name`")
    count: dict[str, int] = {}
    names_: list[str] = []
    for a in m.GetAtoms():
        el = a.GetSymbol()
        count[el] = count.get(el, 0) + 1
        names_.append(f"{el}{count[el]}")
    conf = m.GetConformer()
    atoms = []
    for a, nm in zip(m.GetAtoms(), names_):
        p = conf.GetAtomPosition(a.GetIdx())
        atoms.append(dict(name=nm, el=a.GetSymbol(), chg=a.GetFormalCharge(), arom=a.GetIsAromatic(), leave=False, bb=False,
                          xyz=[round(p.x, 3), round(p.y, 3), round(p.z, 3)]))
    mk = Chem.Mol(m)
    Chem.Kekulize(mk, clearAromaticFlags=True)
    bonds = [[names_[b.GetBeginAtomIdx()], names_[b.GetEndAtomIdx()], _ORDER.get(int(b.GetBondTypeAsDouble()), "SING"),
              m.GetBondWithIdx(b.GetIdx()).GetIsAromatic()] for b in mk.GetBonds()]
    record = dict(id=code, name="custom group", type="NON-POLYMER", parent="?", charge=str(sum(a["chg"] for a in atoms)), atoms=atoms, bonds=bonds)
    attach_name = names_[attach_idx]
    _subset()[code] = record
    comp_mol.cache_clear()
    _CUSTOM[code] = dict(smiles=smiles, attach_atom=attach_name, record=record)
    return code, attach_name


def custom_components(codes=None) -> dict[str, dict]:
    """Definitions of the registered custom groups (JSON-serialisable; stored in the build manifest)."""
    return {c: v for c, v in _CUSTOM.items() if codes is None or c in set(codes)}


def load_custom(defs: dict | None) -> None:
    """Register custom groups from `custom_components()` output (another process reading a manifest)."""
    for code, v in (defs or {}).items():
        if code in _subset() and code not in _CUSTOM:
            raise ValueError(f"custom group {code!r} collides with a wwPDB CCD component")
        _subset()[code] = v["record"]
        _CUSTOM[code] = dict(smiles=v["smiles"], attach_atom=v["attach_atom"], record=v["record"])
    comp_mol.cache_clear()


def user_ccd(code: str) -> str:
    """AF3 `userCCD` mmCIF text of a registered custom group."""
    rec = _CUSTOM[code.upper()]["record"]
    mol = comp_mol(code.upper())
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit.Chem.Descriptors import MolWt
    mass = MolWt(mol)
    lines = [f"data_{code}", "#", f"_chem_comp.id {code}", '_chem_comp.name "custom group"', "_chem_comp.type non-polymer",
             f'_chem_comp.formula "{CalcMolFormula(mol)}"', "_chem_comp.mon_nstd_parent_comp_id ?", "_chem_comp.pdbx_synonyms ?",
             f"_chem_comp.formula_weight {mass:.3f}", "#", "loop_", "_chem_comp_atom.comp_id", "_chem_comp_atom.atom_id",
             "_chem_comp_atom.type_symbol", "_chem_comp_atom.charge", "_chem_comp_atom.pdbx_leaving_atom_flag",
             "_chem_comp_atom.pdbx_model_Cartn_x_ideal", "_chem_comp_atom.pdbx_model_Cartn_y_ideal", "_chem_comp_atom.pdbx_model_Cartn_z_ideal"]
    for a in rec["atoms"]:
        x, y, z = a["xyz"]
        lines.append(f"{code} {a['name']} {a['el']} {a['chg']} N {x:.3f} {y:.3f} {z:.3f}")
    lines += ["#", "loop_", "_chem_comp_bond.comp_id", "_chem_comp_bond.atom_id_1", "_chem_comp_bond.atom_id_2",
              "_chem_comp_bond.value_order", "_chem_comp_bond.pdbx_aromatic_flag"]
    for n1, n2, order, arom in rec["bonds"]:
        lines.append(f"{code} {n1} {n2} {order} {'Y' if arom else 'N'}")
    return "\n".join(lines) + "\n#\n"
