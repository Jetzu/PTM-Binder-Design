"""PTM / covalent-modification support for LigandMPNN.

Upstream LigandMPNN only knows the 20 standard residues. Modified residues (SEP, TPO, PTR, MSE, ...) are
either read as token "X" with their modification atoms silently dropped (when ProDy's `protein` flag
covers them) or become a chain gap plus ligand atoms (when it does not). This module adds:

* a parent-residue table for every L-peptide-linking CCD component with a standard parent
  (ptm_data/modified_residues.json, built by ptm_data/build_modified_residues.py);
* `split_polymer_atoms`: polymer = standard residues + modified residues (+ peptide-bonded unknown
  residues, parent "X"); everything else stays ligand context;
* per-residue records (original atoms, parent type, modification atoms) for verbatim output;
* `find_covalent_links`: ligand atoms covalently bonded to protein side chains (glycans, covalent
  inhibitors, cofactors);
* `apply_ptm_policy`: fix PTM / covalently modified residues and add the modification atoms of the
  fixed residues to the ligand context.
"""
import json
import os

import numpy as np
import torch

_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ptm_data", "modified_residues.json"
)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip

# heavy atoms of the standard residues (everything LigandMPNN's atom37 slots can hold, minus OXT)
STANDARD_ATOMS = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}  # fmt: skip
BACKBONE_NAMES = ("N", "CA", "C", "O")
LINK_BACKBONE_NAMES = BACKBONE_NAMES + ("OXT",)  # a link through any of these never freezes a residue
# the 36 atom names LigandMPNN's xyz_37 slots hold (data_utils.parse_PDB atom_order without OXT); with
# --ligand_mpnn_use_side_chain_context these atoms already reach the model through xyz_37
ATOM37_NAMES = frozenset(
    "N CA C CB O CG CG1 CG2 OG OG1 SG CD CD1 CD2 ND1 ND2 OD1 OD2 SD CE CE1 CE2 CE3 NE NE1 NE2 OE1 OE2 CH2 NH1 NH2 OH "
    "CZ CZ2 CZ3 NZ".split()
)

# single-bond covalent radii (Cordero 2008), Angstrom
COVALENT_RADII = {
    "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "SI": 1.11, "P": 1.07, "S": 1.05,
    "CL": 1.02, "SE": 1.20, "BR": 1.20, "I": 1.39, "AS": 1.19,
}  # fmt: skip
BOND_TOLERANCE = 0.45  # Angstrom added to r_i + r_j

# residue classes the fine-tuned model can tell apart (id 0 = standard residue). Frequency-ordered common
# modified residues; everything else in the CCD table -> OTHER, unknown peptide-bonded residue -> UNKNOWN,
# standard residue carrying a covalently attached ligand -> COVALENT.
PTM_CLASSES = [
    "SEP", "TPO", "PTR", "MSE", "HYP", "MLY", "M3L", "ALY", "MLZ", "KCX", "CSO", "CSD", "OCS", "CME", "LLP",
    "PCA", "SEC", "NEP", "HIC", "SMC", "CGU", "FME", "TPQ", "CSS", "SNC", "YCM", "KPI", "LYZ", "MHO", "OAS",
]  # fmt: skip
PTM_ID_OTHER = len(PTM_CLASSES) + 1
PTM_ID_UNKNOWN = len(PTM_CLASSES) + 2
PTM_ID_COVALENT = len(PTM_CLASSES) + 3
NUM_PTM_CLASSES = len(PTM_CLASSES) + 4
FLAG_NONE, FLAG_PTM_ATOM, FLAG_COVALENT_LIGAND = 0, 1, 2  # per ligand-context atom
NUM_Y_FLAGS = 3
# parameters added on top of the released checkpoints; zero-initialised, so released weights behave unchanged
PTM_PARAM_PREFIXES = ("ptm_emb_node.", "features.flag_type.", "features.flag_nodes.")


def ptm_class_id(resname, parent):
    if parent == "UNK":
        return PTM_ID_UNKNOWN
    return PTM_CLASSES.index(resname) + 1 if resname in PTM_CLASSES else PTM_ID_OTHER


_table_cache = None


def load_modified_residues():
    """{CCD id: parent 3-letter code}. Empty dict (with a warning) if the table file is missing."""
    global _table_cache
    if _table_cache is None:
        try:
            with open(_TABLE_PATH, "r") as fh:
                _table_cache = {k: v["parent"] for k, v in json.load(fh).items()}
        except OSError:
            print(
                f"WARNING: {_TABLE_PATH} not found - modified residues are not recognised "
                "(run ptm_data/build_modified_residues.py)."
            )
            _table_cache = {}
    return _table_cache


def _res_key(chain, resnum, icode):
    return chain + "_" + str(resnum) + "_" + icode


def _indices(sel):
    return np.asarray(sel.getIndices()) if hasattr(sel, "getIndices") else np.arange(sel.numAtoms())


def _residue_groups(atoms, idx):
    """Group atom indices `idx` (into `atoms`) by (chain, resnum, icode, resname); keeps file order."""
    chids = atoms.getChids()
    resnums = atoms.getResnums()
    icodes = atoms.getIcodes()
    resnames = atoms.getResnames()
    groups = {}
    for i in idx:
        groups.setdefault((chids[i], int(resnums[i]), icodes[i], resnames[i]), []).append(int(i))
    return groups


def split_polymer_atoms(atoms, auto_detect=True):
    """Split parsed atoms into polymer (protein incl. modified residues) and ligand atoms.

    atoms: ProDy Selection/AtomGroup (waters are not touched; they stay out of both groups)
    Returns (polymer_atoms, other_atoms, modified) where `modified` maps the residue key
    chain_resnum_icode -> {"resname", "parent" (3-letter or "UNK"), "auto": bool}.
    """
    table = load_modified_residues()
    resnames = atoms.getResnames()
    names = atoms.getNames()
    protein_sel = atoms.select("protein")
    is_poly = np.zeros(atoms.numAtoms(), dtype=bool)
    if protein_sel is not None:
        # Selection indices refer to the parent AtomGroup; compare in that frame
        is_poly = np.isin(_indices(atoms), _indices(protein_sel))
    water_sel = atoms.select("water")
    is_water = np.zeros(atoms.numAtoms(), dtype=bool)
    if water_sel is not None:
        is_water = np.isin(_indices(atoms), _indices(water_sel))
    is_mod_name = np.array([r in table for r in resnames]) & ~is_water

    # modified residues with a CA anchor become polymer residues (missing backbone atoms only mask them,
    # exactly like an incomplete standard residue, instead of silently opening a chain gap)
    modified = {}
    cand = np.where(is_mod_name & ~is_poly)[0]
    for key, members in _residue_groups(atoms, cand).items():
        member_names = {names[i] for i in members}
        if "CA" in member_names:
            is_poly[members] = True
            modified[_res_key(*key[:3])] = {"resname": key[3], "parent": table[key[3]], "auto": False}
    auto_keys = []
    # modified residues already caught by ProDy's `protein` flag (SEP, TPO, PTR, MSE, CSO, ...)
    for key, members in _residue_groups(atoms, np.where(is_mod_name & is_poly)[0]).items():
        modified[_res_key(*key[:3])] = {"resname": key[3], "parent": table[key[3]], "auto": False}

    if auto_detect:
        # unknown residues peptide-bonded to the polymer (needs N, CA, C): modified, parent unknown.
        # Repeat until nothing is added so arbitrarily long runs of unknown residues are followed.
        coords = atoms.getCoords()
        groups = {
            key: {names[i]: i for i in members}
            for key, members in _residue_groups(atoms, np.where(~is_poly & ~is_water)[0]).items()
        }
        groups = {k: m for k, m in groups.items() if {"N", "CA", "C"} <= set(m)}
        poly_idx = np.where(is_poly)[0]
        pN = [coords[i] for i in poly_idx if names[i] == "N"]
        pC = [coords[i] for i in poly_idx if names[i] == "C"]
        added = True
        while added and groups:
            added = False
            for key in list(groups):
                m = groups[key]
                bonded = (pN and min(np.linalg.norm(np.array(pN) - coords[m["C"]], axis=1)) < 1.8) or (
                    pC and min(np.linalg.norm(np.array(pC) - coords[m["N"]], axis=1)) < 1.8
                )
                if bonded:
                    del groups[key]
                    auto_keys.append(key)
                    pN.append(coords[m["N"]])
                    pC.append(coords[m["C"]])
                    added = True
        for key, members in _residue_groups(atoms, np.where(~is_poly & ~is_water)[0]).items():
            if key in auto_keys:
                is_poly[members] = True
                modified[_res_key(*key[:3])] = {"resname": key[3], "parent": "UNK", "auto": True}

    poly_idx = np.where(is_poly)[0]
    other_idx = np.where(~is_poly & ~is_water)[0]
    polymer_atoms = atoms[poly_idx] if len(poly_idx) else None
    other_atoms = atoms[other_idx] if len(other_idx) else None
    return polymer_atoms, other_atoms, modified


def build_residue_records(polymer_atoms, CA_dict, modified, element_dict):
    """Per modified residue: original atoms and the modification ("extra") atoms.

    Returns (records, extra_Y, extra_Y_t, extra_res, extra_names): records maps residue index (row of CA_dict) to
    {"resname","parent","auto","names","elements","coords","betas","hetatm","n_extra"}; extra_* hold the
    heavy modification atoms (not part of the parent's standard atom set) for the ligand context.
    """
    chids = polymer_atoms.getChids()
    resnums = polymer_atoms.getResnums()
    icodes = polymer_atoms.getIcodes()
    names = polymer_atoms.getNames()
    elements = polymer_atoms.getElements()
    coords = polymer_atoms.getCoords()
    betas = polymer_atoms.getBetas()
    hetatm = polymer_atoms.getFlags("hetatm")
    members = {}
    for i in range(len(chids)):
        key = _res_key(chids[i], resnums[i], icodes[i])
        if key in modified:
            members.setdefault(key, []).append(i)

    records, ey, ey_t, ey_res, ey_names = {}, [], [], [], []
    for key, mem in members.items():
        if key not in CA_dict:
            continue
        info = modified[key]
        std = set(STANDARD_ATOMS.get(info["parent"], BACKBONE_NAMES)) | {"OXT"}
        n_extra = 0
        for i in mem:
            el = elements[i].upper()
            if names[i] in std or el in ("H", "D", ""):
                continue
            code = element_dict.get(el, 0)
            if code == 0:
                continue
            ey.append(coords[i])
            ey_t.append(code)
            ey_res.append(CA_dict[key])
            ey_names.append(names[i])
            n_extra += 1
        records[CA_dict[key]] = {
            **info,
            "names": np.array([names[i] for i in mem]),
            "elements": np.array([elements[i] for i in mem]),
            "coords": coords[mem].astype(np.float32),
            "betas": betas[mem].astype(np.float32),
            "hetatm": np.array([bool(hetatm[i]) for i in mem]),
            "n_extra": n_extra,
        }
    extra_Y = np.array(ey, dtype=np.float32).reshape(-1, 3)
    return (
        records,
        extra_Y,
        np.array(ey_t, dtype=np.int32),
        np.array(ey_res, dtype=np.int64),
        np.array(ey_names, dtype=object),
    )


def find_covalent_links(polymer_atoms, other_atoms, CA_dict):
    """Ligand heavy atoms covalently bonded to a polymer atom (distance < r_i + r_j + 0.45 A).

    Metals and hydrogens are skipped (coordination is not covalent attachment). Links through the peptide
    backbone atoms (N, CA, C, O; e.g. ACE/NME caps) are reported with backbone=True and do not freeze a
    residue. Returns a list of dicts.
    """
    if polymer_atoms is None or other_atoms is None:
        return []
    o_el = np.array([e.upper() for e in other_atoms.getElements()])
    o_ok = np.array([e in COVALENT_RADII for e in o_el])
    p_el = np.array([e.upper() for e in polymer_atoms.getElements()])
    p_ok = np.array([e in COVALENT_RADII for e in p_el])
    if not o_ok.any() or not p_ok.any():
        return []
    o_idx = np.where(o_ok)[0]
    p_idx = np.where(p_ok)[0]
    o_xyz = other_atoms.getCoords()[o_idx]
    p_xyz = polymer_atoms.getCoords()[p_idx]
    o_r = np.array([COVALENT_RADII[e] for e in o_el[o_idx]])
    p_r = np.array([COVALENT_RADII[e] for e in p_el[p_idx]])
    p_chid = polymer_atoms.getChids()
    p_resnum = polymer_atoms.getResnums()
    p_icode = polymer_atoms.getIcodes()
    p_name = polymer_atoms.getNames()
    p_resname = polymer_atoms.getResnames()
    o_name = other_atoms.getNames()
    o_resname = other_atoms.getResnames()
    o_chid = other_atoms.getChids()
    o_resnum = other_atoms.getResnums()
    o_icode = other_atoms.getIcodes()

    links = []
    from scipy.spatial import cKDTree  # noqa: PLC0415 - neighbour search: large RNA/ribosome ligand clouds made the dense version O(N*M)

    tree = cKDTree(p_xyz)
    max_thr = float(o_r.max() + p_r.max() + BOND_TOLERANCE)
    for oi, cand in enumerate(tree.query_ball_point(o_xyz, r=max_thr)):
        for pj in cand:
            dist = float(np.linalg.norm(o_xyz[oi] - p_xyz[pj]))
            if dist >= o_r[oi] + p_r[pj] + BOND_TOLERANCE:
                continue
            o = o_idx[oi]
            p = p_idx[pj]
            key = _res_key(p_chid[p], p_resnum[p], p_icode[p])
            if key not in CA_dict:
                continue
            links.append(
                {
                    "res_index": CA_dict[key],
                    "residue": f"{p_chid[p]}{p_resnum[p]}{p_icode[p]}",
                    "resname": p_resname[p],
                    "atom": p_name[p],
                    "backbone": p_name[p] in LINK_BACKBONE_NAMES,
                    "ligand": f"{o_resname[o]}:{o_chid[o]}{o_resnum[o]}",
                    "ligand_key": (o_chid[o], int(o_resnum[o]), o_icode[o], o_resname[o]),
                    "ligand_atom": o_name[o],
                    "distance": dist,
                }
            )
    return links


def find_crosslinks(polymer_atoms, CA_dict):
    """Covalent bonds BETWEEN protein residues beyond the peptide bond: isopeptide (ubiquitin / SUMO Lys-Gly), lanthionine
    and other side-chain cross-links. Both residues are returned as covalently modified (backbone=False, kind "crosslink").

    Skipped: peptide bonds and any backbone-backbone contact (also head-to-tail macrocycle closures), standard Cys-Cys
    disulfides (SG-SG; those residues stay designable, as upstream), metals and hydrogens.
    """
    if polymer_atoms is None:
        return []
    from scipy.spatial import cKDTree  # noqa: PLC0415

    el = np.array([e.upper() for e in polymer_atoms.getElements()])
    ok = np.array([e in COVALENT_RADII for e in el])
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return []
    xyz = polymer_atoms.getCoords()[idx]
    rad = np.array([COVALENT_RADII[e] for e in el[idx]])
    chid, resnum, icode = polymer_atoms.getChids(), polymer_atoms.getResnums(), polymer_atoms.getIcodes()
    name, resname = polymer_atoms.getNames(), polymer_atoms.getResnames()
    row = np.array([CA_dict.get(_res_key(chid[i], resnum[i], icode[i]), -1) for i in idx])
    links = []
    for a, b in cKDTree(xyz).query_pairs(r=float(2 * rad.max() + BOND_TOLERANCE)):
        ra, rb = int(row[a]), int(row[b])
        if ra < 0 or rb < 0 or ra == rb or abs(ra - rb) == 1:
            continue
        ia, ib = idx[a], idx[b]
        na, nb = name[ia], name[ib]
        if na in LINK_BACKBONE_NAMES and nb in LINK_BACKBONE_NAMES:
            continue
        if na == "SG" and nb == "SG":
            continue
        d = float(np.linalg.norm(xyz[a] - xyz[b]))
        if d >= rad[a] + rad[b] + BOND_TOLERANCE:
            continue
        for (r1, i1, n1), (r2, i2, n2) in (((ra, ia, na), (rb, ib, nb)), ((rb, ib, nb), (ra, ia, na))):
            links.append(
                {
                    "res_index": r1,
                    "residue": f"{chid[i1]}{resnum[i1]}{icode[i1]}",
                    "resname": resname[i1],
                    "atom": n1,
                    "backbone": False,
                    "ligand": f"{resname[i2]}:{chid[i2]}{resnum[i2]}",
                    "ligand_key": None,
                    "ligand_atom": n2,
                    "distance": d,
                    "kind": "crosslink",
                }
            )
    return links


def apply_ptm_policy(
    protein_dict,
    explicit_redesign=False,
    fix_ptm=True,
    fix_covalent=True,
    context_atoms="sidechain",
    verbose=False,
):
    """Freeze PTM / covalently modified residues and merge their modification atoms into the ligand context.

    protein_dict: output of parse_PDB with "chain_mask" already set (1 = design, 0 = fixed). Updated in place.
    explicit_redesign: True when the user gave --redesigned_residues; that list is authoritative, so no
        extra residue is frozen (PTM residues the user did not list are fixed by the list itself).
    context_atoms: "sidechain" (default) = all side-chain atoms of fixed PTM residues, minus those already given by
        --ligand_mpnn_use_side_chain_context; "extras" = only the modification atoms (phosphate, methyl, ...);
        "none" = no PTM atoms in the context.
    """
    ptm = protein_dict.get("ptm")
    if not ptm:
        return protein_dict
    device = protein_dict["chain_mask"].device
    chain_mask = protein_dict["chain_mask"]
    L = chain_mask.shape[0]
    ptm_mask = torch.zeros(L, dtype=chain_mask.dtype, device=device)
    for i in ptm["records"]:
        ptm_mask[i] = 1
    covalent_mask = torch.zeros(L, dtype=chain_mask.dtype, device=device)
    for link in ptm["covalent_links"]:
        if not link["backbone"]:
            covalent_mask[link["res_index"]] = 1
    if not explicit_redesign:
        if fix_ptm:
            chain_mask = chain_mask * (1 - ptm_mask)
        if fix_covalent:
            chain_mask = chain_mask * (1 - covalent_mask)
        protein_dict["chain_mask"] = chain_mask
    ptm["fixed"] = {i for i in ptm["records"] if int(chain_mask[i]) == 0}
    # per-residue class id for the model: only for residues that keep their modification (fixed)
    if "ptm_id" in protein_dict:
        ids = protein_dict["ptm_id"].clone()
        for i in range(L):
            if int(chain_mask[i]) == 1:
                ids[i] = 0
        protein_dict["ptm_id"] = ids
    if verbose and ptm["records"]:
        print("Modified residues (PTM) found:")
        for i, r in sorted(ptm["records"].items()):
            state = "fixed" if i in ptm["fixed"] else "redesigned (modification dropped)"
            print(
                f"  index {i}: {r['resname']} -> parent {r['parent']}, {r['n_extra']} modification atoms, {state}"
            )
    if verbose and ptm["covalent_links"]:
        print("Covalent protein-ligand links found:")
        for k in ptm["covalent_links"]:
            tag = " (backbone link, residue not frozen)" if k["backbone"] else ""
            print(
                f"  {k['residue']} {k['resname']}:{k['atom']} -- {k['ligand']}:{k['ligand_atom']} "
                f"{k['distance']:.2f} A{tag}"
            )

    # modification atoms of the residues that stay fixed -> ligand context
    if context_atoms == "none" or not ptm["fixed"]:
        return protein_dict
    ys, yts = [], []
    use_sc = bool(protein_dict.get("_use_side_chain_context", False))
    if context_atoms == "extras":
        sel = np.array([int(r) in ptm["fixed"] for r in ptm["extra_res"]], dtype=bool)
        if use_sc and len(sel):
            # atoms whose names fill an xyz_37 slot (e.g. HYP OD1) already reach the model through the
            # side-chain context of fixed residues; adding them again would count them twice
            sel &= np.array([n not in ATOM37_NAMES for n in ptm["extra_names"]], dtype=bool)
        if len(sel) and sel.any():
            ys.append(ptm["extra_Y"][sel])
            yts.append(ptm["extra_Y_t"][sel])
    elif context_atoms in ("sidechain", "sidechain_noextras"):
        # sidechain_noextras = the same residues WITHOUT their modification atoms (phosphate, methyl, ...): the
        # "unmodified" counterpart of the sidechain context, used as the negative branch of contrastive decoding
        extras_of = {}
        if context_atoms == "sidechain_noextras":
            for rr, nn in zip(ptm["extra_res"], ptm["extra_names"]):
                extras_of.setdefault(int(rr), set()).add(nn)
        for i in ptm["fixed"]:
            r = ptm["records"][i]
            skip = (set(ATOM37_NAMES) if use_sc else set(BACKBONE_NAMES)) | extras_of.get(int(i), set())
            for n, el, xyz in zip(r["names"], r["elements"], r["coords"]):
                code = ptm["element_dict"].get(el.upper(), 0)
                if n in skip or n == "OXT" or el.upper() in ("H", "D", "") or code == 0:
                    continue
                ys.append(xyz[None])
                yts.append(np.array([code], dtype=np.int32))
    else:
        raise ValueError(f"unknown context_atoms mode: {context_atoms}")
    if ys:
        extra_Y = torch.tensor(np.concatenate(ys, 0), device=device, dtype=torch.float32)
        extra_Y_t = torch.tensor(np.concatenate(yts, 0), device=device, dtype=torch.int32)
        extra_Y_m = torch.ones_like(extra_Y_t)
        protein_dict["Y"] = torch.cat([protein_dict["Y"], extra_Y], 0)
        protein_dict["Y_t"] = torch.cat([protein_dict["Y_t"], extra_Y_t], 0)
        protein_dict["Y_m"] = torch.cat([protein_dict["Y_m"], extra_Y_m], 0)
        if "Y_flag" in protein_dict:
            extra_flag = torch.full_like(extra_Y_t, FLAG_PTM_ATOM)
            protein_dict["Y_flag"] = torch.cat([protein_dict["Y_flag"], extra_flag], 0)
    return protein_dict


def add_ptm_arguments(argparser):
    """Command-line flags shared by run.py and score.py."""
    argparser.add_argument(
        "--ptm_aware",
        type=int,
        default=1,
        help="1 - parse modified residues (SEP, TPO, PTR, MSE, HYP, ...) as residues of their parent type, "
        "keep their atoms in the output and find covalent ligand links; 0 - upstream behaviour.",
    )
    argparser.add_argument(
        "--ptm_auto_detect",
        type=int,
        default=1,
        help="1 - also treat residues missing from the CCD table that are peptide-bonded to the chain as "
        "modified residues of unknown parent (token X).",
    )
    argparser.add_argument(
        "--detect_crosslinks",
        type=int,
        default=1,
        help="1 - detect covalent bonds between protein residues (isopeptide Lys-Gly of ubiquitin/SUMO, lanthionine, ...); "
        "both residues are treated as covalently modified (fixed). Standard disulfides are not affected.",
    )
    argparser.add_argument(
        "--fix_ptm_residues",
        type=int,
        default=1,
        help="1 - keep modified residues fixed (their modification only exists on that residue type); "
        "0 - allow redesign (modification atoms are then dropped). Ignored when --redesigned_residues is set.",
    )
    argparser.add_argument(
        "--fix_covalent_residues",
        type=int,
        default=1,
        help="1 - keep residues that carry a covalently attached ligand (glycan, covalent inhibitor) fixed. "
        "Ignored when --redesigned_residues is set.",
    )
    argparser.add_argument(
        "--ptm_context_atoms",
        type=str,
        default="sidechain",
        help="Atoms of fixed modified residues added to the ligand context (ligand_mpnn only): "
        "sidechain - whole side chain (best near the sites in the N001 benchmark); extras - only the "
        "modification atoms (phosphate, methyl, ...); none - keep just the parent identity.",
    )


def load_state_dict_tolerant(model, state_dict):
    """Load a (released or fine-tuned) checkpoint. Released checkpoints lack the PTM parameters
    (PTM_PARAM_PREFIXES): those stay zero-initialised and the model behaves exactly like the released one.
    Any other missing or unexpected key is still an error."""
    res = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in res.missing_keys if not k.startswith(PTM_PARAM_PREFIXES)]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing} unexpected={list(res.unexpected_keys)}")
    return [k for k in res.missing_keys if k.startswith(PTM_PARAM_PREFIXES)]
