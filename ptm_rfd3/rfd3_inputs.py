"""PTM sites of a prepared target -> RFD3 specs (contig, hotspots on the modification atoms)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from biotite import structure as struc

from . import ccd
from .af3_inputs import AA3
from .builder import _mcs_name_map

BACKBONE = {"N", "CA", "C", "O", "OXT"}


@dataclass
class PTMSite:
    chain: str                   # prepared chain letter of the modified residue / ligand
    res_id: int
    code: str                    # CCD code of the modified residue or the ligand
    kind: str                    # 'replace' | 'adduct'
    ptm_atoms: list[str]         # atoms that belong to the modification (hotspot candidates)
    side_atoms: list[str] = field(default_factory=list)   # remaining side-chain atoms (replace)
    anchor: tuple[str, int, str] | None = None            # adduct: (chain, res_id, atom) it is bonded to

    @property
    def key(self) -> str:
        return f"{self.chain}{self.res_id}"


def detect_sites(arr: struc.AtomArray) -> list[PTMSite]:
    """Find modified residues and covalently attached ligands in a PREPARED target."""
    sites: list[PTMSite] = []
    seen = set()
    for i in range(arr.array_length()):
        k = (str(arr.chain_id[i]), int(arr.res_id[i]), str(arr.res_name[i]))
        if k in seen:
            continue
        seen.add(k)
        chain, rid, code = k
        if code in AA3:
            continue
        m = (arr.chain_id == chain) & (arr.res_id == rid)
        names = [str(n) for n in arr.atom_name[m] if str(arr.element[m][list(arr.atom_name[m]).index(n)]) != "H"]
        parent = ccd.parent_of(code) if ccd.has_comp(code) else None
        if parent and not bool(arr.hetero[i]):             # modified residue inside a polymer chain
            nm = _mcs_name_map(parent, code)               # parent atom name -> mod atom name
            mapped = set(nm.values())
            ptm_atoms = [n for n in names if n not in mapped and n not in BACKBONE]
            # parent atoms the modification is attached to (OG of SEP, NZ of M3L/ALY, ...) belong to it too
            mol = ccd.comp_mol(code)
            cn = ccd.names(mol)
            anchors = {cn[nb.GetIdx()] for a_ in mol.GetAtoms() if cn[a_.GetIdx()] in ptm_atoms
                       for nb in a_.GetNeighbors() if cn[nb.GetIdx()] in mapped and cn[nb.GetIdx()] not in BACKBONE}
            ptm_atoms = [n for n in names if n in anchors] + ptm_atoms
            side = [n for n in names if n in mapped and n not in BACKBONE and n != "CB" and n not in anchors]
            sites.append(PTMSite(chain, rid, code, "replace", ptm_atoms, side))
        elif bool(arr.hetero[i]):                          # covalent ligand
            anchor = None
            idx = set(np.where(m)[0])
            if arr.bonds is not None:
                for a, b, _t in arr.bonds.as_array():
                    for u, v in ((int(a), int(b)), (int(b), int(a))):
                        if u in idx and v not in idx:
                            anchor = (str(arr.chain_id[v]), int(arr.res_id[v]), str(arr.atom_name[v]))
            sites.append(PTMSite(chain, rid, code, "adduct", names, [], anchor))
    return sites


def _residue_atoms(arr, chain, rid):
    m = (arr.chain_id == chain) & (arr.res_id == rid)
    return arr[m]


def hotspot_dict(arr: struc.AtomArray, sites: list[PTMSite], mode: str = "ptm",
                 n_neighbors: int = 3, neighbor_cutoff: float = 6.0) -> dict[str, str]:
    """RFD3 `select_hotspots` for the given sites.

    mode 'ptm'      : the modification atoms only (selectivity-oriented: binder must touch the PTM)
         'ptm_site' : modification atoms + the modified residue's side chain
         'patch'    : 'ptm_site' + ALL atoms of the `n_neighbors` surface residues closest to the PTM
    Keys use PREPARED chain letters / residue numbers (= the numbers in the contig).
    """
    hs: dict[str, str] = {}
    for s in sites:
        if s.kind == "replace":
            atoms = list(s.ptm_atoms) + (list(s.side_atoms) if mode in ("ptm_site", "patch") else [])
            if not atoms:                                 # e.g. pure-parent atom swaps: fall back to side chain
                atoms = list(s.side_atoms) or ["CB"]
        else:
            atoms = list(s.ptm_atoms)
        hs[s.key] = ",".join(atoms)
    if mode == "patch":
        pts = np.vstack([_residue_atoms(arr, s.chain, s.res_id).coord for s in sites]) if sites else None
        if pts is not None:
            site_keys = {(s.chain, s.res_id) for s in sites}
            ca = arr[(arr.atom_name == "CA")]
            cand = []
            for c, r, x in zip(ca.chain_id, ca.res_id, ca.coord):
                if (str(c), int(r)) in site_keys:
                    continue
                ra = _residue_atoms(arr, str(c), int(r))
                d = np.linalg.norm(ra.coord[:, None, :] - pts[None], axis=-1).min()
                if d <= neighbor_cutoff:
                    n_env = int((np.linalg.norm(ca.coord - x, axis=1) < 10.0).sum())    # crowding ~ burial
                    cand.append((n_env, d, str(c), int(r)))
            for _n, _d, c, r in sorted(cand)[:n_neighbors]:
                hs[f"{c}{r}"] = "ALL"
    return hs


def hbond_acceptors(arr: struc.AtomArray, sites: list[PTMSite]) -> dict[str, str]:
    """Terminal O of the modification (phosphate O1P-O3P, acetyl O, sulfate O, glycan O...) as `active_acceptor`
    conditioning: tells RFD3 the binder should donate H-bonds to them."""
    out: dict[str, str] = {}
    for s in sites:
        m = (arr.chain_id == s.chain) & (arr.res_id == s.res_id)
        names = [str(n) for n, e in zip(arr.atom_name[m], arr.element[m]) if str(e) == "O" and str(n) in s.ptm_atoms]
        names = [n for n in names if n not in ("OG", "OG1", "OH", "OD1")] or names     # bridging ester O is a weak acceptor
        if names:
            out[s.key] = ",".join(names)
    return out


def make_spec(input_path: str, segments_contig: str, hotspots: dict[str, str], ligand_codes: list[str] | None,
              binder_length: str = "60-100", is_non_loopy: bool = True, hbond_acceptor: dict[str, str] | None = None,
              extra: dict | None = None) -> dict:
    spec = {
        "dialect": 2,
        "infer_ori_strategy": "hotspots",
        "input": input_path,
        "contig": f"{binder_length},/0,{segments_contig}",
        "select_hotspots": hotspots,
        "is_non_loopy": is_non_loopy,
    }
    if hbond_acceptor:
        spec["select_hbond_acceptor"] = hbond_acceptor
    if extra:
        spec.update(extra)
    if ligand_codes:
        # RFD3 allows ONE `ligand` field: a comma list of CCD codes/names (each residue on its own chain)
        spec["ligand"] = ",".join(sorted(set(ligand_codes)))
    return spec


def stage_specs(specs_path: str, dest_dir: str) -> str:
    """Copy every spec's input structure to dest_dir (a path visible to compute nodes) and rewrite the paths."""
    import json
    import shutil
    from pathlib import Path
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    specs = json.load(open(specs_path))
    for name, sp in specs.items():
        src = Path(sp["input"])
        dst = d / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy(src, dst)
        sp["input"] = str(dst)
    out = d / "specs.json"
    out.write_text(json.dumps(specs, indent=1))
    return str(out)


def parent_epitope_atoms(site: PTMSite) -> list[str]:
    """Atoms of the unmodified parent residue that correspond to the PTM epitope (anchor + side chain)."""
    anchors = [n for n in site.ptm_atoms if n in set(ccd.heavy_names(ccd.parent_of(site.code) or site.code))]
    return anchors + list(site.side_atoms)
