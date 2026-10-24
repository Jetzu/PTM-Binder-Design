"""Normalise a (PTM-bearing) target so stock RFD3 loads it exactly as intended.

Facts this encodes (see .agent-research/results/N001-*/reader_rfd3.md, all reproduced on CPU):
* the contig parser takes the FIRST character as chain id, needs gap-free residue numbers per range
  and no insertion codes  -> protein chains are split at gaps and renumbered 1..n per segment;
* a HETATM residue inside a polymer chain is moved to a new chain (PDB path) -> polymer residues,
  PTMs included, are written as ATOM;
* every covalent ligand residue needs its own single-letter chain id that never collides with the
  chain letters RFD3 gives its output (binder=A, target segments B, C, ... in contig order)
  -> ligand chains are taken from the end of the alphabet;
* the covalent link must be a `_struct_conn` covale row (written by biotite from the BondList);
* NaN coordinates crash the run -> checked here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from biotite import structure as struc

from . import ccd
from . import structio as sio
from .af3_inputs import AA3

LIG_CHAIN_POOL = "ZYXWVUTSRQPONM"
LIG_CHAIN_POOL_FULL = "ZYXWVUTSRQPONMLKJIHGFEDCBA"


@dataclass
class PreparedTarget:
    atoms: struc.AtomArray                       # normalised target (chains A.. segments, Z.. ligands)
    seg_map: dict                                # (orig_chain, orig_res_id, orig_ins) -> (new_chain, new_res_id)
    segments: list[tuple[str, int, int]]         # (new_chain, first_res_id, last_res_id) in contig order
    ligand_chains: list[str] = field(default_factory=list)
    lig_map: dict = field(default_factory=dict)  # (orig_chain, orig_res_id, orig_ins) -> new ligand chain id
    id_offset: dict = field(default_factory=dict)  # chain -> amount subtracted from residue ids (compact_ids)

    def contig_target(self) -> str:
        """Target part of an RFD3 contig, e.g. 'A1-120,/0,B1-30'."""
        return ",/0,".join(f"{c}{a}-{b}" for c, a, b in self.segments)

    def key(self, chain: str, res_id: int, ins: str = "") -> tuple[str, int]:
        return self.seg_map[(chain, int(res_id), ins)]


def _is_polymer_residue(code: str) -> bool:
    if code in AA3:
        return True
    return ccd.has_comp(code) and ccd.parent_of(code) in AA3 and "PEPTIDE" in ccd.get_comp(code)["type"].upper()


def _rkey(arr, i):
    return (str(arr.chain_id[i]), int(arr.res_id[i]), str(arr.ins_code[i]).strip(), str(arr.res_name[i]))


def prepare_for_rfd3(arr: struc.AtomArray, keep_chains: list[str] | None = None,
                     max_gap_ca: float = 4.5, drop_other_hetero: bool = True,
                     renumber: bool = True, compact_ids: bool = False) -> PreparedTarget:
    """`renumber=False` keeps every polymer residue's ORIGINAL chain letter and residue number (the user's contigs and
    selections stay valid); it requires single-letter chain ids and no insertion codes, and only the ligand/glycan
    residues get new (single-residue) chain letters from the end of the alphabet.
    `compact_ids=True` (with renumber=False) additionally shifts the residue ids of a chain so that its first residue is 1 whenever
    `chain+id` would be longer than 4 characters: RFD3 keeps the source key of every residue in a 4-character column ('A1138' ->
    'A113'), which makes such keys ambiguous. The shift is returned in `id_offset` (translate your contigs with it)."""
    if not np.isfinite(arr.coord).all():
        raise ValueError("target has NaN/inf coordinates (RFD3 crashes on them)")
    a = arr[(arr.element != "H")]
    if keep_chains:
        a = a[np.isin(a.chain_id, keep_chains) | ~np.array([_is_polymer_residue(r) for r in a.res_name])]
    # residue table in order of appearance; identity INCLUDES the insertion code
    res_keys, seen, atoms_of = [], set(), {}
    for i in range(a.array_length()):
        k = _rkey(a, i)
        atoms_of.setdefault(k, []).append(i)
        if k not in seen:
            seen.add(k)
            res_keys.append(k)
    poly = [k for k in res_keys if _is_polymer_residue(k[3])]
    nonpoly = [k for k in res_keys if not _is_polymer_residue(k[3])]

    def atom_xyz(k, name):
        sel = [i for i in atoms_of[k] if str(a.atom_name[i]) == name]
        return a.coord[sel[0]] if sel else None

    # --- protein segments: split at gaps in numbering or backbone C-N distance
    segs: list[list] = []
    for k in poly:
        if segs and segs[-1][-1][0] == k[0]:
            prev = segs[-1][-1]
            c, n = atom_xyz(prev, "C"), atom_xyz(k, "N")
            near = c is not None and n is not None and np.linalg.norm(c - n) < 2.1
            consecutive = (k[1] - prev[1] == 1 and not k[2] and not prev[2])
            if (near or (consecutive and (c is None or n is None))):
                segs[-1].append(k)
                continue
        segs.append([k])
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    seg_map, segments = {}, []
    id_offset: dict[str, int] = {}
    if renumber:
        for sg in segs:
            cl = next(letters)
            for j, k in enumerate(sg, start=1):
                seg_map[(k[0], k[1], k[2])] = (cl, j)
            segments.append((cl, 1, len(sg)))
    else:
        bad = [f"{k[0]}{k[1]}{k[2]}" for k in poly if len(k[0]) != 1 or k[2]]
        if bad:
            raise ValueError("numbering-preserving preparation needs single-letter chain ids and no insertion codes; "
                             f"offending residues: {bad[:8]}{' ...' if len(bad) > 8 else ''}")
        if compact_ids:
            neg = [f"{k[0]}{k[1]}" for k in poly if k[1] <= 0]
            if neg and any(len(f"{k[0]}{k[1]}") > 4 for k in poly):
                raise ValueError(f"residue ids <= 0 ({neg[:4]}) cannot be combined with id compaction; renumber the input first")
            first: dict[str, int] = {}
            for k in poly:
                first[k[0]] = min(first.get(k[0], k[1]), k[1])
            for c, mn in first.items():
                if any(len(f"{c}{k[1]}") > 4 for k in poly if k[0] == c):
                    id_offset[c] = mn - 1
            too_long = [f"{c}{k[1] - id_offset.get(c, 0)}" for k in poly for c in [k[0]] if len(f"{c}{k[1] - id_offset.get(c, 0)}") > 4]
            if too_long:
                raise ValueError(f"chain+residue keys longer than 4 characters even after shifting: {too_long[:5]}; split the chain or crop the input")
        for k in poly:
            seg_map[(k[0], k[1], k[2])] = (k[0], k[1] - id_offset.get(k[0], 0))
        by_chain: dict[str, list[int]] = {}
        for k in poly:
            by_chain.setdefault(k[0], []).append(k[1])
        segments = [(c, min(r) - id_offset.get(c, 0), max(r) - id_offset.get(c, 0)) for c, r in by_chain.items()]

    # --- covalent ligands: everything non-polymer connected (through bonds) to a kept polymer residue,
    #     so glycan TREES stay whole; every ligand residue gets its own chain letter (RFD3 rule)
    atom_res = {}
    for k, idx in atoms_of.items():
        for i in idx:
            atom_res[i] = k
    adj: dict[tuple, set] = {}
    if a.bonds is not None:
        for i, j, t in a.bonds.as_array():
            if int(t) == int(struc.BondType.COORDINATION):
                continue                                   # metal coordination is not a covalent PTM link
            ki, kj = atom_res.get(int(i)), atom_res.get(int(j))
            if ki is None or kj is None or ki == kj:
                continue
            adj.setdefault(ki, set()).add(kj)
            adj.setdefault(kj, set()).add(ki)
    poly_keys = {(k[0], k[1], k[2]) for k in poly if (k[0], k[1], k[2]) in seg_map}
    # lone ions are coordination, not covalent PTMs - unless they carry an explicit covalent bond (e.g. Hg-S)
    np_set = {k for k in nonpoly if len(atoms_of[k]) >= 2 or adj.get(k)}
    keep_np, frontier = [], []
    for k in nonpoly:
        if k in np_set and any((n[0], n[1], n[2]) in poly_keys and n not in np_set for n in adj.get(k, ())):
            frontier.append(k)
    seen_np = set(frontier)
    while frontier:
        cur = frontier.pop(0)
        keep_np.append(cur)
        for n in sorted(adj.get(cur, ())):
            if n in np_set and n not in seen_np:
                seen_np.add(n)
                frontier.append(n)
    if not drop_other_hetero:
        keep_np += [k for k in nonpoly if k in np_set and k not in seen_np]
    n_out = 1 + len(segments)                                   # binder A + target segments B..
    if renumber and n_out + len(keep_np) > 26:
        raise ValueError(f"{len(segments)} protein segments + {len(keep_np)} ligand residues exceed the 26 "
                         "chain letters RFD3 can address; crop the target (--crop-radius) or keep fewer chains")
    used = {c for c, _r in seg_map.values()}                      # polymer chain letters stay untouched
    free = [c for c in LIG_CHAIN_POOL_FULL if c not in used]
    if len(keep_np) > len(free):
        raise ValueError(f"{len(keep_np)} ligand residues need their own chain letter but only {len(free)} are free; "
                         "crop the target or drop chains")
    pool = iter(free)
    lig_map = {k: next(pool) for k in keep_np}                   # (chain,res_id,ins,res_name) -> new chain

    # --- assemble
    out_idx, new_chain, new_res = [], [], []
    for i in range(a.array_length()):
        k = atom_res[i]
        if _is_polymer_residue(k[3]) and (k[0], k[1], k[2]) in seg_map:
            c, r = seg_map[(k[0], k[1], k[2])]
        elif k in lig_map:
            c, r = lig_map[k], 1
        else:
            continue
        out_idx.append(i)
        new_chain.append(c)
        new_res.append(r)
    out = a[np.array(out_idx)]
    out.chain_id = np.array(new_chain)
    out.res_id = np.array(new_res)
    out.ins_code = np.array([""] * out.array_length())
    out.hetero = np.array([not _is_polymer_residue(str(r)) for r in out.res_name])
    seg_letters = [s_[0] for s_ in segments]
    order = np.argsort(np.array([("0" if c in seg_letters else "1") + c for c in out.chain_id]), kind="stable")
    if not renumber:                                            # keep the input's chain order, ligands last
        order = np.argsort(np.array([("0" if c in seg_letters else "1") for c in out.chain_id]), kind="stable")
    out = out[order]
    return PreparedTarget(atoms=out, seg_map=seg_map, segments=segments,
                          ligand_chains=list(lig_map.values()),
                          lig_map={(k[0], k[1], k[2]): v for k, v in lig_map.items()},
                          id_offset=dict(id_offset))


def crop_around(arr: struc.AtomArray, center_xyz: np.ndarray, radius: float, pad: int = 2,
                protect: set[tuple[str, int]] | None = None) -> struc.AtomArray:
    """Keep polymer residues with any atom within `radius` of center_xyz (+-pad sequence neighbours so
    fragments stay contiguous); ligands and `protect`ed residues are always kept."""
    keep = np.zeros(arr.array_length(), bool)
    chains = {}
    for i in range(arr.array_length()):
        k = (str(arr.chain_id[i]), int(arr.res_id[i]))
        chains.setdefault(k, []).append(i)
    near = set()
    for k, idx in chains.items():
        if _is_polymer_residue(str(arr.res_name[idx[0]])):
            d = np.linalg.norm(arr.coord[idx][:, None, :] - center_xyz[None], axis=-1).min()
            if d <= radius:
                near.add(k)
    polymer_order: dict[str, list[int]] = {}
    for (c, r), idx in chains.items():
        if _is_polymer_residue(str(arr.res_name[idx[0]])):
            polymer_order.setdefault(c, []).append(r)
    sel = set(protect or ())
    for c, rids in polymer_order.items():
        rids = sorted(rids)
        pos = {r: n for n, r in enumerate(rids)}
        for r in [r for r in rids if (c, r) in near]:
            for n in range(max(0, pos[r] - pad), min(len(rids), pos[r] + pad + 1)):
                sel.add((c, rids[n]))
    for k, idx in chains.items():
        if k in sel or not _is_polymer_residue(str(arr.res_name[idx[0]])):
            keep[idx] = True
    return arr[keep]
