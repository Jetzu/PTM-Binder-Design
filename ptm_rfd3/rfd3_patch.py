"""Runtime patch: give stock RFD3 the covalent bonds it drops at inference (no foundry edits).

Finding (N001/reader_rfd3.md, reproduced on CPU at foundry e412591): dialect-2 `build()` loses every
inter-residue bond involving a non-standard residue or a ligand (flanking peptide bonds of an
atomised PTM residue, Asn-NAG, Cys-palmitoyl ...), although the model was TRAINED with them
(atomised Asn, `token_bonds` set by FlagAndReassignCovalentModifications). This module re-adds
those bonds from the loaded source structure to the built design array, so the stock pipeline then
behaves as in training. Dependencies: numpy + biotite only (importable inside rfd3_env).

Use:   python -m ptm_rfd3.rfd3_patch  <the usual hydra args of rfd3.run_inference>
or:    import ptm_rfd3.rfd3_patch as p; p.install()
"""
from __future__ import annotations

import logging
import re
import sys

import numpy as np
from biotite import structure as struc

log = logging.getLogger("ptm_rfd3.rfd3_patch")

STANDARD_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
               "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK"}
_ORIG = {}


def _src_key(arr: struc.AtomArray, i: int, ligand_chains: set[str]):
    """Key under which atom i of the SOURCE array can be found in the BUILT array."""
    return (str(arr.chain_id[i]), int(arr.res_id[i]), str(arr.atom_name[i]))


def _component_index(built, comp, src, src_lig_chains, has_ins) -> dict[tuple[str, str], int]:
    """(full source component 'A1138', atom name) -> built atom index.

    RFD3 stores `src_component` in a fixed-width string column (<U4): 'A1138' is saved as 'A113', so residue numbers of
    four digits (or longer chain+number keys) collide. Truncated keys are resolved by the ORDER of appearance of the
    residues in the source and in the built array, and only accepted when the residue names agree one by one;
    otherwise those residues stay unmapped (never guessed)."""
    out: dict[tuple[str, str], int] = {}
    if comp is None:
        return out
    # the column is created with a width that cuts 'A1138' to 'A113' (observed <U4 at creation, later widened): the
    # truncation length is the longest chain+number key actually present in the built array
    keys_seen = {str(c) for c in set(comp.tolist())}
    lens = [len(c) for c in keys_seen if re.fullmatch(r"[A-Za-z0-9]\d+", c)]
    maxlen = max(lens) if lens else 10 ** 6
    b_groups: dict[str, list[list[int]]] = {}
    b_ids: dict[str, list[tuple[str, int]]] = {}
    seen: dict[tuple, int] = {}
    for k in range(built.array_length()):
        c = str(comp[k])
        if not c:
            continue
        key = (c, str(built.chain_id[k]), int(built.res_id[k]), str(built.ins_code[k]).strip() if "ins_code" in built.get_annotation_categories() else "")
        if key not in seen:
            seen[key] = len(b_groups.setdefault(c, []))
            b_groups[c].append([])
            b_ids.setdefault(c, []).append((str(built.chain_id[k]), int(built.res_id[k])))
        b_groups[c][seen[key]].append(k)
    s_groups: dict[str, list[tuple[str, str, int, str]]] = {}
    s_seen = set()
    for i in range(src.array_length()):
        ch, rid = str(src.chain_id[i]), int(src.res_id[i])
        if ch in src_lig_chains or (ch, rid, ins_of(src, i, has_ins)) in s_seen:
            continue
        s_seen.add((ch, rid, ins_of(src, i, has_ins)))
        full = f"{ch}{rid}"
        s_groups.setdefault(full[:maxlen], []).append((full, str(src.res_name[i]), rid, ins_of(src, i, has_ins)))
    ins_built = "ins_code" in built.get_annotation_categories()
    for trunc, sres in s_groups.items():
        bres = b_groups.get(trunc, [])
        if ins_built and any(str(built.ins_code[b[0]]).strip() for b in bres):
            continue                                               # insertion-coded built residues are never matched
        if len(sres) == 1 and len(bres) == 1:                   # unambiguous (also the non-truncated case)
            pairs = [(sres[0][0], bres[0])]
        elif (len(sres) == len(bres) and not any(t[3] for t in sres)
              and all(a[2] < b[2] for a, b in zip(sres, sres[1:]))
              and all(str(built.res_name[b[0]]) == t[1] for b, t in zip(bres, sres))
              and all((bb[1] - ba[1] == 1) for (ba, bb), (ta, tb) in zip(zip(b_ids[trunc], b_ids[trunc][1:]), zip(sres, sres[1:]))
                      if tb[2] - ta[2] == 1 and ba[0] == bb[0])):
            pairs = [(t[0], b) for t, b in zip(sres, bres)]     # same order, residue names and spacing, no insertion codes
        else:
            continue
        for full, idx in pairs:
            for k in idx:
                out.setdefault((full, str(built.atom_name[k])), k)
    return out


def ins_of(src, i: int, has_ins: bool) -> str:
    return str(src.ins_code[i]).strip() if has_ins else ""


def inject_covalent_bonds(built: struc.AtomArray, src: struc.AtomArray) -> tuple[struc.AtomArray, int, int]:
    """Re-add source inter-residue COVALENT bonds (non-standard residue or ligand involved) to `built`.

    Polymer atoms of `built` are located via annotation `src_component` ('A45' = source chain+res_id).
    Ligand atoms keep their source chain letter (res_id reset to 1) and are located ONLY for source chains
    that consist entirely of non-polymer residues; a polymer endpoint missing from `built` (cropped
    chain) stays unmapped - it must never alias another residue. Coordination bonds and bonds that touch
    an insertion-coded residue (unsupported by RFD3 contigs; `prepare_for_rfd3` removes them) are skipped.
    Returns (built, n_added, n_unmapped).
    """
    if built.bonds is None:
        built.bonds = struc.BondList(built.array_length())
    if src.bonds is None:
        return built, 0, 0
    cats = built.get_annotation_categories()
    comp = built.get_annotation("src_component") if "src_component" in cats else None
    by_lig: dict[tuple[str, str, str], int] = {}
    for k in range(built.array_length()):
        by_lig.setdefault((str(built.chain_id[k]), str(built.res_name[k]), str(built.atom_name[k])), k)
    has_ins = "ins_code" in src.get_annotation_categories()
    # ligand chain = no residue has an amino-acid backbone (N, CA, C). Do NOT use the HETATM flag: modified
    # polymer residues are often HETATM. A ligand chain with several residues cannot be addressed unambiguously
    # (RFD3 itself requires one residue per ligand chain) -> its bonds are reported unmapped, never guessed.
    def _has_backbone(mask) -> bool:
        keys = {}
        for r_, n_ in zip(src.res_id[mask], src.atom_name[mask]):
            keys.setdefault(int(r_), set()).add(str(n_))
        return any({"N", "CA", "C"} <= v for v in keys.values())

    src_lig_chains, multi_res_lig = set(), set()
    for c in np.unique(src.chain_id):
        m = src.chain_id == c
        if not _has_backbone(m):
            src_lig_chains.add(str(c))
            if len(set(zip(src.res_id[m].tolist(), (src.ins_code[m] if has_ins else np.zeros(m.sum())).tolist()))) > 1:
                multi_res_lig.add(str(c))

    def ins(i: int) -> str:
        return str(src.ins_code[i]).strip() if has_ins else ""

    by_comp = _component_index(built, comp, src, src_lig_chains, has_ins)

    def locate(i: int) -> int | None:
        c, r, n = str(src.chain_id[i]), int(src.res_id[i]), str(src.atom_name[i])
        if c in multi_res_lig:
            return None
        if c in src_lig_chains:
            return by_lig.get((c, str(src.res_name[i]), n))
        return by_comp.get((f"{c}{r}", n))

    existing = {tuple(sorted((int(a), int(b)))) for a, b, _ in built.bonds.as_array()}
    added = unmapped = 0
    for i, j, t in src.bonds.as_array():
        i, j, t = int(i), int(j), int(t)
        if t == int(struc.BondType.COORDINATION):
            continue
        ri = (str(src.chain_id[i]), int(src.res_id[i]), ins(i))
        rj = (str(src.chain_id[j]), int(src.res_id[j]), ins(j))
        if ri == rj:
            continue
        if str(src.res_name[i]) in STANDARD_AA and str(src.res_name[j]) in STANDARD_AA:
            continue                                   # canonical polymer links are implicit
        if ri[2] or rj[2]:
            unmapped += 1                              # insertion codes cannot be addressed by RFD3 contigs
            continue
        a, b = locate(i), locate(j)
        if a is None or b is None:
            unmapped += 1
            continue
        key = tuple(sorted((a, b)))
        if key in existing:
            continue
        fixed = built.get_annotation("is_motif_atom_with_fixed_coord") if "is_motif_atom_with_fixed_coord" in cats else None
        if fixed is not None and fixed[a] and fixed[b]:                # both ends keep their source coordinates: the bond length must agree
            d_built = float(np.linalg.norm(built.coord[a] - built.coord[b]))
            d_src = float(np.linalg.norm(src.coord[i] - src.coord[j]))
            if abs(d_built - d_src) > 0.75:
                unmapped += 1                                          # wrong residue pairing: refuse instead of bonding atoms 5-170 A apart
                continue
        built.bonds.add_bond(a, b, t if t in (1, 2, 3, 5, 6, 7) else struc.BondType.SINGLE)
        existing.add(key)
        added += 1
    return built, added, unmapped


def install() -> None:
    """Wrap DesignInputSpecification.build so every spec gets its covalent bonds restored."""
    from rfd3.inference import input_parsing as ip

    cls = ip.DesignInputSpecification
    if "build" in _ORIG:
        return
    _ORIG["build"] = cls.build

    def build(self, return_metadata=False):
        out = _ORIG["build"](self, return_metadata=return_metadata)
        aa, meta = (out if return_metadata else (out, None))
        src = getattr(self, "atom_array_input", None)
        if src is not None:
            aa, n_add, n_miss = inject_covalent_bonds(aa, src)
            if n_add or n_miss:
                log.warning("ptm_rfd3 patch: restored %d covalent bond(s), %d unmapped", n_add, n_miss)
            if meta is not None:
                meta.setdefault("extra", {})["ptm_rfd3_bonds_restored"] = int(n_add)
        return (aa, meta) if return_metadata else aa

    cls.build = build
    log.warning("ptm_rfd3: installed covalent-bond restoration on DesignInputSpecification.build")


def preprocess_argv(argv: list[str]) -> list[str]:
    """If `inputs=<spec file>` uses PTM keys (`ptm`, ...), build the modified targets once and point RFD3 at the
    expanded plain spec file (see ptm_rfd3.rfd3_run). Everything else is passed through unchanged."""
    from pathlib import Path

    from . import rfd3_run

    out_dir = next((a.split("=", 1)[1] for a in argv if a.startswith("out_dir=")), ".")
    argv = list(argv)
    for i, a in enumerate(argv):
        if a.startswith("inputs="):
            new = rfd3_run.expand_inputs(a.split("=", 1)[1].strip("'\""), Path(out_dir.strip("'\"")) / "ptm_build")
            if new:
                argv[i] = f"inputs={new}"
    return argv


def main() -> None:
    import os
    sys.argv = [sys.argv[0]] + preprocess_argv(sys.argv[1:])      # PTM building happens BEFORE RFD3 starts
    if os.environ.get("PTM_RFD3_NO_PATCH") != "1":                # =1 only for ablation benchmarks: stock bond handling
        install()
    from rfd3.run_inference import run_inference

    run_inference()


if __name__ == "__main__":
    main()
