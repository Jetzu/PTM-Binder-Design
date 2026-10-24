"""AlphaFold3 (AlphaFast SIF) input JSON for binder + PTM-bearing target co-folding.

PTMs on polymer residues -> `modifications` (CCD ptmType at the 1-based position in the AF3 chain);
covalent ligands (glycans, lipids, ...) -> `ligand` entries with `bondedAtomPairs` to the anchor atom.
Verified schema/behaviour: .agent-research/results/N001-*/reader_af3_mpnn.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from biotite import structure as struc

from . import ccd

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
# CCD codes AF3 maps to X in the query sequence (reader B: KCR, BTK, DA2 ...). The MSA row must match.
AF3_X_CODES = {"KCR", "BTK", "DA2"}
# Where AF3's modified-sequence alphabet differs from the parent letter (AlphaFast residue_names.py:175)
AF3_QUERY_LETTER = {"PCA": "E"}
AF3_PARENT_MAP_KNOWN = {"SEP": "S", "TPO": "T", "PTR": "Y", "TYS": "Y", "ALY": "K", "MLY": "K", "M2L": "K",
                        "M3L": "K", "MLZ": "K", "CIR": "R", "HYP": "P", "MSE": "M", "CSO": "C"}
PROT_IDS = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
LIG_IDS = "ZYXWVUTSRQPONMLKJIHGFEDCBA"


@dataclass
class TargetEncoding:
    chains: dict                 # af3 chain id -> {"sequence", "query_sequence", "mods":[(pos, code)], "src": [(chain,res_id,ins)]}
    ligands: list = field(default_factory=list)   # {"id", "ccd":[...], "src_res": [(chain,res_id,ins)] in component order}
    bonds: list = field(default_factory=list)     # [[chain,pos,atom],[chain,pos,atom]] (pos = residue / component index)
    chain_map: dict = field(default_factory=dict)  # (orig chain) -> af3 chain id
    lig_comp: dict = field(default_factory=dict)   # (chain,res_id,ins) -> (af3 ligand id, 1-based component index)


def _letter(code: str) -> tuple[str, str | None]:
    """(letter in canonical sequence, ptmType or None)."""
    if code in AA3:
        return AA3[code], None
    p = ccd.parent_of(code) if ccd.has_comp(code) else None
    return (AA3.get(p, "X") if p else "X"), code


def encode_target(arr: struc.AtomArray, first_chain_id: str = "B") -> TargetEncoding:
    """Translate a (PTM-bearing) target AtomArray into AF3 chains / modifications / bonded pairs.

    Covalently attached non-polymer residues that are connected to each other (glycan trees) become ONE AF3
    ligand with several ccdCodes (breadth-first from the residue bonded to the protein); inter-component and
    protein-ligand bonds go to `bonds` as [chain, residue-or-component index, atom].
    """
    enc = TargetEncoding(chains={})
    prot_ids = PROT_IDS[PROT_IDS.index(first_chain_id):]
    # polymer residues = amino-acid-like comps, in order of appearance, grouped by chain
    prot: dict[str, list] = {}
    nonpoly: list[tuple] = []
    seen = set()
    atoms_of: dict[tuple, list[int]] = {}
    for i in range(arr.array_length()):
        k = (str(arr.chain_id[i]), int(arr.res_id[i]), str(arr.ins_code[i]).strip(), str(arr.res_name[i]))
        atoms_of.setdefault(k, []).append(i)
        if k in seen:
            continue
        seen.add(k)
        code = k[3]
        is_aa = code in AA3 or (ccd.has_comp(code) and ccd.parent_of(code) in AA3
                                and "PEPTIDE" in ccd.get_comp(code)["type"].upper())
        if is_aa:
            prot.setdefault(k[0], []).append(k)
        else:
            nonpoly.append(k)
    if len(prot) > len(prot_ids):
        raise ValueError(f"{len(prot)} protein chains exceed the {len(prot_ids)} AF3 chain ids available")
    for ch, ress in prot.items():
        cid = prot_ids[len(enc.chains)]
        enc.chain_map[ch] = cid
        seq, query, mods, src = "", "", [], []
        for pos, (c, rid, ins, code) in enumerate(ress, start=1):
            letter, ptm = _letter(code)
            seq += letter
            query += "X" if (code in AF3_X_CODES) else AF3_QUERY_LETTER.get(code, letter)
            if ptm:
                mods.append((pos, ptm))
            src.append((c, rid, ins))
        enc.chains[cid] = {"sequence": seq, "query_sequence": query, "mods": mods, "src": src}
    if arr.bonds is None:
        return enc
    pos_of = {}
    for cid, d in enc.chains.items():
        for p, (c, rid, ins) in enumerate(d["src"], start=1):
            pos_of[(c, rid, ins)] = (cid, p)
    res_of_atom = {}
    for k, idx in atoms_of.items():
        for i in idx:
            res_of_atom[i] = k
    # adjacency among non-polymer residues + links to polymer residues (atom-level bond list kept for later)
    npset = set(nonpoly)
    adj: dict[tuple, set] = {k: set() for k in nonpoly}
    root_links = []     # (poly_key3, atom_p, np_key, atom_l)
    cross = []          # (np_key_a, atom_a, np_key_b, atom_b)
    for i, j, t in arr.bonds.as_array():
        i, j = int(i), int(j)
        if int(t) == int(struc.BondType.COORDINATION):
            continue
        ki, kj = res_of_atom.get(i), res_of_atom.get(j)
        if ki is None or kj is None or ki == kj:
            continue
        for (u, ku), (v, kv) in (((i, ki), (j, kj)), ((j, kj), (i, ki))):
            if kv in npset and (ku[0], ku[1], ku[2]) in pos_of and ku not in npset:
                root_links.append((ku[:3], str(arr.atom_name[u]), kv, str(arr.atom_name[v])))
        if ki in npset and kj in npset:
            adj[ki].add(kj)
            adj[kj].add(ki)
            cross.append((ki, str(arr.atom_name[i]), kj, str(arr.atom_name[j])))
    lig_ids = iter(LIG_IDS)
    used = set(enc.chains) | {"A"}                                # "A" is the binder
    done: set = set()
    for k0, pa, k_root, la in root_links:
        if k_root in done:
            continue
        order, queue = [], [k_root]
        seen_c = {k_root}
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for nb in sorted(adj[cur]):
                if nb not in seen_c:
                    seen_c.add(nb)
                    queue.append(nb)
        lid = next((x for x in lig_ids if x not in used), None)
        if lid is None:
            raise ValueError("no free AF3 chain id left for the covalent ligand(s); crop the target")
        used.add(lid)
        enc.ligands.append({"id": lid, "ccd": [k[3] for k in order], "src_res": [k[:3] for k in order]})
        for n, k in enumerate(order, start=1):
            enc.lig_comp[k[:3]] = (lid, n)
            done.add(k)
    for k0, pa, k_root, la in root_links:
        cid, p = pos_of[k0]
        lid, n = enc.lig_comp[k_root[:3]]
        enc.bonds.append([[cid, p, pa], [lid, n, la]])
    dup = set()
    for ka, aa, kb, ab in cross:
        if ka[:3] in enc.lig_comp and kb[:3] in enc.lig_comp:
            (l1, n1), (l2, n2) = enc.lig_comp[ka[:3]], enc.lig_comp[kb[:3]]
            key = tuple(sorted(((l1, n1, aa), (l2, n2, ab))))
            if key not in dup:
                dup.add(key)
                enc.bonds.append([[l1, n1, aa], [l2, n2, ab]])
    return enc


def make_af3_json(name: str, binder_seq: str, enc: TargetEncoding, seeds=(1,),
                  target_msa: dict[str, str] | None = None, binder_id: str = "A") -> dict:
    """Binder (single-sequence MSA) + target chains (+MSA if given) + covalent ligands."""
    def prot(cid, seq, query, mods=None, msa=None):
        m = msa or f">query\n{query}\n"
        d = {"id": [cid], "sequence": seq, "unpairedMsa": m, "pairedMsa": f">query\n{query}\n", "templates": []}
        if mods:
            d["modifications"] = [{"ptmType": c, "ptmPosition": p} for p, c in mods]
        return {"protein": d}
    seqs = [prot(binder_id, binder_seq, binder_seq)]
    for cid, d in enc.chains.items():
        msa = None if not target_msa else target_msa.get(cid)
        seqs.append(prot(cid, d["sequence"], d["query_sequence"], d["mods"], msa))
    for L in enc.ligands:
        seqs.append({"ligand": {"id": [L["id"]], "ccdCodes": L["ccd"]}})
    out = {"name": name, "dialect": "alphafold3", "version": 1, "modelSeeds": list(seeds), "sequences": seqs}
    custom = sorted({c for L in enc.ligands for c in L["ccd"] if ccd.is_custom(c)})
    if custom:                                                   # SMILES-defined groups are not in AF3's CCD: ship their definition
        out["userCCD"] = "".join(ccd.user_ccd(c) for c in custom)
    if enc.bonds:
        out["bondedAtomPairs"] = enc.bonds
    return out


def write_af3_json(path, **kw):
    with open(path, "w") as fh:
        json.dump(make_af3_json(**kw), fh)
