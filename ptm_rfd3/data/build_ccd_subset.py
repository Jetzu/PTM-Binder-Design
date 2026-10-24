"""Extract the CCD entries ptm_rfd3 needs from the wwPDB components.cif mirror.

Keeps (a) every amino-acid-like component whose `mon_nstd_parent_comp_id` is one of the 20
canonical residues (all modified residues: SEP, TPO, PTR, ALY, M3L, CIR, HYP, ...) and
(b) a curated list of adduct ligands (glycans, lipids, cofactors) that attach covalently.

Output: ccd_subset.json.gz  {code: {name,type,parent,charge,atoms:[...],bonds:[...]}}

Usage: python build_ccd_subset.py [components.cif] [out.json.gz]
"""
from __future__ import annotations

import gzip
import json
import os
import shlex
import sys
from pathlib import Path

DEFAULT_CCD = os.environ.get("CCD_COMPONENTS_CIF", "components.cif")   # wwPDB Chemical Component Dictionary (https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz)
STD20 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
# Non-residue components that PTMs attach as ligands (glycans, lipids, cofactors, ...).
ADDUCT_CODES = {
    # glycans
    "NAG", "NDG", "NGA", "A2G", "MAN", "BMA", "FUC", "FUL", "GAL", "GLC", "BGC", "SIA", "XYP", "RAM",
    "NDG", "BGC", "GCS", "GCU", "IDR", "BDP", "GLA",
    # lipids / prenyl / fatty acyl
    "PLM", "MYR", "FAR", "GRG", "STE", "OLA", "LAU", "DAO", "PLP",
    # cofactors / small modifiers
    "BTN", "LPA", "APR", "AR6", "AMP", "ADP", "UMP", "PNS", "FMN", "FAD", "HEC", "HEM", "RET", "PLR",
    "UDP", "GDP", "CMP", "NAD", "TPQ",
    # inorganic groups used as covalent LIGANDS (phosphate / sulfate as a ligand on Tyr/Ser/Thr OH: RFD3 conditions ligands better than atomised residues)
    "PO4", "SO4",
    # ubiquitin-like tail / amino acids occurring as isopeptide partners
    "GLY", "ALA", "SER", "LYS",
}
ATOM_COLS = (
    "comp_id atom_id alt_atom_id type_symbol charge pdbx_align pdbx_aromatic_flag "
    "pdbx_leaving_atom_flag pdbx_stereo_config pdbx_backbone_atom_flag pdbx_n_terminal_atom_flag "
    "pdbx_c_terminal_atom_flag model_Cartn_x model_Cartn_y model_Cartn_z pdbx_model_Cartn_x_ideal "
    "pdbx_model_Cartn_y_ideal pdbx_model_Cartn_z_ideal pdbx_component_atom_id pdbx_component_comp_id "
    "pdbx_ordinal"
).split()


def _blocks(path: str):
    cur: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("data_"):
                if cur:
                    yield cur
                cur = [line]
            else:
                cur.append(line)
    if cur:
        yield cur


def _header(lines: list[str]) -> dict:
    kv = {}
    for ln in lines:
        if ln.startswith("_chem_comp."):
            parts = ln.split(None, 1)
            if len(parts) == 2:
                kv[parts[0][len("_chem_comp."):]] = parts[1].strip().strip('"').strip("'")
        elif ln.startswith("loop_") or ln.startswith("_chem_comp_atom"):
            break
    return kv


def _loop_rows(lines: list[str], prefix: str):
    """Return (columns, rows) of the loop whose columns start with `prefix`."""
    cols: list[str] = []
    rows: list[list[str]] = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].startswith("loop_") and i + 1 < n and lines[i + 1].startswith(prefix):
            i += 1
            while i < n and lines[i].startswith("_"):
                cols.append(lines[i].strip().split(".", 1)[1])
                i += 1
            while i < n and not lines[i].startswith("#") and not lines[i].startswith("loop_"):
                if lines[i].strip():
                    rows.append(shlex.split(lines[i], posix=True))
                i += 1
            return cols, rows
        i += 1
    return cols, rows


def parse_block(lines: list[str]) -> dict | None:
    hdr = _header(lines)
    cid = hdr.get("id") or lines[0][5:].strip()
    cols, rows = _loop_rows(lines, "_chem_comp_atom.")
    if not rows:
        return None
    ci = {c: k for k, c in enumerate(cols)}
    atoms = []
    for r in rows:
        def f(name, default="?"):
            return r[ci[name]] if name in ci and ci[name] < len(r) else default

        def num(a, b):
            va, vb = f(a), f(b)
            try:
                return float(va)
            except ValueError:
                try:
                    return float(vb)
                except ValueError:
                    return None

        ideal = [num("pdbx_model_Cartn_x_ideal", "model_Cartn_x"),
                 num("pdbx_model_Cartn_y_ideal", "model_Cartn_y"),
                 num("pdbx_model_Cartn_z_ideal", "model_Cartn_z")]
        atoms.append({
            "name": f("atom_id"),
            "el": f("type_symbol").capitalize(),
            "chg": int(f("charge", "0")) if f("charge", "0").lstrip("-").isdigit() else 0,
            "arom": f("pdbx_aromatic_flag") == "Y",
            "leave": f("pdbx_leaving_atom_flag") == "Y",
            "bb": f("pdbx_backbone_atom_flag") == "Y",
            "xyz": ideal if None not in ideal else None,
        })
    bcols, brows = _loop_rows(lines, "_chem_comp_bond.")
    bi = {c: k for k, c in enumerate(bcols)}
    bonds = []
    for r in brows:
        bonds.append([r[bi["atom_id_1"]], r[bi["atom_id_2"]], r[bi["value_order"]],
                      r[bi["pdbx_aromatic_flag"]] == "Y"])
    return {
        "id": cid,
        "name": hdr.get("name", ""),
        "type": hdr.get("type", ""),
        "parent": hdr.get("mon_nstd_parent_comp_id", "?"),
        "charge": hdr.get("pdbx_formal_charge", "0"),
        "atoms": atoms,
        "bonds": bonds,
    }


def main(ccd: str = DEFAULT_CCD, out: str | None = None):
    out = out or str(Path(__file__).with_name("ccd_subset.json.gz"))
    keep: dict[str, dict] = {}
    n_seen = 0
    for blk in _blocks(ccd):
        n_seen += 1
        hdr = _header(blk)
        cid = blk[0][5:].strip()
        parent = hdr.get("mon_nstd_parent_comp_id", "?").upper()
        ctype = hdr.get("type", "").upper()
        want = cid in ADDUCT_CODES or cid in STD20 or (
            parent in STD20 and "PEPTIDE" in ctype and not ctype.startswith("D-"))
        if not want:
            continue
        rec = parse_block(blk)
        if rec:
            keep[cid] = rec
    with gzip.open(out, "wt") as fh:
        json.dump(keep, fh, separators=(",", ":"))
    print(f"scanned {n_seen} components, kept {len(keep)} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
