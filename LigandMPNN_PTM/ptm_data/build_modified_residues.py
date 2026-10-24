"""Build ptm_data/modified_residues.json from the wwPDB Chemical Component Dictionary.

Usage: python build_modified_residues.py components.cif.gz ptm_data/modified_residues.json
(components.cif.gz: https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz)

Keeps every L-peptide-linking component whose parent (mon_nstd_parent_comp_id) is one of the
20 standard amino acids. Output: {CCD_id: {"parent": "SER", "name": "PHOSPHOSERINE"}}.
"""
import gzip
import json
import shlex
import sys

STANDARD = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
# natural but non-canonical residues the CCD lists without a parent
MANUAL = {
    "SEC": {"parent": "CYS", "name": "SELENOCYSTEINE"},
    "PYL": {"parent": "LYS", "name": "PYRROLYSINE"},
}
KEEP_TYPES = {"L-PEPTIDE LINKING", "PEPTIDE LINKING", "L-PEPTIDE COOH CARBOXY TERMINUS",
              "L-PEPTIDE NH3 AMINO TERMINUS"}


def main(src, dst):
    out, cur = {}, {}
    with gzip.open(src, "rt") as fh:
        for line in fh:
            if line.startswith("data_"):
                _flush(cur, out)
                cur = {}
            elif line.startswith("_chem_comp."):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key = parts[0][len("_chem_comp."):]
                    if key in ("id", "name", "type", "mon_nstd_parent_comp_id"):
                        cur[key] = shlex.split(parts[1].strip())[0] if parts[1].strip() else ""
            elif line.startswith("loop_") or line.startswith("_chem_comp_atom"):
                pass
    _flush(cur, out)
    out.update(MANUAL)
    with open(dst, "w") as fh:
        json.dump(dict(sorted(out.items())), fh, indent=0, sort_keys=False)
    print(len(out), "modified residues ->", dst)


def _flush(cur, out):
    if not cur or "id" not in cur:
        return
    parent = cur.get("mon_nstd_parent_comp_id", "?")
    if cur.get("type", "").upper() not in KEEP_TYPES or cur["id"] in STANDARD:
        return
    # a few entries list several parents ("SER,THR"); keep only unambiguous ones
    if parent in STANDARD:
        out[cur["id"]] = {"parent": parent, "name": cur.get("name", "")}


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
