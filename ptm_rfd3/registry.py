"""Named post-translational modifications -> CCD construction recipes.

`replace` entries swap the residue for a CCD modified amino acid (parent atoms stay in place);
`adduct` entries attach a CCD ligand covalently to an anchor atom of the residue.

Spec grammar used by the CLI (`--ptm`):
    A45=phospho-Ser          registry name (case/underscore-insensitive), parent type is checked
    A45=SEP                  CCD code of a modified amino acid (replace)
    A120=N-GlcNAc            registry adduct
    A120=adduct:NAG@ND2:C1   generic adduct  LIGAND@ANCHOR_ATOM:LIGAND_ATOM[-LEAVING1,LEAVING2]  (LIGAND = any CCD code of the shipped subset)
    A9=smiles:*C(C)=O|NZ|ZAC  custom group from SMILES: `*` = attachment point, then the residue atom it binds to, optional 3-char code
"""
from __future__ import annotations

from dataclasses import dataclass

from . import ccd
from .builder import AdductSpec


@dataclass(frozen=True)
class Replace:
    parent: str
    code: str
    note: str = ""


@dataclass(frozen=True)
class Adduct:
    parent: tuple[str, ...]
    spec: AdductSpec
    note: str = ""


REGISTRY: dict[str, Replace | Adduct] = {
    # --- phosphorylation
    "phospho-Ser": Replace("SER", "SEP"),
    "phospho-Thr": Replace("THR", "TPO"),
    "phospho-Tyr": Replace("TYR", "PTR"),
    "phospho-His-N1": Replace("HIS", "NEP", "1-phosphohistidine (pros)"),
    "phospho-His-N3": Replace("HIS", "HIP", "3-phosphohistidine (tele)"),
    "phospho-Asp": Replace("ASP", "PHD"),
    # --- lysine acylation / methylation
    "acetyl-Lys": Replace("LYS", "ALY"),
    "crotonyl-Lys": Replace("LYS", "KCR"),
    "methyl-Lys": Replace("LYS", "MLZ"),
    "dimethyl-Lys": Replace("LYS", "MLY"),
    "trimethyl-Lys": Replace("LYS", "M3L"),
    "carboxy-Lys": Replace("LYS", "KCX"),
    "hydroxy-Lys": Replace("LYS", "LYZ", "5-hydroxylysine"),
    # --- arginine
    "citrulline": Replace("ARG", "CIR"),
    "dimethyl-Arg-asym": Replace("ARG", "DA2", "ADMA"),
    "dimethyl-Arg-sym": Replace("ARG", "2MR", "SDMA (CCD 2MR)"),
    # --- others
    "hydroxy-Pro": Replace("PRO", "HYP", "4R-hydroxyproline"),
    "sulfo-Tyr": Replace("TYR", "TYS"),
    "nitro-Tyr": Replace("TYR", "NIY"),
    "sulfenic-Cys": Replace("CYS", "CSO"),
    "S-nitroso-Cys": Replace("CYS", "SNC"),
    "S-palmitoyl-Cys": Replace("CYS", "P1L"),
    "pyroglutamate": Replace("GLN", "PCA", "N-terminal Gln only"),
    # --- glycosylation (ligand + covalent bond)
    "N-GlcNAc": Adduct(("ASN",), AdductSpec("NAG", "ND2", "C1", ("O1",)), "N-linked core GlcNAc (beta)"),
    "O-GlcNAc-Ser": Adduct(("SER",), AdductSpec("NAG", "OG", "C1", ("O1",)), "O-GlcNAc (beta)"),
    "O-GlcNAc-Thr": Adduct(("THR",), AdductSpec("NAG", "OG1", "C1", ("O1",)), "O-GlcNAc (beta)"),
    "O-GalNAc-Ser": Adduct(("SER",), AdductSpec("A2G", "OG", "C1", ("O1",)), "mucin-type Tn (alpha)"),
    "O-GalNAc-Thr": Adduct(("THR",), AdductSpec("A2G", "OG1", "C1", ("O1",)), "mucin-type Tn (alpha)"),
    "O-Fuc-Ser": Adduct(("SER",), AdductSpec("FUC", "OG", "C1", ("O1",)), "O-fucose (alpha-L)"),
    "O-Fuc-Thr": Adduct(("THR",), AdductSpec("FUC", "OG1", "C1", ("O1",)), "O-fucose (alpha-L)"),
    # --- lipids / other covalent groups
    "N-myristoyl-Gly": Adduct(("GLY",), AdductSpec("MYR", "N", "C1", ("O2",)), "N-terminal Gly"),
    "S-farnesyl-Cys": Adduct(("CYS",), AdductSpec("FAR", "SG", "C1", ()), "prenylation"),
    "biotinyl-Lys": Adduct(("LYS",), AdductSpec("BTN", "NZ", "C11", ("O12",))),
    "lipoyl-Lys": Adduct(("LYS",), AdductSpec("LPA", "NZ", "C1", ("O1",))),
}


def _norm(s: str) -> str:
    return s.lower().replace("_", "-").replace(" ", "-")


_BY_NORM = {_norm(k): k for k in REGISTRY}


def lookup(name: str) -> tuple[str, Replace | Adduct]:
    k = _BY_NORM.get(_norm(name))
    if k is None:
        raise KeyError(f"unknown PTM {name!r}; known: {', '.join(REGISTRY)}")
    return k, REGISTRY[k]


def parse_spec(text: str):
    """'A45=phospho-Ser' -> (ResKey-string, ('replace', code) | ('adduct', AdductSpec))."""
    res, _, what = text.partition("=")
    if not what:
        raise ValueError(f"PTM spec needs RES=WHAT, got {text!r}")
    what = what.strip()
    if what.lower().startswith("smiles:"):                                  # smiles:<SMILES with * = attachment point>|<anchor atom>[|<3-char code>]
        parts = what.split(":", 1)[1].split("|")
        if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
            raise ValueError(f"bad smiles PTM {text!r}; expected 'RES=smiles:<SMILES>|<anchor atom>[|<code>]', e.g. A9=smiles:*C(C)=O|NZ|ZAC")
        code, lig_atom = ccd.register_smiles(parts[0], parts[2] if len(parts) == 3 else None)
        return res.strip(), ("adduct", AdductSpec(code, parts[1].strip(), lig_atom, ()), None)
    if what.lower().startswith("adduct:"):
        body = what.split(":", 1)[1]
        lig_anchor, _, rest = body.partition(":")
        lig, _, anchor = lig_anchor.partition("@")
        lig_atom, _, leave = rest.partition("-")
        if not leave:
            leaving = tuple(a["name"] for a in ccd.get_comp(lig)["atoms"] if a["leave"] and a["el"] != "H")
        else:
            leaving = tuple(x for x in leave.split(",") if x)
        return res.strip(), ("adduct", AdductSpec(lig.upper(), anchor, lig_atom, leaving), None)
    if ccd.has_comp(what.upper()) and what.upper() == what and len(what) <= 5 and _norm(what) not in _BY_NORM:
        return res.strip(), ("replace", what.upper(), None)
    name, ent = lookup(what)
    if isinstance(ent, Replace):
        return res.strip(), ("replace", ent.code, ent.parent)
    return res.strip(), ("adduct", ent.spec, ent.parent)
