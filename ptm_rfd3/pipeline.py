"""Glue between RFD3 outputs, PTM-aware MPNN, and AF3 co-folding / scoring.

Design naming: RFD3 writes `<prefix><spec>_<i>_model_<j>.cif.gz/.json`; `spec` keys come from
`ptm-rfd3 inputs` (`<name>_ptm_k<K>_<mode>` or `<name>_unmod_ctrl`) and `manifest.json` of that run holds the
prepared target per variant (chain letters, sites, ligand chains). Output chain convention (RFD3):
binder = A, target segments = B, C, ... in contig order, covalent ligands keep their source letter.
"""
from __future__ import annotations

import glob
import gzip
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
from biotite import structure as struc
from biotite.structure.io import pdb, pdbx

from . import af3_inputs, ccd, rfd3_inputs
from . import structio as sio
from .af3_score import PTMAtomRef

BACKBONE = {"N", "CA", "C", "O"}


# ----------------------------------------------------------------------------- RFD3 outputs
def iter_designs(src: str):
    """Yield (design_name, json_meta, cif_text) from a directory of outputs or a tar written by the sbatch."""
    p = Path(src)
    if p.is_dir():
        for jp in sorted(p.glob("*_model_*.json")):
            cp = jp.with_suffix(".cif.gz")
            if cp.exists():
                yield jp.name[:-5], json.loads(jp.read_text()), gzip.open(cp, "rt").read()
    else:
        with tarfile.open(p) as tf:
            names = {m.name.lstrip("./"): m for m in tf.getmembers() if m.isfile()}
            for n, m in sorted(names.items()):
                if n.endswith(".json") and "_model_" in n:
                    c = n[:-5] + ".cif.gz"
                    if c in names:
                        yield (n[:-5], json.loads(tf.extractfile(m).read()),
                               gzip.open(io.BytesIO(tf.extractfile(names[c]).read()), "rt").read())


def read_cif_text(text: str) -> struc.AtomArray:
    f = pdbx.CIFFile.read(io.StringIO(text))
    a = pdbx.get_structure(f, model=1, use_author_fields=False)
    return a[a.element != "H"]


def spec_key_from_meta(meta: dict) -> str:
    return meta["specification"]["extra"]["example"]


def to_mpnn_pdb(design: struc.AtomArray, path: str | Path, keep_residues: set[int] | None = None) -> None:
    """Binder (A) reduced to backbone N,CA,C,O (all residues named GLY); target kept verbatim (PTM residues as
    HETATM, ligands too). `keep_residues` = binder residue ids that keep their name and ALL atoms (fixed motif)."""
    keep_res = set(keep_residues or ())
    is_keep = (design.chain_id == "A") & np.isin(design.res_id, list(keep_res))
    keep = ~((design.chain_id == "A") & ~np.isin(design.atom_name, list(BACKBONE)) & ~is_keep)
    a = design[keep].copy()
    mask_gly = (a.chain_id == "A") & ~np.isin(a.res_id, list(keep_res))
    a.res_name = np.where(mask_gly, "GLY", a.res_name)
    cats = a.get_annotation_categories()
    if "b_factor" not in cats:
        a.add_annotation("b_factor", float)
    if "occupancy" not in cats:
        a.add_annotation("occupancy", float)
        a.occupancy[:] = 1.0
    a.hetero = np.array([(r not in af3_inputs.AA3) for r in a.res_name])
    f = pdb.PDBFile()
    pdb.set_structure(f, a)
    f.write(str(path))


# ----------------------------------------------------------------------------- AF3
def site_atom_refs(enc: af3_inputs.TargetEncoding, sites: list[dict], control: bool = False) -> list[PTMAtomRef]:
    """PTM sites of a prepared target (prepared chain/res_id/atoms) -> AF3 chain id / position / atom."""
    refs: list[PTMAtomRef] = []
    pos_of = {}
    for cid, d in enc.chains.items():
        for p, (c, rid, _ins) in enumerate(d["src"], start=1):
            pos_of[(c, rid)] = (cid, p)
            pos_of[(c, rid, _ins)] = (cid, p)
    for s in sites:
        if s["kind"] == "replace":
            cid, p = pos_of[(s["chain"], s["res_id"], "")] if (s["chain"], s["res_id"], "") in pos_of else pos_of[(s["chain"], s["res_id"])]
            atoms = s["ptm_atoms"]
            if control:      # same epitope on the unmodified residue: anchor + side-chain atoms the parent has
                atoms = rfd3_inputs.parent_epitope_atoms(rfd3_inputs.PTMSite(**s))
            refs += [PTMAtomRef(cid, p, a) for a in atoms]
        else:
            lid = enc.lig_comp.get((s["chain"], s["res_id"], ""))
            if lid:
                refs += [PTMAtomRef(lid[0], lid[1], a) for a in s["ptm_atoms"]]
    return refs


def write_af3_inputs(manifest: dict, designs: list[dict], out_dir: str, seeds=(1,), target_msa=None,
                     arms: tuple[str, ...] = ("ptm", "ctl")) -> list[dict]:
    """designs: [{name, spec_key, binder_seq}] -> PTM + control AF3 JSONs (`arms` limits which). Returns index rows."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ccd.load_custom(manifest.get("custom_components"))           # SMILES-defined groups of this target
    variants = {f"{manifest['name']}_ptm_k{v['k']}": v for v in manifest["variants"]}
    rows = []
    enc_cache = {}
    side = out.with_name(out.name + "_names.json")           # name -> design of every earlier call into this folder (NOT inside it: af3_batch globs *.json)
    seen_names: dict[str, str] = json.loads(side.read_text()) if side.exists() else {}
    for d in designs:
        m = re.match(r"(.*_ptm_k\d+)_", d["spec_key"] + "_")
        vkey = m.group(1) if m else None
        is_ctrl = d["spec_key"].endswith("_unmod_ctrl")
        v = variants[vkey] if not is_ctrl else variants[f"{manifest['name']}_ptm_k0"]
        cif = manifest["control"]["cif"] if is_ctrl else v["cif"]
        sites = v["sites"]
        if cif not in enc_cache:
            enc_cache[cif] = af3_inputs.encode_target(sio.read_structure(cif))
        enc = enc_cache[cif]
        ctrl_enc = af3_inputs.encode_target(sio.read_structure(manifest["control"]["cif"]))
        for tag, e in ((("ctl", ctrl_enc),) if is_ctrl else (("ptm", enc), ("ctl", ctrl_enc))):
            if tag not in arms:
                continue
            nm = re.sub(r"[^a-z0-9_.-]", "_", f"{d['name']}_{tag}".lower())
            ident = [d["name"], d["binder_seq"]]
            if seen_names.setdefault(nm, ident) != ident:
                raise ValueError(f"AF3 input name collision: {d['name']!r} and {seen_names[nm][0]!r} (or the same name with another sequence) "
                                 f"both become {nm!r}")
            js = af3_inputs.make_af3_json(nm, d["binder_seq"], e, seeds=seeds, target_msa=target_msa)
            (out / f"{nm}.json").write_text(json.dumps(js))
            rows.append({"af3_name": nm, "design": d["name"], "label": d.get("label"), "spec_key": d["spec_key"], "arm": tag,
                         "binder_seq": d["binder_seq"], "target_chains": list(e.chains),
                         "ptm_atoms": [r.__dict__ for r in site_atom_refs(e, sites, control=(tag == "ctl"))]})
    side.write_text(json.dumps(seen_names))
    # NOT inside the input folder: af3_batch globs *.json there
    out.with_name(out.name + "_index.json").write_text(json.dumps(rows, indent=1))
    return rows


def design_name_of(af3_row: dict) -> str:
    return re.sub(r"_s\d+$", "", af3_row["design"])


def score_af3_run(af3_tar: str, index_rows: list[dict], design_src: str, target_chain_map: dict[str, str] | None = None
                  ) -> list[dict]:
    """Score every AF3 prediction of an index (PTM and control arms) against its RFD3 design."""
    from . import af3_score as S
    entries = S.read_tar(af3_tar)
    designs = {name: read_cif_text(cif) for name, _m, cif in iter_designs(design_src)}
    out = []
    for r in index_rows:
        e = entries.get(r["af3_name"])
        if not e or "summary" not in e:
            out.append({**{k: r[k] for k in ("af3_name", "design", "spec_key", "arm")}, "status": "missing"})
            continue
        refs = [PTMAtomRef(**a) for a in r["ptm_atoms"]]
        des = designs.get(design_name_of(r))
        tmap = target_chain_map or {c: c for c in r["target_chains"]}
        sc = S.score_entry(e, "A", r["target_chains"], refs, design=des, design_binder="A", design_target_map=tmap)
        out.append({**{k: r[k] for k in ("af3_name", "design", "spec_key", "arm")}, "status": "ok", **sc})
    return out


def pack_inputs(src_dir: str, tar_path: str, remove: bool = True) -> int:
    """Pack a folder of AF3 input JSONs into ONE tar (hnvme has an inode quota) and optionally delete the folder."""
    import shutil
    import tarfile
    n = 0
    with tarfile.open(tar_path, "w") as tf:
        for f in sorted(Path(src_dir).glob("*.json")):
            tf.add(f, arcname=f.name)
            n += 1
    if remove:
        shutil.rmtree(src_dir, ignore_errors=True)
    return n
