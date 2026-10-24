"""After RFD3: geometry filter -> PTM-aware LigandMPNN -> AF3 inputs (PTM arm + unmodified control). Library behind `ptm-rfd3 stage`.

Input: the output tar(s) of an RFD3 run started through `ptm_rfd3.rfd3_patch` / `ptm-rfd3 design` with PTM keys in the spec
(the tar carries `ptm_build/<spec>_<hash>/manifest.json` and the prepared targets, which are used for the AF3 inputs).
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from biotite import structure as struc

from . import ccd
from . import mpnn_stage as M
from . import pipeline as P
from . import rfd3_inputs
from .af3_inputs import AA3


def load_manifests(tar: str, dest: Path) -> dict[str, dict]:
    """Manifests of the built PTM targets of an RFD3 run (output tar -> extracted into `dest`, or an out_dir) as {spec name: manifest}."""
    mans: dict[str, dict] = {}
    if Path(tar).is_dir():                                           # a plain RFD3 out_dir: ptm_build/ sits inside it
        for mf in Path(tar).glob("ptm_build/*/manifest.json"):
            d = json.loads(mf.read_text())
            ccd.load_custom(d.get("custom_components"))
            for v in d["variants"]:
                v["cif"] = str(mf.parent / Path(v["cif"]).name)
            d["control"]["cif"] = str(mf.parent / Path(d["control"]["cif"]).name)
            mans[d["name"]] = d
        return mans
    shutil.rmtree(dest, ignore_errors=True)                          # never pick up manifests of an earlier stage run
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            n = m.name.lstrip("./")
            if m.isfile() and n.startswith("ptm_build/") and n.count("/") == 2:
                folder = n.split("/")[1]
                d = dest / folder
                d.mkdir(exist_ok=True)
                (d / Path(n).name).write_bytes(tf.extractfile(m).read())
    for mf in dest.glob("*/manifest.json"):
        d = json.loads(mf.read_text())
        ccd.load_custom(d.get("custom_components"))
        for v in d["variants"]:
            v["cif"] = str(mf.parent / Path(v["cif"]).name)
        d["control"]["cif"] = str(mf.parent / Path(d["control"]["cif"]).name)
        mans[d["name"]] = d
    return mans


def _portable(man: dict, out: Path, key: str) -> dict:
    """Copy the built target CIFs next to the stage output and return the manifest with paths RELATIVE to `out`: `control` then works from the
    stage output alone (a later session, another host), not only while the scratch directory of the RFD3 run still exists."""
    d = out / "targets" / re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    d.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(man)
    for ref in [*m["variants"], m["control"]]:
        src = Path(ref["cif"])
        dst = d / src.name                                                   # ptm_k0.cif, ptm_k1.cif, ..., unmod.cif: unique within one built target
        shutil.copy(src, dst)
        ref["cif"] = str(dst.relative_to(out))
    return m


def _absolute(man: dict, base: Path) -> dict:
    """Inverse of `_portable`: resolve the manifest's relative paths against the directory of manifests.json."""
    for ref in [*man["variants"], man["control"]]:
        if not Path(ref["cif"]).is_absolute():
            ref["cif"] = str(base / ref["cif"])
    return man


def _anionic(site) -> bool:
    """Phosphate / sulfate / sulfonate-like site: a P or S atom with >= 3 BONDED oxygens (distance < 1.9 A). Biotin (S + 3 O in other groups),
    farnesyl, S-nitroso ... are not anionic."""
    el = site.element
    for i in np.where(np.isin(el, ["P", "S"]))[0]:
        d = np.linalg.norm(site.coord - site.coord[i], axis=1)
        if int(((el == "O") & (d < 1.9)).sum()) >= 3:
            return True
    return False


def design_metrics(name: str, meta: dict, cif: str) -> dict:
    """Geometry of one RFD3 design against its modification: closest approach, polar contacts, charged partners, clashes."""
    a = P.read_cif_text(cif)
    tgt = a[a.chain_id != "A"].copy()
    ca_res = {(str(c), int(r)) for c, r, n in zip(tgt.chain_id, tgt.res_id, tgt.atom_name) if n == "CA"}
    tgt.hetero = np.array([(str(c), int(r)) not in ca_res for c, r in zip(tgt.chain_id, tgt.res_id)])   # residues with a backbone are polymer, even if HETATM
    sites = rfd3_inputs.detect_sites(tgt)
    binder = a[a.chain_id == "A"]
    atoms = []
    for s in sites:
        m = (a.chain_id == s.chain) & (a.res_id == s.res_id) & np.isin(a.atom_name, s.ptm_atoms)
        atoms.append(a[m])
    out = dict(name=name, spec=None, k=None, dmin_ptm=np.nan, hb_ptm=0, n_arg_lys_ptm=0, contact_res=0, anionic=False, polar_site=False,
               clash=meta["metrics"].get("n_clashing.interresidue_clashes_w_sidechain", np.nan))
    if not atoms or sum(len(x) for x in atoms) == 0:
        return out
    site = struc.concatenate(atoms)
    d = np.linalg.norm(binder.coord[:, None] - site.coord[None], axis=-1)
    polar_site = np.isin(site.element, ["N", "O"])
    pol = np.isin(binder.element, ["N", "O"])
    out["polar_site"] = bool(polar_site.any())
    out["dmin_ptm"] = float(d.min())
    if polar_site.any():                                                  # H-bond / salt-bridge partners need polar site atoms
        dp = d[:, polar_site]
        out["hb_ptm"] = int(((dp < 3.4) & pol[:, None]).any(1).sum())
        out["n_arg_lys_ptm"] = int(sum(1 for n in binder.res_name[(dp < 4.5).any(1)] if n in ("ARG", "LYS")))
    out["anionic"] = _anionic(site)
    poly_chains = {c for c, _r in ca_res}                                 # contacts over EVERY target polymer chain (representation-independent)
    tp = a[np.isin(a.chain_id, list(poly_chains))]
    dall = np.linalg.norm(binder.coord[:, None] - tp.coord[None], axis=-1)
    out["contact_res"] = int(len({(c, r) for c, r in zip(tp.chain_id[(dall < 4.5).any(0)], tp.res_id[(dall < 4.5).any(0)])}))
    return out


def _parse_labelled(items: list[str]) -> dict[str, str]:
    """`LABEL=path` (LABEL an identifier) or a bare path; paths containing '=' are not mistaken for labels; labels must be unique."""
    out: dict[str, str] = {}
    for i, t in enumerate(items):
        m = re.match(r"^([A-Za-z0-9_.-]+)=(.+)$", t)
        label, path = (m.group(1), m.group(2)) if m and not Path(t).exists() else (f"r{i}", t)
        if "__" in label:
            raise ValueError(f"run label {label!r} must not contain '__' (it separates label and design name)")
        if label in out:
            raise ValueError(f"duplicate run label {label!r}; give every RFD3 output its own LABEL=path")
        out[label] = path
    return out


def stage(rfd3_tars: dict[str, str], out_dir: str, top: int = 6, n_seq: int = 2, workers: int = 8, seed: int = 0,
          min_dist: float = 2.4, max_dist: float = 6.0, min_hb: int = 2, omit_aa: str = "C", ala_bias: float = 1.0, phos_bias: bool = False,
          temperature: float = 0.1, af3_spec: str | None = None, two_stage: bool = False, geometry_only: bool = False,
          mpnn_model: str = "released", keep_sequence: bool = False, geometry_gate: bool = True, exclude_uids: set | None = None, include_uids: set | None = None) -> dict:
    """Geometry-filter the RFD3 designs, design sequences, write the AF3 input tar. Returns paths and counts.
    keep_sequence: use the sequence that is IN each design (chain A) and skip LigandMPNN - for generators that co-design sequence and structure (RFD4).
    exclude_uids: designs (label__name) that are NOT staged.  include_uids: ONLY these designs are staged (applied after the gate, if any).
    geometry_gate: False sends EVERY design to AF3 (the geometry columns are still computed and written) - the 'look at all designs' control of a generator."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("mpnn_in", "mpnn_p", "mpnn_b", "af3_in", "af3_in_ctl"):          # a re-run must not mix with the previous one
        shutil.rmtree(out / stale, ignore_errors=True)
    for stale in ("af3_inputs.tar", "af3_inputs_ctl.tar", "af3_index.json", "af3_index_ctl.json", "manifests.json", "geometry_metrics.csv",
                  "af3_in_index.json", "af3_in_names.json"):
        (out / stale).unlink(missing_ok=True)
    rows, mans = [], {}
    for label, tar in rfd3_tars.items():
        mans[label] = load_manifests(tar, out / f"build_{label}")
        for name, meta, cif in P.iter_designs(tar):
            key = P.spec_key_from_meta(meta)
            m = re.match(r"^(.*?)_ptm_k(\d+)", key)
            r = design_metrics(name, meta, cif)
            r.update(label=label, spec=m.group(1) if m else key, k=int(m.group(2)) if m else 0, spec_key=key)
            rows.append(r)
    if not rows:
        raise ValueError("no RFD3 designs found in " + ", ".join(rfd3_tars.values()))
    df = pd.DataFrame(rows)
    df["uid"] = df.label + "__" + df.name                                             # run identity travels with the design
    if df.uid.duplicated().any():
        raise ValueError("duplicate design names inside one run: " + ", ".join(df.uid[df.uid.duplicated()].head(3)))
    needs_hb = df.polar_site.astype(bool)
    engaged = (df.dmin_ptm <= max_dist) & (df.contact_res >= 1)                       # the binder must actually sit at the modification
    df["geo_pass"] = ((df.clash == 0) & (df.dmin_ptm > min_dist) & engaged & (~needs_hb | (df.hb_ptm >= min_hb))
                      & (~df.anionic.astype(bool) | (df.n_arg_lys_ptm >= 1)))
    df["score"] = df.hb_ptm + 2 * df.n_arg_lys_ptm + 0.5 * df.contact_res
    df.to_csv(out / "geometry_metrics.csv", index=False)
    good = df[df.geo_pass] if geometry_gate else df
    if exclude_uids:                                                                    # e.g. designs that an earlier run already scored (a fresh, independent sample)
        good = good[~good.uid.isin(exclude_uids)]
    if include_uids is not None:                                                        # an explicit design set (same backbones for every model that is compared)
        good = good[good.uid.isin(include_uids)]
    sel = good.sort_values("score", ascending=False).groupby(["label", "spec", "k"]).head(top if geometry_gate else len(df))
    if geometry_only:                                                                   # engagement statistics only (no MPNN / AF3 inputs)
        summ = df.groupby(["label", "spec"]).agg(n=("name", "size"), geo_pass=("geo_pass", "mean"), n_pass=("geo_pass", "sum"),
                                                 dmin=("dmin_ptm", "median"), hb=("hb_ptm", "median")).round(3)
        summ.to_csv(out / "geometry_summary.csv")
        print(summ.to_string())
        return dict(n_designs=len(df), geo_pass=int(df.geo_pass.sum()), metrics=str(out / "geometry_metrics.csv"),
                    summary=str(out / "geometry_summary.csv"))
    if sel.empty:
        raise ValueError(f"no design passes the geometry filter ({len(df)} designs; see {out / 'geometry_metrics.csv'})")
    chosen = set(sel.uid)
    pdbs = {}
    for label, tar in rfd3_tars.items():
        for name, meta, cif in P.iter_designs(tar):
            uid = f"{label}__{name}"
            if uid in chosen and not keep_sequence:
                p = out / "mpnn_in" / f"{uid}.pdb"
                p.parent.mkdir(exist_ok=True)
                P.to_mpnn_pdb(P.read_cif_text(cif), p)
                pdbs[uid] = str(p)
    anionic = {r.uid: bool(r.anionic) for r in sel.itertuples()}
    if keep_sequence:
        seqs = []
        for label, tar in rfd3_tars.items():
            for name, meta, cif in P.iter_designs(tar):
                uid = f"{label}__{name}"
                if uid in chosen:
                    b = P.read_cif_text(cif)
                    b = b[b.chain_id == "A"]
                    seq = "".join(AA3.get(str(n), "X") for n in b.res_name[struc.get_residue_starts(b)])
                    if "X" in seq:
                        raise ValueError(f"{uid}: binder sequence contains non-standard residues ({seq}); cannot be staged for AF3")
                    seqs.append(("d", {"design": uid, "seq_idx": 0, "binder_seq": seq}))
    else:
        seqs = _design_sequences(pdbs, out, anionic, ala_bias, phos_bias, n_seq, workers, seed, omit_aa, temperature, mpnn_model)
    info = {r.uid: (r.label, r.spec, r.k) for r in sel.itertuples()}
    return _write_af3(out, mans, seqs, info, df, sel, two_stage, af3_spec)


def _design_sequences(pdbs, out, anionic, ala_bias, phos_bias, n_seq, workers, seed, omit_aa, temperature, mpnn_model):
    extra_ala = ["--bias_AA", f"A:{-abs(ala_bias)}"] if ala_bias else []
    seqs = []
    # plain PTM-aware MPNN by default: the phosphate Arg/Lys bias LOWERED the token-matched selectivity by 0.057 on average
    # (paired by backbone, n = 188, p ~ 1e-7; N014) - it raises the control arm's confidence as much as the PTM arm's
    modes = [("p", lambda n: [])] + ([("b", lambda n: ["--phos-bias"] if anionic.get(n) else [])] if phos_bias else [])
    for tag, per_design in modes:
        extra = {n: per_design(n) + extra_ala for n in pdbs}
        rows_, errs = M.design_sequences(pdbs, str(out / f"mpnn_{tag}"), n_seq=n_seq, workers=workers, seed=seed, omit_aa=omit_aa,
                                         temperature=temperature, extra_by_name=extra, model=mpnn_model)
        got = {r["design"] for r in rows_}
        missing = [n for n in pdbs if n not in got]
        if errs or missing:
            bad = dict(errs) | {n: "no sequences produced" for n in missing if n not in errs}
            raise RuntimeError(f"LigandMPNN failed for {len(bad)} of {len(pdbs)} designs (mode {tag}); first: "
                               + "; ".join(f"{k}: {v[-160:]}" for k, v in list(bad.items())[:2]))
        seqs += [(tag, r) for r in rows_]
    if not seqs:
        raise RuntimeError("LigandMPNN produced no sequences")
    return seqs


def _write_af3(out, mans, seqs, info, df, sel, two_stage, af3_spec):
    idx_all, n_des, saved = [], 0, {}
    for label, by_spec in mans.items():
        for spec, man in by_spec.items():
            # af3_spec: score every design against the SAME AF3 target (e.g. the phospho-Tyr arm) although RFD3 saw another
            # representation of the PTM (ligand phosphate); conformer index k selects the variant
            tgt = by_spec[af3_spec] if af3_spec else man
            tname = af3_spec or spec
            designs = [{"name": f"{r['design']}_{t}{r['seq_idx']}", "spec_key": f"{tname}_ptm_k{min(info[r['design']][2], len(tgt['variants']) - 1)}",
                        "binder_seq": r["binder_seq"], "label": label} for t, r in seqs if info[r["design"]][:2] == (label, spec)]
            if designs:
                idx_all += P.write_af3_inputs(tgt, designs, str(out / "af3_in"), arms=("ptm",) if two_stage else ("ptm", "ctl"))
                saved[f"{label}/{tname}"] = _portable(tgt, out, f"{label}__{tname}")
                n_des += len(designs)
    (out / "af3_index.json").write_text(json.dumps(idx_all))
    (out / "manifests.json").write_text(json.dumps(saved, default=str))          # for `ptm-rfd3 control` (stage 2 of a two-stage screen)
    n_json = P.pack_inputs(str(out / "af3_in"), str(out / "af3_inputs.tar")) if idx_all else 0
    skipped = sorted({(info[r["design"]][0], info[r["design"]][1]) for _t, r in seqs} - {tuple(k.split("/", 1)) for k in saved})
    if skipped:
        print("WARNING: designs of specs without a built-PTM manifest were not staged (spec has no PTM keys?): "
              + ", ".join(f"{a}/{b}" for a, b in skipped))
    if not idx_all:
        raise RuntimeError("no AF3 inputs were written (no matching manifests for the selected designs)")
    return dict(skipped_specs=[f"{a}/{b}" for a, b in skipped], n_designs=len(df), geo_pass=int(df.geo_pass.sum()), selected=len(sel), sequences=n_des, af3_inputs=n_json,
                af3_tar=str(out / "af3_inputs.tar"), index=str(out / "af3_index.json"), metrics=str(out / "geometry_metrics.csv"))


def control_stage(scores_csv: str, index: str, manifests: str, out_dir: str, crit: dict | None = None) -> dict:
    """Stage 2 of the two-stage screen: write the unmodified-control AF3 inputs ONLY for binders whose PTM-arm prediction passes the
    engagement filters (ipTM, PTM-atom contact, closest approach, pLDDT). Selection requires those filters anyway, so nothing is lost;
    the control arm of the other ~70% of binders (about a third of the AF3 time) is never computed."""
    from . import selectivity as SEL
    sc = pd.read_csv(scores_csv)
    c = {**SEL.CRITERIA, **(crit or {})}
    ok = sc[(sc.arm == "ptm") & (sc.iptm >= c["iptm"]) & (SEL.contact_frac(sc) >= c["contact"]) & (sc.ptm_min_dist >= c["min_dist"])
            & (sc.plddt_binder >= c["plddt"])]
    rows = {r["af3_name"]: r for r in json.loads(Path(index).read_text())}
    mans = {k: _absolute(v, Path(manifests).resolve().parent) for k, v in json.loads(Path(manifests).read_text()).items()}
    by_group: dict[str, list[dict]] = {}
    for n in ok.af3_name:
        r = rows[n]
        label = r.get("label") or (r["design"].split("__", 1)[0] if "__" in r["design"] else "")
        key = f"{label}/{r['spec_key'].rsplit('_ptm_k', 1)[0]}"
        by_group.setdefault(key, []).append({"name": r["design"], "spec_key": r["spec_key"], "binder_seq": r["binder_seq"]})
    out = Path(out_dir)
    shutil.rmtree(out / "af3_in_ctl", ignore_errors=True)
    for stale in ("af3_in_ctl_names.json", "af3_in_ctl_index.json", "af3_inputs_ctl.tar", "af3_index_ctl.json"):
        (out / stale).unlink(missing_ok=True)
    if ok.empty:
        raise ValueError(f"no PTM-arm prediction passes the engagement filters ({len(sc)} scored); nothing to confirm with a control arm")
    idx = []
    for key, designs in by_group.items():
        if key not in mans:
            raise ValueError(f"no saved manifest for run/spec {key!r} (have: {sorted(mans)[:6]}); re-run `ptm-rfd3 stage`")
        idx += P.write_af3_inputs(mans[key], designs, str(out / "af3_in_ctl"), arms=("ctl",))
    (out / "af3_index_ctl.json").write_text(json.dumps(idx))
    n_json = P.pack_inputs(str(out / "af3_in_ctl"), str(out / "af3_inputs_ctl.tar")) if idx else 0
    return dict(ptm_scored=int((sc.arm == "ptm").sum()), pass_ptm_filters=len(ok), control_inputs=n_json, tar=str(out / "af3_inputs_ctl.tar"),
                index=str(out / "af3_index_ctl.json"))
