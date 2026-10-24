"""AF3 co-fold validation for SINGLE-CHAIN domain targets (full entity sequence + MSA, PTM vs unmodified arm).

Differences to pipeline.write_af3_inputs: the AF3 target chain is the FULL entity sequence (unobserved residues
included, real MSA), so AF3 can fold the domain; designs (RFD3) only saw the observed, renumbered residues, so
design residues are mapped to entity positions through the prepared target's seg_map.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import af3_score as S
from . import pipeline as P


def design_to_entity(manifest_variant: dict, orig_chain: str) -> dict[tuple[str, int], int]:
    """(design chain letter, design res_id) -> entity position, via the prepared target's seg_map.

    RFD3 output: binder = A, target segments = B, C, ... in contig (= prepared segment) order.
    """
    segs = [s[0] for s in manifest_variant["segments"]]            # prepared letters A, B, ...
    out = {}
    for key, (new_chain, new_res) in manifest_variant["seg_map"].items():
        c, r, _ins = key.split(":")
        if c != orig_chain:
            continue
        out[(chr(ord("A") + 1 + segs.index(new_chain)), int(new_res))] = int(r)
    return out


def write_domain_af3_inputs(name: str, designs: list[dict], seq: str, msa_text: str, site: tuple[int, str],
                            out_dir: str, seeds=(1,)) -> list[dict]:
    """designs: [{name, spec_key, binder_seq}] -> one PTM-arm and one control-arm JSON per design."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    pos, code = site
    for d in designs:
        for arm in ("ptm", "ctl"):
            nm = re.sub(r"[^a-z0-9_.-]", "_", f"{d['name']}_{arm}".lower())
            tgt = {"id": ["B"], "sequence": seq, "unpairedMsa": msa_text, "pairedMsa": f">query\n{seq}\n", "templates": []}
            if arm == "ptm":
                tgt["modifications"] = [{"ptmType": code, "ptmPosition": int(pos)}]
            binder = {"id": ["A"], "sequence": d["binder_seq"], "unpairedMsa": f">query\n{d['binder_seq']}\n",
                      "pairedMsa": f">query\n{d['binder_seq']}\n", "templates": []}
            js = {"name": nm, "dialect": "alphafold3", "version": 1, "modelSeeds": list(seeds),
                  "sequences": [{"protein": binder}, {"protein": tgt}]}
            (out / f"{nm}.json").write_text(json.dumps(js))
            rows.append({"af3_name": nm, "design": d["name"], "spec_key": d["spec_key"], "arm": arm,
                         "binder_seq": d["binder_seq"], "site_pos": pos, "site_code": code})
    out.with_name(out.name + "_index.json").write_text(json.dumps(rows, indent=1))
    return rows


def _ca(a, chain):
    m = (a.chain_id == chain) & (a.atom_name == "CA")
    return {int(r): x for r, x in zip(a.res_id[m], a.coord[m])}


def score_domain(af3_tar: str, rows: list[dict], design_src: str, d2e: dict[tuple[str, int], int],
                 ptm_atoms: list[str]) -> list[dict]:
    entries = S.read_tar(af3_tar)
    designs = {n: P.read_cif_text(c) for n, _m, c in P.iter_designs(design_src)}
    out = []
    for r in rows:
        e = entries.get(r["af3_name"])
        base = {k: r[k] for k in ("af3_name", "design", "spec_key", "arm")}
        if not e or "summary" not in e:
            out.append({**base, "status": "missing"})
            continue
        arr = S.read_model(e["cif"])
        order = S._chain_order(arr)
        refs = [S.PTMAtomRef("B", int(r["site_pos"]), a) for a in ptm_atoms]
        res = {"iptm_binder_target": S.iptm_binder_target(e["summary"], order, "A", ["B"]),
               "iptm": float(e["summary"].get("iptm", np.nan)), "ranking_score": float(e["summary"].get("ranking_score", np.nan)),
               "plddt_binder": S.plddt_mean(e["conf"], "A"), "plddt_target": S.plddt_mean(e["conf"], "B"),
               "ipae_binder_target": S.interface_pae(e["conf"], "A", ["B"]),
               "has_clash": bool(e["summary"].get("has_clash", False))}
        res.update(S.ptm_engagement(arr, "A", refs))
        des = designs.get(re.sub(r"_s\d+$", "", r["design"]))
        if des is not None:
            tgt_pred, tgt_des = _ca(arr, "B"), {}
            for (c, rid), pos in d2e.items():
                m = (des.chain_id == c) & (des.res_id == rid) & (des.atom_name == "CA")
                if m.any():
                    tgt_des[pos] = des.coord[m][0]
            common = sorted(set(tgt_pred) & set(tgt_des))
            bp, bd = _ca(arr, "A"), _ca(des, "A")
            if len(common) >= 20 and set(bp) == set(bd):
                P_, Q_ = np.array([tgt_pred[p] for p in common]), np.array([tgt_des[p] for p in common])
                R, t = S._kabsch(P_, Q_)
                xb = np.array([bp[k] for k in sorted(bp)]) @ R.T + t
                yb = np.array([bd[k] for k in sorted(bd)])
                res["pose_rmsd"] = float(np.sqrt(((xb - yb) ** 2).sum(-1).mean()))
                R2, t2 = S._kabsch(np.array([bp[k] for k in sorted(bp)]), yb)
                res["fold_rmsd"] = float(np.sqrt((((np.array([bp[k] for k in sorted(bp)]) @ R2.T + t2) - yb) ** 2).sum(-1).mean()))
                res["target_rmsd"] = float(np.sqrt((((P_ @ R.T + t) - Q_) ** 2).sum(-1).mean()))
        out.append({**base, "status": "ok", **res})
    return out
