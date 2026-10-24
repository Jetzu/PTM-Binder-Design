"""PTM selectivity scoring and selection on AF3 outputs (library behind `ptm-rfd3 score` / `ptm-rfd3 select`).

For every binder sequence the AF3 co-fold of the PTM-bearing target (arm `ptm`) is compared with the same binder against the
unmodified target (arm `ctl`). AF3 tokenises a modified residue per atom, so ipTM(ptm) - ipTM(ctl) over-weights the PTM;
the primary selectivity score is the TOKEN-MATCHED confidence (`af3_score.tm_residue_weighted`: one weight per residue).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import af3_score as S

CRITERIA = dict(iptm=0.5, contact=0.8, min_dist=2.2, plddt=85.0, d_tm=0.15)


def score_tar(tar: str | list[str], index: str | list[str], binder: str = "A", target: str | None = None) -> pd.DataFrame:
    """Score every AF3 prediction of an index (rows written by pipeline.write_af3_inputs): ipTM, token-matched confidence,
    binder pLDDT and PTM-atom engagement (fraction of the PTM atoms within 4 A of the binder, closest approach)."""
    entries: dict = {}
    for t in ([tar] if isinstance(tar, str) else tar):          # an AF3 run may be split over several tars
        for k, e in S.read_tar(t).items():
            old = entries.get(k)
            if old is None or not all(x in old for x in ("summary", "cif", "conf")):      # never let a partial duplicate replace a complete entry
                entries[k] = {**(old or {}), **e}
    rows = []
    index_rows = [r for ix in ([index] if isinstance(index, str) else index) for r in json.load(open(ix))]
    for r in index_rows:
        e = entries.get(r["af3_name"])
        if not e or not all(x in e for x in ("summary", "cif", "conf")):
            continue                                                                       # counted in n_missing (incomplete AF3 output)
        arr = S.read_model(e["cif"])
        order = S._chain_order(arr)
        tok = list(dict.fromkeys(map(str, e["conf"].get("token_chain_ids", []))))          # axis order of the confidence matrices
        if tok and set(tok) != set(order):
            raise ValueError(f"{r['af3_name']}: chains of the model {order} and of the confidences {tok} differ")
        order = tok or order
        targets = [target] if target else [c for c in order if c != binder]           # every non-binder chain (peptide, glycan ligands, ...)
        res = dict(af3_name=r["af3_name"], design=r["design"], spec_key=r["spec_key"], arm=r["arm"],
                   iptm=S.iptm_binder_target(e["summary"], order, binder, targets),
                   tm_res=S.tm_residue_weighted(e["conf"], binder, [c for c in targets if c in set(map(str, e["conf"].get("token_chain_ids", [])))]),
                   plddt_binder=S.plddt_mean(e["conf"], binder))
        res.update(S.ptm_engagement(arr, binder, [S.PTMAtomRef(**a) for a in r["ptm_atoms"]]))
        rows.append(res)
    df = pd.DataFrame(rows)
    df.attrs["n_missing"] = len(index_rows) - len(rows)                                  # predictions in the index without a usable AF3 output
    return df


def contact_frac(df: pd.DataFrame) -> pd.Series:
    """Engagement of the modification: the fraction of the atoms of the BEST-contacted SITE within 4 A of the binder. For a single-site target this is
    the share of all modification atoms (the original criterion); with several sites (or several attached groups) the binder only has to engage one of
    them completely, not all at once. Score tables written before the per-site column existed fall back to the all-atom fraction. Non-finite values
    (inf, NaN, garbage) in either source are 'unknown' and rejected downstream (NaN fails every >= test)."""
    def clean(col):
        return pd.to_numeric(df[col], errors="coerce").where(lambda x: np.isfinite(x))
    allatom = clean("ptm_frac_contacted")
    return clean("ptm_site_frac_max").fillna(allatom) if "ptm_site_frac_max" in df else allatom


def pair_arms(df: pd.DataFrame) -> pd.DataFrame:
    """One row per binder: PTM-arm scores plus the paired differences d_tm (token-matched) and d_iptm (AF3 native)."""
    if df.empty:
        raise ValueError("no AF3 predictions were scored (empty tar or index mismatch)")
    key = ["design", "spec_key"]
    dup = df.duplicated(key + ["arm"], keep=False)
    if dup.any():
        raise ValueError("duplicate (design, spec, arm) rows: " + ", ".join(df.loc[dup, "af3_name"].head(3)))
    a = df[df.arm == "ptm"].set_index(key)
    b = df[df.arm == "ctl"].set_index(key)
    n_unpaired = len(a.index.difference(b.index)) + len(b.index.difference(a.index))
    a = a.loc[a.index.intersection(b.index)].copy()
    a.attrs["n_unpaired"] = n_unpaired
    a["tm_res_ctl"] = b.loc[a.index, "tm_res"]
    a["iptm_ctl"] = b.loc[a.index, "iptm"]
    a["d_tm"] = a.tm_res - a.tm_res_ctl
    a["d_iptm"] = a.iptm - a.iptm_ctl
    return a.reset_index()


def select(paired: pd.DataFrame, crit: dict | None = None) -> pd.DataFrame:
    """Binders passing the PTM-engagement filters and the token-matched selectivity threshold, best first."""
    c = {**CRITERIA, **(crit or {})}
    ok = paired[(paired.iptm >= c["iptm"]) & (contact_frac(paired) >= c["contact"]) & (paired.ptm_min_dist >= c["min_dist"])
                & (paired.plddt_binder >= c["plddt"]) & (paired.d_tm >= c["d_tm"])]
    return ok.sort_values("d_tm", ascending=False)


def bootstrap_ci(values, stat=np.median, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of a statistic (e.g. median matched delta, fraction >= 0.10)."""
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    lo, hi = np.percentile([stat(rng.choice(x, len(x))) for _ in range(n)], [2.5, 97.5])
    return float(lo), float(hi)


def export_models(selected: pd.DataFrame, tar: str, out_dir: str | Path) -> int:
    """Write the AF3 model CIF of every selected PTM-arm prediction."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = S.read_tar(tar)
    n = 0
    for name in selected.af3_name:
        if name in entries:
            (out / f"{name}.cif").write_text(entries[name]["cif"])
            n += 1
    return n
