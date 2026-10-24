"""Seed confirmation of single-seed hits (library behind `ptm-rfd3 seeds` / `ptm-rfd3 confirm`).

A binder that passes the selectivity filters in ONE AF3 seed is usually a lucky draw (winner's curse: in our EGFR campaign the
single-seed hits regressed from a mean delta of 0.24 to 0.19, and half did not hold up). `seeds` writes N-seed AF3 inputs for the
hits (both arms, one seed per prediction so the normal `score` step works unchanged); `confirm` summarises the per-seed deltas and
flags the binders that are selective in EVERY seed.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import selectivity as SEL

SEED_TAG = "@sd"                 # spec_key suffix: pairs the two arms of one seed, and groups the seeds of one binder


def read_input_jsons(tars: list[str]) -> dict[str, dict]:
    """{af3 name: input json} from AF3 input tars (as written by `ptm-rfd3 stage`)."""
    out = {}
    for t in tars:
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".json") and not os.path.basename(m.name).startswith("index"):
                    out[os.path.basename(m.name)[:-5]] = json.load(tf.extractfile(m))
    return out


def make_seed_inputs(paired: pd.DataFrame, index_rows: list[dict], input_tars: list[str], out_prefix: str,
                     n_seeds: int = 5, seed_base: int = 100, crit: dict | None = None) -> dict:
    """AF3 inputs with `n_seeds` different seeds for every binder of `paired` that passes the single-seed selection.

    Writes `<out_prefix>_inputs.tar` (JSONs `<name>_sd<k>.json`, modelSeeds=[seed_base+k]) and `<out_prefix>_index.json`
    (rows of `ptm-rfd3 score`'s index format; `spec_key` gets the suffix `@sd<k>`)."""
    sel = SEL.select(paired, crit)
    keys = set(zip(sel.design, sel.spec_key))
    rows = [r for r in index_rows if (r["design"], r["spec_key"]) in keys]
    js = read_input_jsons(input_tars)
    have = {(r["design"], r["spec_key"]) for r in rows if r["af3_name"] in js}
    complete = {k for k in have if {r["arm"] for r in rows if (r["design"], r["spec_key"]) == k and r["af3_name"] in js} >= {"ptm", "ctl"}}
    new_rows, files = [], []
    for r in rows:
        if (r["design"], r["spec_key"]) not in complete:
            continue
        for k in range(1, n_seeds + 1):
            name = f"{r['af3_name']}_sd{k}"
            files.append((f"{name}.json", json.dumps(dict(js[r["af3_name"]], name=name, modelSeeds=[seed_base + k])).encode()))
            new_rows.append({**r, "af3_name": name, "spec_key": f"{r['spec_key']}{SEED_TAG}{k}"})
    if not new_rows:
        raise ValueError("no binder passes the single-seed selection with both arms present in the AF3 inputs")
    tar_path = f"{out_prefix}_inputs.tar"
    with tarfile.open(tar_path, "w") as tf:
        for fn, data in files:
            ti = tarfile.TarInfo(fn)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    Path(f"{out_prefix}_index.json").write_text(json.dumps(new_rows))
    return dict(selected=len(sel), with_inputs=len(complete), missing_inputs=len(keys) - len(complete), af3_inputs=len(files),
                af3_tar=tar_path, index=f"{out_prefix}_index.json")


def summarize(paired: pd.DataFrame, n_seeds: int = 5, min_worst: float = 0.05, require_engagement: bool = False) -> pd.DataFrame:
    """One row per binder over its seeds: mean/sd/worst token-matched delta, ipTM and engagement.

    A seed counts only with a finite delta and a seed tag (`@sd<k>` in the spec key; `k` is read as a number). Rows without a usable seed or
    delta (`n_invalid`) and duplicate rows of one seed with different values (`inconsistent`) block confirmation instead of being dropped silently.
    `selective_all` = `n_seeds` distinct valid seeds and the worst delta > `min_worst` (the definition of the 27 confirmed EGFR binders);
    `engaged_all` = in every valid seed the engagement filters (ipTM, >= 80 % of the PTM atoms within 4 A, closest approach, pLDDT) hold
    (only 4 of those 27 binders meet it). `confirmed` = `selective_all`, or `selective_all & engaged_all` with `require_engagement`."""
    df = paired.copy()
    df["base_spec"] = df.spec_key.str.split(SEED_TAG).str[0]
    df["seed_id"] = pd.to_numeric(df.spec_key.str.extract(r"@sd(\d+)$")[0], errors="coerce")
    for c in ("d_tm", "iptm", "ptm_frac_contacted", "ptm_site_frac_max", "ptm_min_dist", "plddt_binder"):
        if c not in df:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda x: np.isfinite(x))
    df["engaged"] = ((df.iptm >= SEL.CRITERIA["iptm"]) & (SEL.contact_frac(df) >= SEL.CRITERIA["contact"])
                     & (df.ptm_min_dist >= SEL.CRITERIA["min_dist"]) & (df.plddt_binder >= SEL.CRITERIA["plddt"]))
    df["contact"] = SEL.contact_frac(df)
    df["usable"] = df.d_tm.notna() & df.seed_id.notna()
    key = ["design", "base_spec"]
    inconsistent = (df[df.usable].groupby(key + ["seed_id"]).agg(n=("d_tm", "size"), spread=("d_tm", lambda x: x.max() - x.min()))
                    .query("n > 1 and spread > 1e-9").reset_index()[key].drop_duplicates().assign(inconsistent=True))
    valid = df[df.usable].drop_duplicates(key + ["seed_id"])
    out = valid.groupby(key).agg(n_seeds=("seed_id", "nunique"), d_tm_mean=("d_tm", "mean"), d_tm_sd=("d_tm", "std"), d_tm_worst=("d_tm", "min"),
                                 iptm_mean=("iptm", "mean"), iptm_ctl_mean=("iptm_ctl", "mean"), contact_mean=("contact", "mean"),
                                 min_dist_worst=("ptm_min_dist", "min"), plddt_mean=("plddt_binder", "mean"), engaged_all=("engaged", "all")).reset_index()
    counts = df.groupby(key).agg(n_invalid=("usable", lambda x: int((~x).sum()))).reset_index()
    out = out.merge(counts, on=key, how="right").merge(inconsistent, on=key, how="left").reset_index(drop=True)
    out["n_seeds"] = out.n_seeds.fillna(0).astype(int)
    out["engaged_all"] = out.engaged_all.fillna(False).astype(bool)
    out["inconsistent"] = out.inconsistent.fillna(False).astype(bool)
    out["d_tm_sd"] = out.d_tm_sd.fillna(0.0)
    clean = (out.n_invalid == 0) & ~out.inconsistent
    out["selective_all"] = (out.n_seeds >= n_seeds) & (out.d_tm_worst > min_worst) & clean
    out["confirmed"] = out.selective_all & (out.engaged_all if require_engagement else True)
    return out.sort_values(["confirmed", "d_tm_mean"], ascending=False).reset_index(drop=True)
