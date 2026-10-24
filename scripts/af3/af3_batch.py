#!/usr/bin/env python3
"""Run AlphaFast/AF3 on a folder of input JSONs (binder + PTM target), bucketed by REAL token count.

Run INSIDE the SIF (stdlib + the container's alphafold3 package only):
  apptainer exec --nv --bind $WEIGHTS:/data/models --bind $W:$W $SIF \
      python af3_batch.py --in_dir IN --tar OUT.tar [--samples 5 --recycles 10]

* PTM residues are 1 token per heavy atom (reader_af3_mpnn.md A), so the C2TNFA bucket formula
  undercounts; tokens are computed with the pipeline's own featuriser, then inputs are grouped into a
  small set of buckets (a compile costs ~75 s, a prediction ~21 s).
* Resumable: names already present in OUT.tar are skipped. Only the top-ranked model's files
  (model.cif, confidences.json, summary_confidences.json, ranking_scores.csv) are kept.
"""
import argparse
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import af3_tar  # noqa: E402  (what a usable prediction is; shared with the benchmark campaign driver)

BUCKETS = [128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048]


def token_count(path):
    import numpy as np
    from alphafold3.common import folding_input as fi_
    from alphafold3.constants import chemical_components as cc
    from alphafold3.data import featurisation as fz
    fi = fi_.Input.from_json(open(path).read())
    ccd = cc.Ccd(user_ccd=fi.user_ccd) if getattr(fi, "user_ccd", None) else cc.Ccd()      # SMILES-defined groups ship a `userCCD` block
    return int(np.asarray(fz.featurise_input(fi, ccd=ccd, buckets=[16])[0]["seq_mask"]).sum())


def bucket_for(n):
    for b in BUCKETS:
        if n <= b:
            return b
    return -(-n // 256) * 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--recycles", type=int, default=10)
    ap.add_argument("--budget_min", type=float, default=95.0)
    a = ap.parse_args()
    t0 = time.time()
    jobid = os.environ.get("SLURM_JOB_ID", "local")
    T = f"/tmp/ptm_af3_{jobid}"
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(f"{T}/jax")
    usable = {}
    if os.path.exists(a.tar):
        try:
            usable = af3_tar.usable_predictions(a.tar)
        except (tarfile.TarError, OSError, EOFError) as exc:                  # an unreadable partial archive must not block every retry
            os.replace(a.tar, a.tar + ".corrupt")
            print(f"unreadable partial archive moved to {a.tar}.corrupt ({exc!r}); starting from scratch", flush=True)
    done = set()                                                               # names whose usable prediction was computed from exactly the current input
    stamp = {}
    files = sorted(glob.glob(os.path.join(a.in_dir, "*.json")))
    items = []
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)       # PTM tokens flood the log with pseudo-beta warnings
    except Exception:                                            # noqa: BLE001
        pass
    n_inputs = 0
    wanted = set()                                                  # every input name: the run is complete only if all of them are in the tar
    for f in files:
        d = json.load(open(f))
        if not isinstance(d, dict) or "name" not in d:
            print(f"skip non-input json {f}", flush=True)
            continue
        n_inputs += 1
        nm = d["name"]
        wanted.add(nm)
        stamp[nm] = af3_tar.canon(d)
        if usable.get(nm) == stamp[nm]:
            done.add(nm)
            continue
        try:
            items.append((token_count(f), nm, f))
        except Exception as exc:                                  # noqa: BLE001
            print(f"FEATURISE FAIL {nm}: {exc!r}", flush=True)
    if n_inputs == 0:
        sys.exit(f"ERROR: no AF3 input JSONs (with a `name`) in {a.in_dir}")
    rerun = {nm for _n, nm, _f in items} | (wanted - done)                      # their old (stale/partial) members are dropped when the archive is rewritten
    print(f"{n_inputs} inputs, {len(done)} done, {len(items)} to run; tokens "
          f"{min((i[0] for i in items), default=0)}-{max((i[0] for i in items), default=0)}", flush=True)
    groups = {}
    for n, nm, f in sorted(items):
        groups.setdefault(bucket_for(n), []).append((n, nm, f))
    n_ok = n_fail = 0
    per_pred = None
    guess = lambda b: 3.0 if b <= 192 else 8.0 if b <= 384 else 20.0              # seconds per prediction (H100, 5 samples)
    queue = [(b, grp) for b, grp in sorted(groups.items())]
    while queue:
        b, grp = queue.pop(0)
        pp = per_pred.get(b, guess(b)) if isinstance(per_pred, dict) else guess(b)
        remaining = a.budget_min - (time.time() - t0) / 60.0 - 2.0
        fit = int(max(remaining, 0) * 60.0 / (pp * 1.3))
        if fit < 1:
            print(f"TIME CAP before bucket {b}: leaving {len(grp) + sum(len(g) for _k, g in queue)} for a resubmit", flush=True)
            break
        if fit < len(grp):                                                        # run what fits, leave the rest
            grp, rest = grp[:fit], grp[fit:]
            queue.insert(0, (b, rest))
            print(f"bucket {b}: running {len(grp)} now, {len(rest)} left for a resubmit", flush=True)
            queue = []                                                            # budget exhausted after this chunk
        cin, cout = f"{T}/in_{b}_{int(time.time())}", f"{T}/out_{b}_{int(time.time())}"
        os.makedirs(cin)
        os.makedirs(cout)
        for _, nm, f in grp:
            shutil.copy(f, f"{cin}/{nm}.json")
        t1 = time.time()
        cmd = ["python", "/app/alphafold/run_alphafold.py", f"--input_dir={cin}", "--model_dir=/data/models",
               "--norun_data_pipeline", f"--output_dir={cout}", "--force_output_dir", f"--buckets={b}",
               f"--jax_compilation_cache_dir={T}/jax", f"--num_diffusion_samples={a.samples}",
               f"--num_recycles={a.recycles}"]
        rc = subprocess.run(cmd, stdout=open(f"{T}/af3_{b}.log", "a"), stderr=subprocess.STDOUT).returncode
        dt = time.time() - t1
        print(f"bucket {b}: {len(grp)} inputs rc={rc} {dt:.0f}s", flush=True)
        if not isinstance(per_pred, dict):
            per_pred = {}
        per_pred[b] = dt / max(len(grp), 1)
        keep = []
        for _, nm, _f in grp:
            d = f"{cout}/{nm}"
            if os.path.exists(f"{d}/{nm}_summary_confidences.json"):
                keep.append((nm, d))
                done.add(nm)
                n_ok += 1
            else:
                n_fail += 1
        with tarfile.open(a.tar + ".part", "w") as tf:
            if os.path.exists(a.tar):
                with tarfile.open(a.tar) as old:
                    for m in old.getmembers():
                        if af3_tar.top(m.name) in rerun:
                            continue
                        tf.addfile(m, old.extractfile(m) if m.isfile() else None)
            for nm, d in keep:
                for suf in ("_model.cif", "_confidences.json", "_summary_confidences.json", "_ranking_scores.csv"):
                    p_ = f"{d}/{nm}{suf}"
                    if os.path.exists(p_):
                        tf.add(p_, arcname=f"{nm}/{nm}{suf}")
                data = stamp[nm].encode()                                      # the input this prediction was computed from (see af3_tar.py)
                ti = tarfile.TarInfo(f"{nm}/{nm}_input.json")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        os.replace(a.tar + ".part", a.tar)
        rerun -= {nm for nm, _d in keep}
        shutil.rmtree(cin, ignore_errors=True)
        shutil.rmtree(cout, ignore_errors=True)
    missing = sorted(wanted - done)
    print(f"done ok={n_ok} fail={n_fail} wall={(time.time()-t0)/60:.1f} min", flush=True)
    shutil.rmtree(T, ignore_errors=True)
    if missing:                                                      # featurisation failure, failed inference, or the time cap: NOT a success (the campaign resumes from the tar)
        print(f"INCOMPLETE: {len(missing)} of {len(wanted)} predictions missing, e.g. {missing[:5]}", flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
