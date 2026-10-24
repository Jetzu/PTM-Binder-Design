"""What counts as a finished AF3 prediction - ONE definition shared by the dispatcher (af3_batch.py, runs inside the AF3 container) and the benchmark
campaign driver (benchmarks/campaign.py).  Standard library only.

A prediction of input `n` is usable when its tar holds `n/n_model.cif`, `n/n_confidences.json`, `n/n_summary_confidences.json` AND `n/n_input.json`, the
canonical JSON of the input it was computed from.  A prediction without that stamp, or whose stamp differs from the current input, is stale: it is
re-computed, never reused (a retry after the inputs were regenerated must not score old predictions as new ones)."""
from __future__ import annotations

import json
import os
import tarfile

SUFFIXES = ("_model.cif", "_confidences.json", "_summary_confidences.json", "_input.json")


def canon(d) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def top(member_name: str) -> str:
    return os.path.normpath(member_name).split("/")[0]


def inputs_from_tar(path) -> dict:
    """{name: canonical input} of every JSON dict with a `name` in a tar of inputs (the rule af3_batch.py applies to its input folder)."""
    out = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".json"):
                d = json.load(tf.extractfile(m))
                if isinstance(d, dict) and "name" in d:
                    out[d["name"]] = canon(d)
    return out


def usable_predictions(path) -> dict:
    """{name: stamped canonical input} of the complete predictions in an output tar (raises tarfile.TarError / OSError when it cannot be read)."""
    files, stamp = {}, {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            n = top(m.name)
            base = os.path.basename(m.name)
            for suf in SUFFIXES:
                if base == n + suf:
                    files.setdefault(n, set()).add(suf)
                    if suf == "_input.json":
                        stamp[n] = tf.extractfile(m).read().decode()
    return {n: stamp[n] for n, s in files.items() if s == set(SUFFIXES)}
