"""Numpy evaluator for the LightGBM conformer ranker (no lightgbm needed at run time).

Model file = JSON from `Booster.dump_model()` + {"keys": feature keys of pool_dataset.FEATURE_KEYS, "codes": one-hot
PTM codes, "no_tors": bool}. Feature layout (identical to training): [raw F | pool z-score | pool rank | code one-hot].
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

MODEL_PATH = Path(__file__).with_name("data") / "ranker.json"


def pool_features(F: np.ndarray, code: str, codes: list[str], tors_cols: list[int] | None = None) -> np.ndarray:
    """[raw | pool z-score | pool average-rank | code one-hot]; INVARIANT to the order of the candidates.

    Ties get their average rank (a constant column -> 0.5 everywhere), constant columns have z = 0. Used by BOTH the
    trainer and the builder so the layouts cannot drift apart.
    """
    F = np.asarray(F, np.float64).copy()
    if tors_cols:
        F[:, tors_cols] = 0.0
    sd = F.std(0)
    z = np.where(sd > 1e-9, (F - F.mean(0)) / np.where(sd > 1e-9, sd, 1.0), 0.0)
    n = len(F)
    rk = (rankdata(F, axis=0, method="average") - 1.0) / max(n - 1, 1) if n > 1 else np.full_like(F, 0.5)
    oh = np.zeros((n, len(codes)))
    if code in codes:
        oh[:, codes.index(code)] = 1
    return np.hstack([F, z, rk, oh])


class Ranker:
    def __init__(self, blob: dict):
        self.keys = blob["keys"]
        self.codes = blob["codes"]
        self.no_tors = blob.get("no_tors", True)
        self.tors_cols = [i for i, k in enumerate(self.keys) if k.startswith("tlib") or (k.startswith("t") and k[1:2].isdigit())]
        self.trees = [self._flatten(t["tree_structure"]) for t in blob["model"]["tree_info"]]
        self.fitted_codes = set(blob.get("fitted_codes", self.codes))

    @staticmethod
    def _flatten(node: dict):
        feat, thr, left, right, val, dflt = [], [], [], [], [], []

        def walk(n):
            i = len(feat)
            feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); val.append(0.0); dflt.append(0)
            if "leaf_value" in n:
                val[i] = float(n["leaf_value"])
            else:
                if n.get("decision_type", "<=") != "<=":
                    raise ValueError(f"unsupported LightGBM split type {n.get('decision_type')!r} (categorical)")
                feat[i] = int(n["split_feature"]); thr[i] = float(n["threshold"])
                mt = n.get("missing_type", "None")
                # LightGBM: with missing_type None a NaN is treated as 0.0 (not routed by default_left)
                dflt[i] = 2 if mt == "None" else (1 if n.get("default_left", True) else 0)
                left[i] = walk(n["left_child"]); right[i] = walk(n["right_child"])
            return i
        walk(node)
        return (np.array(feat), np.array(thr), np.array(left), np.array(right), np.array(val), np.array(dflt))

    def knows(self, code: str) -> bool:
        """True only for PTM codes that actually contributed fitting pools."""
        return code in self.fitted_codes

    def features(self, F: np.ndarray, code: str) -> np.ndarray:
        return pool_features(F, code, self.codes, self.tors_cols if self.no_tors else None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X))
        for feat, thr, left, right, val, dflt in self.trees:
            node = np.zeros(len(X), dtype=int)
            active = feat[node] >= 0
            while active.any():
                f = feat[node[active]]
                x = X[active, f]
                d = dflt[node[active]]
                th = thr[node[active]]
                x0 = np.where(np.isnan(x) & (d == 2), 0.0, x)             # missing_type None: NaN -> 0
                go_left = np.where(np.isnan(x0), d == 1, x0 <= th)
                node[active] = np.where(go_left, left[node[active]], right[node[active]])
                active = feat[node] >= 0
            out += val[node]
        return out


@lru_cache(maxsize=1)
def load(path: str = str(MODEL_PATH)) -> Ranker | None:
    p = Path(path)
    return Ranker(json.loads(p.read_text())) if p.exists() else None
