"""Environment-aware scoring of candidate PTM conformers (heavy atoms only, numpy/scipy)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90, "F": 1.47, "CL": 1.75,
       "BR": 1.85, "H": 1.10,
       # metal ions are coordinated by O/N at ~2.0-2.3 A: small effective radius
       "ZN": 1.0, "MG": 1.0, "MN": 1.0, "FE": 1.0, "CO": 1.0, "NI": 1.0, "CU": 1.0, "CA": 1.2, "NA": 1.2,
       "K": 1.5, "CD": 1.1, "HG": 1.1}
POLAR = {"N", "O"}
CLASH_SCALE = 0.80      # clash if d < 0.80*(ri+rj)
HB_POLAR_MIN = 2.50     # below this a polar pair is a clash, 2.5-3.3 is an H-bond candidate
HB_POLAR_MAX = 3.30
SALT_MAX = 4.00         # anionic group O .. Arg/Lys N


def vdw(el: str) -> float:
    return VDW.get(el.upper(), 1.8)


@dataclass
class Environment:
    """Heavy atoms the PTM must coexist with (everything except the modified residue itself)."""
    coord: np.ndarray       # (M, 3)
    element: np.ndarray     # (M,) upper-case symbols
    atom_name: np.ndarray   # (M,)
    res_name: np.ndarray    # (M,)
    chain_id: np.ndarray | None = None
    res_id: np.ndarray | None = None

    def __post_init__(self):
        self.tree = cKDTree(self.coord) if len(self.coord) else None
        self.radius = np.array([vdw(e) for e in self.element]) if len(self.coord) else np.zeros(0)
        # cationic / H-bond donor-capable sidechain atoms of Arg/Lys/His for salt-bridge reward
        cat = (((self.res_name == "ARG") & np.isin(self.atom_name, ["NE", "NH1", "NH2"]))
               | ((self.res_name == "LYS") & (self.atom_name == "NZ"))
               | ((self.res_name == "HIS") & np.isin(self.atom_name, ["ND1", "NE2"])))
        self.cationic = cat
        self.polar = np.isin(self.element, list(POLAR))
        self.charge = _env_charges(self.res_name, self.atom_name)
        self.ring_centroids = _ring_centroids(self)


W = {"clash": 8.7, "strain": 0.08, "hbond": 0.2, "salt": 0.0, "contact": 0.002, "coul": 0.17, "cationpi": 0.0, "tors": 0.4, "tlib": 0.0}  # tuned on N002 pools (2-fold CV gain 0.07-0.15 A top-1 median)


def score_conformer(
    new_xyz: np.ndarray,
    new_el: np.ndarray,
    new_is_anion: np.ndarray | None = None,
    env: Environment | None = None,
    new_q: np.ndarray | None = None,
) -> dict:
    """Clash / H-bond / salt-bridge / electrostatic terms for the NEW heavy atoms of one conformer."""
    out = {"clash": 0.0, "n_clash": 0, "n_hbond": 0, "n_salt": 0, "n_contact": 0, "coul": 0.0, "n_cp": 0}
    if env is None or env.tree is None or len(new_xyz) == 0:
        return out
    r_new = np.array([vdw(e) for e in new_el])
    pairs = env.tree.query_ball_point(new_xyz, r=6.0)
    for i, js in enumerate(pairs):
        if not js:
            continue
        js = np.asarray(js)
        d = np.linalg.norm(env.coord[js] - new_xyz[i], axis=1)
        pol = (new_el[i] in POLAR) & env.polar[js]
        thr = np.where(pol, HB_POLAR_MIN, CLASH_SCALE * (r_new[i] + env.radius[js]))
        viol = np.clip(thr - d, 0, None)
        out["clash"] += float((viol ** 2).sum())
        out["n_clash"] += int((viol > 0.3).sum())
        out["n_contact"] += int((d < 4.5).sum())
        out["n_hbond"] += int((pol & (d >= HB_POLAR_MIN) & (d <= HB_POLAR_MAX)).sum())
        if new_is_anion is not None and new_is_anion[i]:
            out["n_salt"] += int((env.cationic[js] & (d <= SALT_MAX)).sum())
    if new_q is not None:
        out["coul"], out["n_cp"] = coulomb_and_cationpi(new_xyz, new_q, env)
    return out


def total_score(terms: dict, e_rel: float) -> float:
    """Lower is better. Weights chosen so one 0.5 A clash (~0.25) beats ~3 kcal/mol strain."""
    return (W["clash"] * terms["clash"] + W["strain"] * min(max(e_rel, 0.0), 60.0)
            - W["hbond"] * min(terms["n_hbond"], 4) - W["salt"] * min(terms["n_salt"], 3)
            - W["contact"] * min(terms["n_contact"], 40)
            + W["coul"] * float(np.clip(terms.get("coul", 0.0), -20.0, 20.0))
            - W["cationpi"] * min(terms.get("n_cp", 0), 2)
            + W["tors"] * terms.get("tors", 0.0)
            + W["tlib"] * terms.get("tlib", 0.0))


_ENV_Q = {("ARG", "NE"): 0.33, ("ARG", "NH1"): 0.33, ("ARG", "NH2"): 0.34, ("LYS", "NZ"): 1.0,
          ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5, ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
          ("SEP", "O1P"): -0.67, ("SEP", "O2P"): -0.67, ("SEP", "O3P"): -0.66,
          ("TPO", "O1P"): -0.67, ("TPO", "O2P"): -0.67, ("TPO", "O3P"): -0.66,
          ("PTR", "O1P"): -0.67, ("PTR", "O2P"): -0.67, ("PTR", "O3P"): -0.66,
          ("M3L", "NZ"): 1.0, ("MLY", "NZ"): 1.0, ("MLZ", "NZ"): 1.0}
_RINGS = {"PHE": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")], "TYR": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
          "TRP": [("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"), ("CG", "CD1", "NE1", "CE2", "CD2")],
          "PTR": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")], "TYS": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")]}


def _env_charges(res_name, atom_name) -> np.ndarray:
    return np.array([_ENV_Q.get((r, a), 0.0) for r, a in zip(res_name, atom_name)])


def _ring_centroids(env: "Environment") -> np.ndarray:
    if env.chain_id is None or not len(env.coord):
        return np.zeros((0, 3))
    cents = []
    key = {}
    for i, (c, r) in enumerate(zip(env.chain_id, env.res_id)):
        key.setdefault((c, int(r)), []).append(i)
    for (c, r), idx in key.items():
        rn = str(env.res_name[idx[0]])
        for ring in _RINGS.get(rn, ()):
            pos = [idx[list(env.atom_name[idx]).index(a)] for a in ring if a in set(env.atom_name[idx])]
            if len(pos) == len(ring):
                cents.append(env.coord[pos].mean(0))
    return np.array(cents) if cents else np.zeros((0, 3))


def new_atom_charges(symbols, neighbors_sym, formal=None) -> np.ndarray:
    """Heuristic physiological partial charges for newly built heavy atoms (see builder)."""
    return np.zeros(len(symbols))


def coulomb_and_cationpi(new_xyz: np.ndarray, new_q: np.ndarray, env: "Environment") -> tuple[float, int]:
    """Screened Coulomb (eps=4r) in kcal/mol between new charged atoms and env charges + cation-pi count."""
    if env.tree is None or not np.any(new_q != 0):
        return 0.0, 0
    e = 0.0
    n_cp = 0
    for i in np.where(new_q != 0)[0]:
        js = np.asarray(env.tree.query_ball_point(new_xyz[i], r=9.0), dtype=int)
        if len(js):
            q = env.charge[js]
            m = q != 0
            if m.any():
                d = np.maximum(np.linalg.norm(env.coord[js[m]] - new_xyz[i], axis=1), 2.2)
                e += float((83.0 * new_q[i] * q[m] / d ** 2).sum())
        if new_q[i] > 0 and len(env.ring_centroids):
            d = np.linalg.norm(env.ring_centroids - new_xyz[i], axis=1)
            n_cp += int(((d > 3.3) & (d < 5.6)).sum())
    return e, n_cp


# --------------------------------------------------------------------------- learned-ranker descriptors
DESC_KEYS = ["d_bur6_mean", "d_bur6_min", "d_bur10_mean", "d_bur10_min", "d_n_close35", "d_polar35", "d_polar40",
             "d_hb_bbN", "d_hb_bbO", "d_cat4", "d_cat6", "d_an4", "d_an6", "d_newcat_an4", "d_arom_min", "d_arom55",
             "d_hphob45", "d_own_N_min", "d_next_N_min", "d_prev_O_min", "d_extent", "d_exposed_frac", "d_n_new",
             "d_nonpolar_buried"]
_ANION_ENV = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"), ("SEP", "O1P"), ("SEP", "O2P"),
              ("SEP", "O3P"), ("TPO", "O1P"), ("TPO", "O2P"), ("TPO", "O3P"), ("PTR", "O1P"), ("PTR", "O2P"), ("PTR", "O3P")}


def _env_masks(env: "Environment"):
    if getattr(env, "_masks", None) is None:
        an = np.array([(r, a) in _ANION_ENV for r, a in zip(env.res_name, env.atom_name)])
        bbN = np.array([a == "N" for a in env.atom_name])
        bbO = np.array([a == "O" for a in env.atom_name])
        env._masks = (an, bbN, bbO)
    return env._masks


def descriptors(new_xyz, new_el, new_q, new_anion, env: "Environment", own_N=None, own_CB=None, site=None) -> dict:
    """Fixed-length descriptor dict (DESC_KEYS) of one conformer's NEW heavy atoms vs the environment.
    `site` = (chain, res_id) of the modified residue (to find i+1 / i-1 backbone atoms in env)."""
    out = {k: 0.0 for k in DESC_KEYS}
    n = len(new_xyz)
    out["d_n_new"] = float(n)
    if n == 0 or env is None or env.tree is None:
        return out
    an_env, bbN, bbO = _env_masks(env)
    polar_new = np.isin(new_el, ["N", "O"])
    carbon_new = new_el == "C"
    d_all = np.linalg.norm(env.coord[None, :, :] - new_xyz[:, None, :], axis=-1) if len(env.coord) < 4000 else None
    if d_all is None:
        d_all = np.stack([np.linalg.norm(env.coord - x, axis=1) for x in new_xyz])
    n6 = (d_all < 6.0).sum(1)
    n10 = (d_all < 10.0).sum(1)
    out["d_bur6_mean"], out["d_bur6_min"] = float(n6.mean()), float(n6.min())
    out["d_bur10_mean"], out["d_bur10_min"] = float(n10.mean()), float(n10.min())
    out["d_exposed_frac"] = float((n6 < 12).mean())
    out["d_n_close35"] = float((d_all < 3.5).sum())
    pe = env.polar[None, :] & polar_new[:, None]
    out["d_polar35"] = float(((d_all < 3.5) & pe).sum())
    out["d_polar40"] = float(((d_all < 4.0) & pe).sum())
    out["d_hb_bbN"] = float(((d_all < 3.4) & (d_all > 2.5) & (polar_new[:, None] & (new_el == "O")[:, None]) & bbN[None, :]).sum())
    out["d_hb_bbO"] = float(((d_all < 3.4) & (d_all > 2.5) & ((new_el == "N")[:, None]) & bbO[None, :]).sum())
    anion = np.asarray(new_anion, bool) if new_anion is not None else np.zeros(n, bool)
    cationic_new = np.asarray(new_q, float) > 0.15 if new_q is not None else np.zeros(n, bool)
    cat_e = env.cationic[None, :]
    out["d_cat4"] = float(((d_all < 4.0) & cat_e & anion[:, None]).sum())
    out["d_cat6"] = float(((d_all < 6.0) & cat_e & anion[:, None]).sum())
    out["d_an4"] = float(((d_all < 4.0) & an_env[None, :] & anion[:, None]).sum())
    out["d_an6"] = float(((d_all < 6.0) & an_env[None, :] & anion[:, None]).sum())
    out["d_newcat_an4"] = float(((d_all < 4.5) & an_env[None, :] & cationic_new[:, None]).sum())
    if len(env.ring_centroids):
        da = np.linalg.norm(env.ring_centroids[None, :, :] - new_xyz[:, None, :], axis=-1)
        out["d_arom_min"] = float(da.min())
        out["d_arom55"] = float((da < 5.5).sum())
    else:
        out["d_arom_min"] = 20.0
    ce = np.char.startswith(env.element.astype(str), "C")
    out["d_hphob45"] = float(((d_all < 4.5) & ce[None, :] & carbon_new[:, None]).sum())
    out["d_nonpolar_buried"] = float(((n6 > 18) & carbon_new).sum())
    if own_N is not None and polar_new.any():
        out["d_own_N_min"] = float(np.linalg.norm(new_xyz[polar_new] - own_N, axis=1).min())
    else:
        out["d_own_N_min"] = 15.0
    if own_CB is not None:
        out["d_extent"] = float(np.linalg.norm(new_xyz - own_CB, axis=1).max())
    if site is not None and env.chain_id is not None:
        c, r = site
        same = env.chain_id == c
        nxt = same & (env.res_id == r + 1) & bbN
        prv = same & (env.res_id == r - 1) & bbO
        out["d_next_N_min"] = float(d_all[:, nxt].min()) if nxt.any() else 15.0
        out["d_prev_O_min"] = float(d_all[:, prv].min()) if prv.any() else 15.0
    else:
        out["d_next_N_min"] = out["d_prev_O_min"] = 15.0
    return out


# --------------------------------------------------------------------------- extended shell descriptors (ranker v4)
ENV_CLASSES = ["C", "N_bb", "N_cat", "N_other", "O_bb", "O_an", "O_other", "S"]
SHELLS = [(0.0, 3.2), (3.2, 4.0), (4.0, 5.0), (5.0, 6.5), (6.5, 8.5)]
EXT_KEYS = [f"e_{grp}_{c}_{i}" for grp in ("all", "an") for c in ENV_CLASSES for i in range(len(SHELLS))]


def _env_class(env: "Environment") -> np.ndarray:
    if getattr(env, "_cls", None) is None:
        an_env, bbN, bbO = _env_masks(env)
        el = env.element.astype(str)
        cls = np.full(len(el), 7, dtype=int)            # default S/other
        cls[el == "C"] = 0
        n = el == "N"
        cls[n] = 3
        cls[n & bbN] = 1
        cls[n & env.cationic] = 2
        o = el == "O"
        cls[o] = 6
        cls[o & bbO] = 4
        cls[o & an_env] = 5
        env._cls = cls
    return env._cls


def ext_descriptors(new_xyz, new_anion, env: "Environment") -> dict:
    """Sum over new atoms of env-atom counts per class and distance shell; separately for anionic new atoms."""
    out = {k: 0.0 for k in EXT_KEYS}
    if env is None or env.tree is None or len(new_xyz) == 0:
        return out
    cls = _env_class(env)
    d = np.stack([np.linalg.norm(env.coord - x, axis=1) for x in new_xyz])          # (n_new, M)
    anion = np.asarray(new_anion, bool) if new_anion is not None else np.zeros(len(new_xyz), bool)
    for si, (lo, hi) in enumerate(SHELLS):
        inside = (d >= lo) & (d < hi)
        for ci in range(len(ENV_CLASSES)):
            m = inside & (cls[None, :] == ci)
            out[f"e_all_{ENV_CLASSES[ci]}_{si}"] = float(m.sum())
            if anion.any():
                out[f"e_an_{ENV_CLASSES[ci]}_{si}"] = float(m[anion].sum())
    return out
