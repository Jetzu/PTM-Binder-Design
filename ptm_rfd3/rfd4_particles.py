"""Particle guidance (SMC / weighted ensemble) for RFD4-Proteina, as a runtime patch (the rfproteina package is not edited).

Hook (rfproteina, AIPD container rfd4-proteina-cu128.sif; lab copy ed63183 src/rfproteina/flow_matching/
product_space_flow_matcher.py): `ProductSpaceFlowMatcher.full_simulation` (:701) runs, per step,
    batch["x_t"] = x;  nn_out = self.get_clean_pred_n_guided_vector(batch=..., ...);  x_1_pred = ...(nn_out)
    x = self.simulation_step(x_t=x, nn_out=nn_out, ...)
row by row over the nsamples particles (:796-855). We wrap full_simulation and, for the duration of the call, shadow the
instance's `get_clean_pred_n_guided_vector`: after the network call the particles are scored on the clean prediction
nn_out[dm]["x_1"], and when the engine resamples, the loop's own state dict (`batch["x_t"]` IS the loop's `x`) is reindexed
IN PLACE and nn_out is returned reindexed the same way, so `simulation_step` produces exactly the resampled next state.
Same trick as ptm_rfd3.rfd3_particles; the same engine (ptm_rfd3.particles).

Reward: RFD4's state is CA (+ latents) per token, so the binder is represented by its predicted CA atoms. The PTM residue
(e.g. PTR, an NCAA token, one token not atoms) contributes its conditioned PTM atom coordinates (C_CRD atom37 slots),
translated with the token's predicted CA (target tokens are generated too, held only softly to C_CRD). Other target tokens:
predicted CA. Defaults r0 = 8 A, d0 = 3.8 A for CA-level counts (override PTMSMC_R0 / PTMSMC_D0 / PTMSMC_N_SAT).

Requirements (checked): batch_size = num_replicates-multiple with ONE example per batch (identical token layout across
particles; fixed binder length), guidance_w == 1 (no CFG second pass).

Run: python -m ptm_rfd3.rfd4_particles <rfd4-generate hydra overrides>, with PTMSMC_SCHEME=smc|we|twe|none,
PTMSMC_PTM_ATOMS="PTR:OH,P,O1P,O2P,O3P" and the ParticlePolicy / BurialReward fields as PTMSMC_<FIELD>.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from pathlib import Path

import torch

from .particles import BurialReward, ParticleEngine, ParticlePolicy, fold_clusters

log = logging.getLogger("ptm_rfd3.rfd4_particles")
_ORIG: dict = {}
_CALLS = {"n": 0}
CA_SLOT = 1


def enabled() -> bool:
    """On when PTMSMC_SCHEME is set (not 'off') or a PTMSMC_CONFIG file is given."""
    if os.environ.get("PTMSMC_SCHEME", "") == "off":
        return False
    return bool(os.environ.get("PTMSMC_SCHEME") or os.environ.get("PTMSMC_CONFIG"))


def reward_from_env(env=os.environ):
    """(policy overrides from PTMSMC_CONFIG, RewardModel) with CA-level geometry defaults (ptm_rfd3.reward_model)."""
    from .reward_model import load

    return load(env, generator="rfd4")


def _parse_ptm_atoms(spec: str) -> dict[str, list[str]]:
    out = {}
    for part in spec.split(";"):
        if part.strip():
            res, names = part.split(":")
            out[res.strip()] = [n.strip() for n in names.split(",") if n.strip()]
    return out


def _nested_reindex(obj, idx, K):
    if torch.is_tensor(obj):
        return obj.index_select(0, idx.to(obj.device)) if obj.ndim and obj.shape[0] == K else obj
    if isinstance(obj, dict):
        return {k: _nested_reindex(v, idx, K) for k, v in obj.items()}
    return obj


class Geometry:
    """Static per-example index sets + the per-step coordinate assembly for the burial reward."""

    def __init__(self, batch: dict):
        from atomworks.ml.conditions import C_CRD
        from rfproteina.datapipes.schema.proteina import TokenType
        from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

        pad = batch["padding_mask"].bool()
        for key in ("padding_mask", "token_type", "coord_cond_mask", "residue_type"):
            v = batch.get(key)
            if torch.is_tensor(v) and v.shape[0] > 1 and bool((v != v[:1]).any()):
                raise RuntimeError(f"ptmsmc/rfd4: batch[{key!r}] differs across particles (one example per batch, fixed "
                                   "binder length required: batch_size divides num_replicates and the contig length is fixed)")
        pad = pad[0]
        cc = batch["coord_cond_mask"][0].bool() if "coord_cond_mask" in batch else None
        if cc is None:
            raise RuntimeError("ptmsmc/rfd4: no coord_cond_mask in the batch (target must be coordinate-conditioned)")
        self.binder = pad & ~cc
        self.target = pad & cc
        ttype = batch["token_type"][0]
        rtype = batch["residue_type"][0]
        crd = batch[C_CRD.full_name][0].float()          # [n, 37, 3] Angstrom, NaN where not conditioned
        spec = _parse_ptm_atoms(os.environ.get("PTMSMC_PTM_ATOMS", ""))
        tok, slot, self.ptm_names = [], [], []
        for res, names in spec.items():
            if res not in ENC.token_atoms:
                continue  # not in the vocabulary -> atomized, handled below
            rid = ENC.token_to_idx[res]
            layout = list(ENC.token_atoms[res])
            sl = [(layout.index(n), res, n) for n in names if n in layout]
            for t in ((rtype == rid) & self.target).nonzero().flatten().tolist():
                for s, r_, n_ in sl:
                    tok.append(t); slot.append(s); self.ptm_names.append((r_, n_))
        self.ptm_tok = torch.tensor(tok, dtype=torch.long, device=pad.device)
        self.ptm_ref = crd[self.ptm_tok, torch.tensor(slot, dtype=torch.long, device=pad.device)] if tok else None
        self.ptm_ca_ref = crd[self.ptm_tok, CA_SLOT] if tok else None
        self.ptm_atomized = (ttype == TokenType.ATOMIZED) & self.target
        if not tok and not bool(self.ptm_atomized.any()):
            raise RuntimeError(f"ptmsmc/rfd4: no PTM atoms found (PTMSMC_PTM_ATOMS={os.environ.get('PTMSMC_PTM_ATOMS')!r}, "
                               "no atomized target tokens)")
        if self.ptm_ref is not None and not bool(torch.isfinite(self.ptm_ref).all() & torch.isfinite(self.ptm_ca_ref).all()):
            raise RuntimeError("ptmsmc/rfd4: PTM atoms are not coordinate-conditioned (C_CRD) in the input")
        ptm_res = torch.zeros_like(pad)
        if tok:
            ptm_res[self.ptm_tok] = True
        self.other = self.target & ~ptm_res & ~self.ptm_atomized
        self.info = dict(n_binder=int(self.binder.sum()), n_target=int(self.target.sum()), n_ptm_atoms=len(tok),
                         n_ptm_atomized=int(self.ptm_atomized.sum()), ptm_tokens=sorted(set(tok)))
        self.binder_tok = self.binder.nonzero().flatten()
        self.chem = None
        if self.ptm_names and not bool(self.ptm_atomized.any()):
            try:  # PTM chemistry from the CCD entry of the residue token (biotite's bundled CCD)
                import numpy as np
                from biotite.structure import info as binfo

                from .ptm_physics import PTMChem, group_charges, residue_graph

                q, el = [], []
                cache = {}
                for r_, n_ in self.ptm_names:
                    if r_ not in cache:
                        a = binfo.residue(r_)
                        a = a[(a.element != "H") & (a.atom_name != "OXT")]
                        e, adj, ring = residue_graph(a, np.arange(len(a)))
                        cache[r_] = {str(n): (float(c), str(x)) for n, c, x in zip(a.atom_name, group_charges(e, adj, ring), a.element)}
                    c, x = cache[r_][n_]
                    q.append(c); el.append(x)
                self.chem = PTMChem.from_lists(el, q)
            except Exception as e:  # noqa: BLE001
                log.warning("ptmsmc/rfd4: PTM chemistry unavailable (%s); physics terms off", e)
        self.info["ptm_charges"] = self.chem.charge.tolist() if self.chem is not None else None

    def terms(self, reward, bb_ca_nm: torch.Tensor, seq_logits: torch.Tensor | None, progress: float = 1.0):
        X, P, Q, B = self.assemble(bb_ca_nm)
        probs = None
        if torch.is_tensor(seq_logits) and seq_logits.ndim == 3:
            probs = torch.softmax(seq_logits[..., 1:21].float(), -1).index_select(1, self.binder_tok.to(seq_logits.device))
        ca_idx = B.nonzero().flatten()                                       # binder CA in token order
        return reward.evaluate(X, P, Q, B, chem=self.chem, ca_idx=ca_idx, cb_idx=None, bbpol_idx=None, probs=probs,
                               progress=progress)

    def assemble(self, bb_ca_nm: torch.Tensor):
        """bb_ca [K, n, 3] in nm -> X [K, m, 3] Angstrom with masks (ptm, other, binder) over m."""
        X = 10.0 * bb_ca_nm.detach().float()
        parts, kinds = [], []
        if self.ptm_ref is not None:
            shift = X[:, self.ptm_tok] - self.ptm_ca_ref.to(X.device)        # [K, a, 3]: follow the generated PTR token
            parts.append(self.ptm_ref.to(X.device) + shift); kinds.append(0)
        if bool(self.ptm_atomized.any()):
            parts.append(X[:, self.ptm_atomized]); kinds.append(0)
        parts.append(X[:, self.other]); kinds.append(1)
        parts.append(X[:, self.binder]); kinds.append(2)
        lab = torch.cat([torch.full((p.shape[1],), k, dtype=torch.long) for p, k in zip(parts, kinds)]).to(X.device)
        return torch.cat(parts, 1), lab == 0, lab == 1, lab == 2


def install() -> None:
    if "full" in _ORIG:
        return
    from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher as PSFM

    _ORIG["full"] = PSFM.full_simulation
    sig = inspect.signature(_ORIG["full"])

    def full_simulation(self, *args, **kwargs):
        if not enabled():
            return _ORIG["full"](self, *args, **kwargs)
        b = sig.bind(self, *args, **kwargs)
        b.apply_defaults()
        batch, nsteps, nsamples = b.arguments["batch"], b.arguments["nsteps"], b.arguments["nsamples"]
        if b.arguments.get("guidance_w", 1.0) != 1.0:
            raise RuntimeError("ptmsmc/rfd4: CFG (guidance_w != 1) is not supported with particle guidance")
        if os.environ.get("PTMSMC_DEBUG") == "1" and _CALLS["n"] == 0:
            print("[ptmsmc] batch keys:", {k: (tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else type(v).__name__
                                           for k, v in batch.items()}, flush=True)
        geom = Geometry(batch)
        cfg_pol, reward = reward_from_env()
        env = dict(os.environ)
        for k_, v_ in cfg_pol.items():  # PTMSMC_CONFIG policy section; explicit env vars win
            env.setdefault(k_, v_)
        env["PTMSMC_N_PARTICLES"] = str(nsamples)
        env.setdefault("PTMSMC_START", "0.25")
        env.setdefault("PTMSMC_STOP", "0.9")
        pol = ParticlePolicy.from_env(env)
        engine = ParticleEngine(pol)
        orig_g = self.get_clean_pred_n_guided_vector
        state = {"k": 0, "kw": 0}

        def guided(*, batch, predict_for_sampling, guidance_w):
            nn_out = orig_g(batch=batch, predict_for_sampling=predict_for_sampling, guidance_w=guidance_w)
            step = state["k"]
            state["k"] += 1
            progress = (step + 1) / nsteps
            if not engine.window(progress):
                return nn_out
            state["kw"] += 1
            if not engine.wants_score(state["kw"] - 1):
                return nn_out
            tt = geom.terms(reward, nn_out["bb_ca"]["x_1"], nn_out.get("sequence", {}).get("logits"), progress=progress)
            Xg, Pm, _Qm, Bm = geom.assemble(nn_out["bb_ca"]["x_1"])
            cl = fold_clusters(Xg[:, Bm], Xg[:, Pm], engine.policy.fold_thr)
            parents = engine.score(tt["reward"], progress, step=step, clusters=cl)
            engine.log[-1]["terms"] = {k: [round(float(v_), 4) for v_ in v.detach().cpu().flatten()]
                                       for k, v in tt.items() if k != "reward"}
            if parents is None:
                return nn_out
            x = batch["x_t"]
            idx = parents.to(next(iter(x.values())).device)
            for dm in list(x):
                x[dm].copy_(x[dm].index_select(0, idx.to(x[dm].device)))
            return _nested_reindex(nn_out, idx, nsamples)

        self.get_clean_pred_n_guided_vector = guided
        try:
            x, info = _ORIG["full"](*b.args, **b.kwargs)
        finally:
            del self.get_clean_pred_n_guided_vector
        terms = geom.terms(reward, x["bb_ca"], info.get("sequence_logits"), progress=1.0)
        if pol.terminal_resample:
            raise RuntimeError("ptmsmc/rfd4: terminal_resample is not supported (outputs are written per particle)")
        res = engine.finish(terms["reward"])
        k = _CALLS["n"]
        _CALLS["n"] += 1
        log_dir = os.environ.get("PTMSMC_LOG_DIR")
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            ids = batch.get("example_id") or batch.get("id") or batch.get("key")
            from .reward_model import describe
            rec = dict(call=k, example_ids=ids if isinstance(ids, (list, tuple)) else None, policy=res["policy"],
                       reward_terms=describe(reward),
                       geometry=geom.info, nsteps=nsteps, terms={a: v.detach().cpu().tolist() for a, v in terms.items()},
                       weights=res["weights"].tolist(), ancestry=res["ancestry"].tolist(), lam=res["lam"], log=res["log"])
            (Path(log_dir) / f"rfd4_call{k:04d}.ptmsmc.json").write_text(json.dumps(rec))
        return x, info

    PSFM.full_simulation = full_simulation
    log.warning("ptm_rfd3: installed RFD4 particle guidance (PTMSMC_SCHEME=%s)", os.environ.get("PTMSMC_SCHEME"))


def main() -> None:
    install()
    import rfproteina.generate as G

    G.main_hydra()


if __name__ == "__main__":
    sys.exit(main())
