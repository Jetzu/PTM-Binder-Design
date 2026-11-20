"""Particle guidance (SMC / weighted ensemble) for stock RFD3, as a runtime patch (no foundry edits).

How it hooks in (foundry e412591, models/rfd3/src/rfd3):
* `AADesignTrainer.validation_step` (trainer/rfd3.py:198) is wrapped at class level. Before the forward it builds boolean atom
  masks over the L atoms of the example (`example["atom_array"]` is in the same order as the sampler's X_L, trainer/rfd3.py:339
  via trainer_utils.py:69-80): PTM atoms, other fixed target atoms, binder backbone+CB atoms. After the forward it writes each
  particle's weight, reward terms and lineage into `prediction_metadata[idx]["ptmsmc"]` (dumped next to `<id>_model_<idx>`),
  plus the full per-step log to `$PTMSMC_LOG_DIR/<example_id>.json`.
* `SampleDiffusionWithMotif.sample_diffusion_like_af3` (model/inference_sampler.py:146) is wrapped: the diffusion module it
  receives is replaced by a proxy. The loop calls `outs = diffusion_module(X_noisy_L=X_noisy_L, ...)` (:260), takes
  `X_denoised_L = outs["X_L"]` (:268) and updates `X_L = X_noisy_L + step_scale * d_t * (X_noisy_L - X_denoised_L) / t_hat`
  (:320), row by row. The proxy scores the particles on `outs["X_L"]` (the x0 prediction) and, when the engine resamples,
  reindexes the loop's own `X_noisy_L` IN PLACE and returns `outs` reindexed the same way, so the update produces exactly the
  resampled next state. Nothing else carries across steps (recycling is within one call; f["attn_indices"] is rebuilt from
  X_noisy_L every step). The CFG reference pass (f is f_ref, :285-291) is passed through untouched.
* Resampling is clamped to steps that still inject churn noise (c_t > gamma_min, :214): later copies would stay identical.

Switch on with PTMSMC_SCHEME=smc|we|twe|none (none = score + log only). Other knobs: ptm_rfd3.particles.ParticlePolicy
fields as PTMSMC_<FIELD>, the reward as PTMSMC_R0/D0/N_SAT/ALPHA/CLASH_W, window in noise level PTMSMC_SIGMA_START (A,
default 20) / PTMSMC_SIGMA_STOP (A, default 1.5) unless PTMSMC_START / PTMSMC_STOP (progress) are given, PTM atoms
PTMSMC_PTM_ATOMS="PTR:OH,P,O1P,O2P,O3P;SEP:OG,P,O1P,O2P,O3P" (default: ptm_rfd3.rfd3_inputs.detect_sites on the fixed atoms).
The particle count is the diffusion batch size (diffusion_batch_size=K).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .particles import BurialReward, ParticleEngine, ParticlePolicy, fold_clusters

log = logging.getLogger("ptm_rfd3.rfd3_particles")
_ORIG: dict = {}
_STATE: dict = {}          # masks / results of the example being processed (one example at a time per process)
BINDER_ATOMS = ("N", "CA", "C", "O", "CB")
BACKBONE = ("N", "CA", "C", "O", "OXT")


def enabled() -> bool:
    """On when PTMSMC_SCHEME is set (not 'off') or a PTMSMC_CONFIG file is given."""
    if os.environ.get("PTMSMC_SCHEME", "") == "off":
        return False
    return bool(os.environ.get("PTMSMC_SCHEME") or os.environ.get("PTMSMC_CONFIG"))


def reward_from_env(env=os.environ):
    """(policy overrides from PTMSMC_CONFIG, RewardModel): ptm_rfd3.reward_model (multi-term, scheduled, tunable)."""
    from .reward_model import load

    return load(env, generator="rfd3")


def _terms(reward, X, masks, outs=None, progress: float = 1.0):
    """All reward terms for the particles X [K,L,3]; sequence probabilities from the step's sequence logits [K,I,32]
    (AF3 order, the 20 amino acids at 0..19), when the module returned them."""
    probs = None
    logits = outs.get("sequence_logits_I") if isinstance(outs, dict) else outs
    r = masks["res"]
    if torch.is_tensor(logits) and logits.ndim == 3:
        probs = torch.softmax(logits[..., :20].float(), -1).index_select(1, r["tok"].to(logits.device))
    return reward.evaluate(X, masks["ptm"], masks["other"], masks["binder"], chem=masks.get("chem"), ca_idx=r["ca_idx"],
                           cb_idx=r["cb_idx"], bbpol_idx=r["bbpol_idx"], probs=probs, progress=progress,
                           binder_all=masks.get("binder_all"))


def _parse_ptm_atoms(spec: str) -> dict[str, set[str]]:
    out = {}
    for part in spec.split(";"):
        if part.strip():
            res, names = part.split(":")
            out[res.strip()] = {n.strip() for n in names.split(",") if n.strip()}
    return out


def ptm_atom_mask(aa, fixed: np.ndarray) -> tuple[np.ndarray, str]:
    """Boolean [L] mask of the modification atoms among the fixed (target) atoms, and how it was found."""
    names = np.asarray(aa.atom_name).astype(str)
    resn = np.asarray(aa.res_name).astype(str)
    spec = os.environ.get("PTMSMC_PTM_ATOMS", "")
    if spec:
        m = np.zeros(len(names), bool)
        for res, ns in _parse_ptm_atoms(spec).items():
            m |= fixed & (resn == res) & np.isin(names, list(ns))
        return m, "env"
    try:  # the pipeline's own definition (anchor + atoms not mapped to the parent residue)
        from .rfd3_inputs import detect_sites

        idx = np.where(fixed)[0]
        sub = aa[idx]
        m = np.zeros(len(names), bool)
        for s in detect_sites(sub):
            sel = (np.asarray(sub.chain_id).astype(str) == s.chain) & (np.asarray(sub.res_id) == s.res_id) \
                & np.isin(np.asarray(sub.atom_name).astype(str), list(s.ptm_atoms))
            m[idx[sel]] = True
        if m.any():
            return m, "detect_sites"
    except Exception as e:  # noqa: BLE001 - fall back below, but say why
        log.warning("ptmsmc: detect_sites on the example failed (%s); falling back to non-standard residues", e)
    from .af3_inputs import AA3

    m = fixed & ~np.isin(resn, list(AA3)) & ~np.isin(names, list(BACKBONE))
    return m, "nonstandard"


def build_masks(aa, device) -> dict:
    fixed = np.asarray(aa.is_motif_atom_with_fixed_coord).astype(bool)
    names = np.asarray(aa.atom_name).astype(str)
    elem = np.asarray(aa.element).astype(str)
    real = (elem != "VX") & (elem != "H")
    ptm, how = ptm_atom_mask(aa, fixed)
    other = fixed & ~ptm & real
    binder = ~fixed & np.isin(names, BINDER_ATOMS)
    if not ptm.any():
        raise RuntimeError("ptmsmc: no PTM atoms found among the fixed atoms (set PTMSMC_PTM_ATOMS)")
    if not binder.any():
        raise RuntimeError("ptmsmc: no designed (binder) atoms found")
    t = lambda m: torch.as_tensor(m, dtype=torch.bool, device=device)
    li = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
    ca = np.where(~fixed & (names == "CA"))[0]
    cb = np.where(~fixed & (names == "CB"))[0]
    tok = np.asarray(aa.token_id)
    cb_by_tok = {int(tok[i]): int(i) for i in cb}
    cb_for_ca = np.array([cb_by_tok.get(int(tok[i]), -1) for i in ca])
    res = dict(ca_idx=li(ca), cb_idx=li(cb_for_ca) if (cb_for_ca >= 0).all() else None, tok=li(tok[ca]),
               bbpol_idx=li(np.where(~fixed & np.isin(names, ("N", "O")))[0]))
    chem = None
    try:
        from .ptm_physics import ptm_chem_for
        chem = ptm_chem_for(aa, np.where(ptm)[0])
    except Exception as e:  # noqa: BLE001 - physics terms are optional; burial still works
        log.warning("ptmsmc: PTM chemistry could not be derived (%s); physics terms off", e)
    binder_all = ~fixed & (elem != "H")                                   # every predicted binder atom incl. side-chain slots
    return dict(ptm=t(ptm), other=t(other), binder=t(binder), binder_all=t(binder_all), how=how, res=res, chem=chem,
                ptm_atoms=sorted({f"{c}{r}:{n}" for c, r, n in zip(np.asarray(aa.chain_id)[ptm],
                                                                    np.asarray(aa.res_id)[ptm], names[ptm])}))


class _Proxy:
    """Stands in for the diffusion module inside one sample_diffusion_like_af3 call (see module docstring)."""

    def __init__(self, module, f_main, engine, reward, masks, nsteps, last_noisy_step):
        self.m, self.f_main, self.engine, self.reward, self.masks = module, f_main, engine, reward, masks
        self.nsteps, self.last_noisy = nsteps, last_noisy_step
        self.k = 0
        self.k_window = 0

    def __getattr__(self, name):
        return getattr(self.m, name)

    def _forward(self, *args, X_noisy_L, f, **kw):
        """The denoiser call, optionally in chunks of policy.chunk particles (rows are independent given the shared f;
        per-row arguments are X_noisy_L and the tiled t)."""
        D = X_noisy_L.shape[0]
        ch = self.engine.policy.chunk
        if not ch or D <= ch:
            return self.m(*args, X_noisy_L=X_noisy_L, f=f, **kw)
        t = kw.pop("t")
        parts = [self.m(*args, X_noisy_L=X_noisy_L[i:i + ch], f=f, t=t[i:i + ch], **kw) for i in range(0, D, ch)]
        if isinstance(parts[0], dict):
            return {k: (torch.cat([q[k] for q in parts], 0) if torch.is_tensor(parts[0][k]) and parts[0][k].ndim
                        and parts[0][k].shape[0] == X_noisy_L[:ch].shape[0] else parts[0][k]) for k in parts[0]}
        return torch.cat(parts, 0)

    def __call__(self, *args, X_noisy_L, f, **kw):
        if f is not self.f_main:  # CFG reference pass
            return self._forward(*args, X_noisy_L=X_noisy_L, f=f, **kw)
        outs = self._forward(*args, X_noisy_L=X_noisy_L, f=f, **kw)
        step = self.k
        self.k += 1
        progress = (step + 1) / self.nsteps
        if not self.engine.window(progress):
            return outs
        if step > self.last_noisy and self.engine.policy.scheme != "none":  # no churn left: copies would stay identical
            return outs
        self.k_window += 1
        if not self.engine.wants_score(self.k_window - 1):
            return outs
        X0 = outs["X_L"] if isinstance(outs, dict) else outs
        tt = _terms(self.reward, X0, self.masks, outs, progress=progress)
        cl = fold_clusters(X0[:, self.masks["res"]["ca_idx"]], X0[:, self.masks["ptm"]], self.engine.policy.fold_thr)
        parents = self.engine.score(tt["reward"], progress, step=step, clusters=cl)
        self.engine.log[-1]["terms"] = {k: [round(float(x), 4) for x in v.detach().cpu().flatten()]
                                        for k, v in tt.items() if k != "reward"}
        if parents is None:
            return outs
        D = X_noisy_L.shape[0]
        idx = parents.to(X_noisy_L.device)
        X_noisy_L.copy_(X_noisy_L.index_select(0, idx))
        if isinstance(outs, dict):
            return {k: (v.index_select(0, idx.to(v.device)) if torch.is_tensor(v) and v.ndim and v.shape[0] == D else v)
                    for k, v in outs.items()}
        return outs.index_select(0, idx)


def _policy_for(self, nsched: torch.Tensor, D: int, cfg_policy: dict | None = None) -> tuple[ParticlePolicy, int]:
    env = dict(os.environ)
    for k, v in (cfg_policy or {}).items():  # PTMSMC_CONFIG policy section; explicit env vars win
        env.setdefault(k, v)
    env["PTMSMC_N_PARTICLES"] = str(D)
    nsteps = len(nsched) - 1
    sig = nsched[1:].double().cpu()  # c_t of step i
    gmin = float(getattr(self, "gamma_min", 1.0))
    noisy = (sig > gmin).nonzero().flatten() if float(getattr(self, "gamma_0", 0.0)) > 0 else torch.empty(0, dtype=torch.long)
    last_noisy = int(noisy.max()) if noisy.numel() else -1
    if not env.get("PTMSMC_START"):
        s0 = float(env.get("PTMSMC_SIGMA_START", "20"))
        i0 = int((sig <= s0).nonzero().flatten().min()) if bool((sig <= s0).any()) else nsteps - 1
        env["PTMSMC_START"] = str((i0 + 1) / nsteps)
    if not env.get("PTMSMC_STOP"):
        s1 = float(env.get("PTMSMC_SIGMA_STOP", "1.5"))
        i1 = int((sig >= s1).nonzero().flatten().max()) if bool((sig >= s1).any()) else 0
        i1 = min(i1, last_noisy) if last_noisy >= 0 else i1
        env["PTMSMC_STOP"] = str((i1 + 1) / nsteps)
    pol = ParticlePolicy.from_env(env)
    if pol.scheme in ("smc", "we", "twe") and last_noisy < 0:
        raise RuntimeError("ptmsmc: the sampler injects no noise (gamma_0 = 0): resampled particles would stay identical")
    if pol.stop > (last_noisy + 1) / nsteps and pol.scheme != "none":
        log.warning("ptmsmc: window stop %.3f is past the last noisy step %d/%d; resampling stops there",
                    pol.stop, last_noisy, nsteps)
    return pol, last_noisy


def install() -> None:
    """Patch the sampler and the trainer (idempotent). Called by ptm_rfd3.rfd3_patch.main when PTMSMC_SCHEME is set."""
    if "sample" in _ORIG:
        return
    from rfd3.model.inference_sampler import SampleDiffusionWithMotif
    from rfd3.trainer.rfd3 import AADesignTrainer

    _ORIG["sample"] = SampleDiffusionWithMotif.sample_diffusion_like_af3
    _ORIG["validation_step"] = AADesignTrainer.validation_step

    def sample_diffusion_like_af3(self, *, f, diffusion_module, diffusion_batch_size, **kw):
        masks = _STATE.get("masks")
        if masks is None:
            return _ORIG["sample"](self, f=f, diffusion_module=diffusion_module,
                                   diffusion_batch_size=diffusion_batch_size, **kw)
        dev = kw["coord_atom_lvl_to_be_noised"].device
        nsched = self._construct_inference_noise_schedule(device=dev, partial_t=f.get("partial_t", None))
        cfg_pol, reward = reward_from_env()
        pol, last_noisy = _policy_for(self, nsched, diffusion_batch_size, cfg_pol)
        engine = ParticleEngine(pol)
        proxy = _Proxy(diffusion_module, f, engine, reward, masks, len(nsched) - 1, last_noisy)
        out = _ORIG["sample"](self, f=f, diffusion_module=proxy, diffusion_batch_size=diffusion_batch_size, **kw)
        terms = _terms(reward, out["X_L"], masks, {"sequence_logits_I": out.get("sequence_logits_I")}, progress=1.0)
        res = engine.finish(terms["reward"])
        par = res["parents"]
        if not bool((par == torch.arange(len(par))).all()):  # terminal resampling: reorder every per-particle output
            D = out["X_L"].shape[0]
            for k, v in list(out.items()):
                if torch.is_tensor(v) and v.ndim and v.shape[0] == D:
                    out[k] = v.index_select(0, par.to(v.device))
            terms = {k: v.index_select(0, par.to(v.device)) for k, v in terms.items()}
        from .reward_model import describe
        _STATE["result"] = dict(res, reward_terms=describe(reward), terms={k: v.detach().cpu().tolist() for k, v in terms.items()},
                                last_noisy_step=last_noisy, nsteps=len(nsched) - 1)
        return out

    def validation_step(self, batch, *a, **kw):
        if not enabled():
            return _ORIG["validation_step"](self, batch, *a, **kw)
        example = batch[0] if not isinstance(batch, dict) else batch
        dev = example["coord_atom_lvl_to_be_noised"].device
        _STATE.clear()
        _STATE["masks"] = build_masks(example["atom_array"], dev)
        try:
            out = _ORIG["validation_step"](self, batch, *a, **kw)
            res = _STATE.get("result")
            if res is not None:
                m = _STATE["masks"]
                meta = out.get("prediction_metadata")
                for i in range(len(res["weights"])):
                    info = dict(scheme=res["policy"]["scheme"], weight=float(res["weights"][i]),
                                lineage=int(res["ancestry"][i]), lam=float(res["lam"]),
                                **{k: float(v[i]) for k, v in res["terms"].items()})
                    if meta is not None and i < len(meta):
                        meta[i]["ptmsmc"] = info
                log_dir = os.environ.get("PTMSMC_LOG_DIR")
                if log_dir:
                    Path(log_dir).mkdir(parents=True, exist_ok=True)
                    rec = dict(example_id=example["example_id"], policy=res["policy"], reward_terms=res["reward_terms"], ptm_atoms=m["ptm_atoms"],
                               ptm_mask_source=m["how"], n_ptm=int(m["ptm"].sum()), n_other=int(m["other"].sum()),
                               n_binder=int(m["binder"].sum()), last_noisy_step=res["last_noisy_step"],
                               nsteps=res["nsteps"], terms=res["terms"], weights=res["weights"].tolist(),
                               ancestry=res["ancestry"].tolist(), log=res["log"])
                    (Path(log_dir) / f"{example['example_id']}.ptmsmc.json").write_text(json.dumps(rec))
            return out
        finally:
            _STATE.clear()

    SampleDiffusionWithMotif.sample_diffusion_like_af3 = sample_diffusion_like_af3
    AADesignTrainer.validation_step = validation_step
    log.warning("ptm_rfd3: installed particle guidance (PTMSMC_SCHEME=%s)", os.environ.get("PTMSMC_SCHEME"))
