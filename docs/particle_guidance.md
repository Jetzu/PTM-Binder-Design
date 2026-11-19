# Particle guidance: steering RFD3 / RFD4 toward binders that bury the modification

Inference-time steering for both generators of this pipeline, with no model change: K particles (designs) are
sampled together and, while they are being generated, re-weighted and resampled by a reward computed on each particle's
predicted final structure. The primary reward is **burial of the post-translational modification** by the binder; more
terms (contact share, cheap PTM-general physics, compactness, ...) can be added, weighted and scheduled over diffusion
time from one config file.

> **Status: research code.** The mechanics are tested (CPU unit tests, GPU smoke runs on EGFR pTyr1138); which
> settings help is being measured. Treat any setting as a starting point and confirm designs downstream (AF3, both
> target states) as for unguided runs.

## Quick start

```bash
# RFD3 (stock foundry; the pipeline's patched entry point picks the config up)
PTMSMC_CONFIG=examples/particle_guidance/burial_smc.yaml \
  python -m ptm_rfd3.rfd3_patch ckpt_path=$RFD3_CKPT inputs=examples/specs/egfr_pY1138.json out_dir=out \
  diffusion_batch_size=16 n_batches=4

# RFD4-Proteina (AIPD container; scripts/slurm/rfd4_design.sbatch forwards every PTMSMC_* variable)
PTMSMC_CONFIG=$PWD/examples/particle_guidance/burial_smc.yaml SPECS=examples/specs/egfr_pY1138.json OUTSUB=egfr \
  LENGTHS=90 NREP=32 BS=16 sbatch scripts/slurm/rfd4_design.sbatch
```

- The **diffusion batch is the particle population**: `diffusion_batch_size` (RFD3) or `batch_size` (RFD4) = K.
- Every batch is an independent population.
- Outputs are written as usual, one structure per particle.
- Each design's metadata JSON gets a `ptmsmc` entry with:
  - its final importance **weight**;
  - its **lineage** (which initial particle it descends from);
  - every reward term.
- The full per-step log (rewards, weights, effective sample size (ESS), resampling events, fold clusters) goes to
  `out/ptmsmc/<example>.ptmsmc.json`.

Unset `PTMSMC_CONFIG` / `PTMSMC_SCHEME` and nothing changes: the samplers run exactly as before.

## How it works

- **Hook.** The model is not edited. A thin wrapper around the generator's denoiser call scores the particles on the clean prediction (x0 in RFD3, x1 in RFD4). When the population is resampled, the sampler's own state is reindexed in place. Code: `ptm_rfd3/rfd3_particles.py`, `ptm_rfd3/rfd4_particles.py`.
- **Engine.** The same particle engine serves both generators (`ptm_rfd3/particles.py`). Schemes:
  - `smc`: Feynman–Kac SMC with a tempered difference potential. The target is q(x)·exp(λ_max·R(x)), where q is the generator's own distribution. Resampling happens when the ESS falls below `ess_frac`·K (systematic resampling).
  - `we`: weighted ensemble (Huber & Kim). Particles are binned (on the reward or by fold) and split/merged within bins. The weights stay exact for the *untilted* generator, so it spreads particles over the bins rather than tilting toward high reward.
  - `twe`: the `smc` weights combined with the `we` bin allocation.
  - `none`: score and log only. Used for pilots and for measuring when the reward becomes predictive.
  - `fork`: diagnostic. Every particle is duplicated once at `start`, to measure how late copies still diverge.
- **Resampling needs noise.** Copies of a particle only diverge if the sampler still injects noise. RFD3 stops injecting noise at step ~158 of 199 (`gamma_min`), and resampling is clamped to the steps before that.
- **Fold diversity.** At every scored step, particles are clustered by their predicted binder geometry. The distance is the RMSD of internal distance matrices (binder–binder and binder–PTM), so it is alignment-free and covers both fold and pose; the default threshold is 2.0 Å. On top of this clustering:
  - `fold_dedup` + `fold_mix` (smc): offspring are drawn from a proposal that gives every fold cluster a larger share. Each child carries the weight correction W/q, so the population still targets the same distribution.
  - `we_bin_by: fold` (we/twe): one bin per fold cluster, so no fold present at a resampling step can be lost.

  Without these, late resampling copies already-committed folds. In a smoke test, WE went from 8 to 2 fold clusters, with siblings at TM 0.92.

## The config file

`PTMSMC_CONFIG=file.yaml` (or `.json`). Every key is optional. Environment variables override the file:
- `PTMSMC_<FIELD>` for policy, geometry and physics fields;
- `PTMSMC_W_<TERM>` to set a term's weight (this adds the term if absent);
- `PTMSMC_TERMS="ptm_burial:1,clash:0.1"` to replace the term list.

| Section | Fields |
|---|---|
| `policy` | `scheme`, `start`, `stop`, `every`, `lam_max`, `lam_power`, `cess_frac`, `ess_frac`, `max_offspring`, `islands`, `fold_dedup`, `fold_mix`, `fold_thr`, `we_bins`, `we_every`, `we_bin_by`, `terminal_resample`, `null`, `seed` |
| `geometry` | `mode` (`solid` / `count`), `n_dir`, `ray_len`, `ray_r`, `r0`, `d0`, `n_sat`, `clash_d` |
| `physics` | `a0`, `r_cut`, `unsat_burial`, `hb_d`, `pol_cb_d`, `charge_offset` |
| `terms` | a list of `{name, weight, schedule, normalize}` entries |

Each term entry:
- **`schedule`** (time in diffusion; p = sampler progress, 0 → 1): `start`, `stop`, `shape` (`const` / `linear` / `cosine` / `power` / `step`), `power`, `floor`, `hold`. The term is off before `start` and ramps by `shape` inside the window. After `stop` it holds its last value (or switches off if `hold: false`).
- **`normalize`**: `mode` (`none` / `scale` / `zscore`), `scale`, `clip`.

The final structures are scored at p = 1, so **the terms active at p = 1 define the target**. Terms that are off at p = 1 only steer the path; the importance weights correct for them. Use `zscore` only for path terms.

### Reward terms

P are the atoms the modification adds, detected by `rfd3_inputs.detect_sites`: atoms not in the parent residue, plus the anchor atom. Q are the other target atoms; B are the binder atoms. Every term is "higher is better". All terms are PTM-type general: nothing is specific to phosphate.

| Term | Meaning |
|---|---|
| `ptm_burial` | Solid-angle fraction of directions from each P atom blocked by binder atoms (`mode: solid`), or a saturating soft contact count (`mode: count`). **The primary term.** |
| `ptm_contacts` | Soft count of binder atoms around P, min(n_P / n_sat, 1) |
| `peptide_overburial` | −max(0, burial(Q) − burial(P)): penalises binding the peptide more than the modification |
| `contact_share` | n_P / (n_P + n_Q): the fraction of the binder's target contacts that land on the modification |
| `clash` | −(target–binder atom pairs closer than `clash_d`) / \|P\| |
| `solvation` | −ΔG_solv of P, Eisenberg–McLachlan atomic solvation parameters: buried apolar atoms are favourable, buried charged atoms costly |
| `electrostatics` | −E between the P charges and the binder's expected side-chain charges. Uses a distance-dependent dielectric (ε = 4r). Binder charges come from the per-step sequence prediction. |
| `unsat_polar` | −(fraction of polar P atoms that are buried with no binder backbone N/O or polar side chain nearby) |
| `compactness` | −max(0, Rg / (2.2·N^0.38) − 1) of the binder Cα atoms |

How P charges are assigned (`ptm_rfd3/ptm_physics.py`):
- They come from each modification's own bond graph at pH 7, so no residue tables are involved.
- Checked against the CCD:

  | PTM | Net charge |
  |---|---|
  | PTR / SEP / TPO (phospho) | −2 |
  | TYS (sulfo-Tyr) | −1 |
  | ALY (acetyl-Lys) | 0 |
  | MLY / M3L (methyl-Lys) | +1 |
  | CIR (citrulline) | 0 |
  | HYP (hydroxy-Pro) | 0 |
  | NAG (GlcNAc) | 0 |

- Modifications defined by SMILES use the bonds of the built target.

How the terms are evaluated during sampling:
- **RFD3:** binder backbone N, Cα, C, O and Cβ.
- **RFD4:** Cα only, with a virtual Cβ. Its geometry defaults are set for Cα-level distances.

## Choosing settings

- **Scoring window (`start` / `stop`).**
  - Score only where the per-step reward predicts the final one. Measure this with a `scheme: none` run over the whole window: the per-step log gives the correlation with the final value.
  - On RFD3 / EGFR pTyr1138 the burial reward reached ρ 0.83 at step 140 and 0.93 at step 150 of 199. Churn noise ends at step 158, so the useful window is narrow.
  - A larger `gamma_min` ends the noise earlier; a smaller one extends it.
- **Tilt strength (`lam_max`).**
  - Scale it to the reward's spread across particles. With a spread of ~0.08, λ = 8 barely tilts, and SMC never resampled.
  - Adaptive tempering (`cess_frac`) caps how much one step can reweight.
- **Diversity.** Always check fold diversity against an unguided run at the same compute, e.g. by clustering the
  binders at TM 0.6. Each design's `ptmsmc.lineage` tells which designs are resampled siblings.

## Files

| File | Contents |
|---|---|
| `ptm_rfd3/particles.py` | engine (schemes, ESS / CESS, systematic resampling, WE, fold clusters, burial geometry) |
| `ptm_rfd3/reward_model.py` | multi-term reward, schedules, normalisation, config loading |
| `ptm_rfd3/ptm_physics.py` | PTM chemistry (charges, polarity, solvation parameters) and the physics terms |
| `ptm_rfd3/rfd3_particles.py` | RFD3 hook |
| `ptm_rfd3/rfd4_particles.py` | RFD4 hook |
| `ptm_rfd3/rfd4_run.py` | RFD4 design stage (spec → condition spec → designs in the RFD3 layout) |
| `examples/particle_guidance/*.yaml` | starting configs |
| `tests/test_particles.py`, `test_ptm_physics.py`, `test_reward_model.py` | CPU tests |
