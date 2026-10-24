#!/bin/bash -l
#SBATCH --job-name=ptm_rfd3
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --requeue
# Stock RFD3 (rfd3_latest.ckpt, pristine foundry tree) binder design against a PTM-bearing target, with
# the covalent-bond restoration patch (ptm_rfd3.rfd3_patch). One GPU, outputs tarred (small file count on shared filesystems).
# Placement: GPU - diffusion sampling is >>2x faster than CPU. Submit from the repository root (or set PTM_RFD3_REPO);
# the partition / time above are examples from our cluster: override with `sbatch -p <partition> -t <time> ...`.
# Site configuration (environment, nothing is hard-coded):
#   PTM_RFD3_WORK   work directory (default ~/ptm_rfd3_work): outputs go to $PTM_RFD3_WORK/out/<OUTSUB>/<TAG>.tar
#   RFD3_CKPT       RFD3 checkpoint (stock rfd3_latest.ckpt, `foundry install rfd3`), or CKPT=<fine-tuned checkpoint>
#   RFD3_PYTHON     python with rc-foundry[rfd3] + this package installed (default: python3 on PATH)
#   RFD3_ENV_TAR    alternative: a tar of a relocatable conda env, extracted node-locally; RFD3_ENV_PYREL = python inside it
#   FOUNDRY         optional foundry checkout (a fine-tuned checkpoint with new modules needs its tree: FOUNDRY=<tree>/foundry)
# Required env: SPECS=<specs.json> OUTSUB=<name>; optional KEYS="k1,k2" NB=1 DBS=8 SEED=1 TAG=run STEP_SCALE=1.5 GAMMA0=0.6 (default; N010: better than the docs' 3/0.2 for pTyr binders) EXTRA="<more hydra overrides, e.g. inference_sampler.use_classifier_free_guidance=True inference_sampler.cfg_scale=2>"
set -uo pipefail
abspath() { if [ -e "$1" ]; then readlink -f "$1"; else printf '%s' "$1"; fi; }      # paths given relative to the submission directory stay valid
W=$(abspath "${PTM_RFD3_WORK:-$HOME/ptm_rfd3_work}")
REPO=$(abspath "${PTM_RFD3_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}")
F=$(abspath "${FOUNDRY:-}")
: "${SPECS:?}"; : "${OUTSUB:?}"
SPECS=$(abspath "$SPECS"); [ -f "$SPECS" ] || { echo "ERROR: SPECS=$SPECS not found"; exit 2; }
[ -z "$F" ] || [ -d "$F/models/rfd3/src" ] || { echo "ERROR: FOUNDRY=$F is not a Foundry checkout (no models/rfd3/src)"; exit 2; }
[ -z "${ATOMWORKS:-}" ] || ATOMWORKS=$(abspath "$ATOMWORKS")
[ -z "${RFD3_ENV_TAR:-}" ] || RFD3_ENV_TAR=$(abspath "$RFD3_ENV_TAR")
CKPT_IN=${CKPT:-${RFD3_CKPT:-}}; [ -n "$CKPT_IN" ] || { echo "ERROR: set RFD3_CKPT (stock checkpoint) or CKPT"; exit 2; }
CKPT_PATH=$(abspath "$CKPT_IN")
NB=${NB:-1}; DBS=${DBS:-8}; SEED=${SEED:-1}; TAG=${TAG:-run}
KEYARG=""; [ -n "${KEYS:-}" ] && KEYARG="json_keys_subset=[${KEYS}]"
TAR=$W/out/$OUTSUB/$TAG.tar
mkdir -p $W/out/$OUTSUB
[ -f $TAR ] && { echo "already done: $TAR"; exit 0; }
T=/tmp/ptm_rfd3_${SLURM_JOB_ID:-local}_$TAG; mkdir -p $T/out
echo "### $TAG host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) extract $(date +%T)"
if [ -n "${RFD3_ENV_TAR:-}" ]; then
  tar xf "$RFD3_ENV_TAR" -C $T || { echo EXTRACT FAILED; exit 1; }
  PY=$T/${RFD3_ENV_PYREL:-rfd3_ssadj_test/bin/python}
else
  PY=${RFD3_PYTHON:-python3}
fi
export PYTHONPATH="$REPO${F:+:$F/src:$F/models/rfd3/src}${ATOMWORKS:+:$ATOMWORKS}${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# no `cd`: relative `input:` paths of the spec file keep resolving from the submission directory
$PY -c "import rfd3.engine as e; print('### rfd3 engine from', e.__file__, '| ptm diagnostic present:', hasattr(e.RFD3InferenceEngine, '_log_ptm_conditioning'))" 2>&1 | tail -1   # which foundry tree actually ran
$PY -m ptm_rfd3.rfd3_patch ckpt_path=$CKPT_PATH out_dir=$T/out inputs=$SPECS $KEYARG \
   n_batches=$NB diffusion_batch_size=$DBS skip_existing=False global_prefix=${TAG}_ seed=$SEED \
   inference_sampler.step_scale=${STEP_SCALE:-1.5} inference_sampler.gamma_0=${GAMMA0:-0.6} dump_trajectories=False ${EXTRA:-} 2>&1 | grep -vE "Warning|warn" | tee $T/design.log | tail -40
RC=${PIPESTATUS[0]}
grep -E "PTM conditioning|has a PTM adapter" $T/design.log | sed 's/^/### /' | sort | uniq -c | head -12   # PTM-adapter checkpoints: how many PTM atoms reached the model
N=$(ls $T/out | grep -c 'cif.gz$')
echo "### design rc=$RC n_cif=$N $(date +%T)"
OK=0
if [ "$RC" -eq 0 ] && [ "$N" -gt 0 ]; then
  if tar cf $TAR.part -C $T/out . && mv $TAR.part $TAR; then OK=1; echo "### wrote $TAR ($(du -h $TAR | cut -f1))"
  else echo "### PACKING FAILED: keeping the designs in $W/out/$OUTSUB/failed_$TAG"; rm -f $TAR.part; fi
else echo "### INCOMPLETE rc=$RC n=$N"; fi
if [ "$OK" -ne 1 ]; then mkdir -p $W/out/$OUTSUB/failed_$TAG && cp -r $T/out/. $W/out/$OUTSUB/failed_$TAG/ 2>/dev/null; fi
rm -rf $T
echo "### done $(date +%T)"
[ "$OK" -eq 1 ]                                                    # the job's exit status = the designs are in the tar (dependencies / requeue)
