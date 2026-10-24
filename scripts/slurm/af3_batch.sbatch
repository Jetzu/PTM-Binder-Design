#!/bin/bash -l
#SBATCH --job-name=ptm_af3
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --time=01:55:00
#SBATCH --requeue
# AlphaFast/AF3 co-folding of binder + PTM-bearing target (+ unmodified control) from a folder of input JSONs.
# Placement: 1 x H100 (preempt) - AF3 inference ~21 s/prediction on GPU, hopeless on CPU. Node-local /tmp scratch.
# Required: IN_DIR (JSONs on hnvme), TAR (output tar on hnvme). Optional: SAMPLES=5 RECYCLES=10 BUDGET_MIN=95
set -uo pipefail
# Site configuration (environment): AF3_SIF = AlphaFast/AF3 apptainer image, AF3_WEIGHTS = directory with the AF3 model
# parameters (obtained from Google DeepMind under their terms; not distributed here), PTM_RFD3_WORK = work directory,
# PTM_RFD3_REPO = this repository (default: the directory you submit from).
: "${AF3_SIF:?set AF3_SIF (AlphaFast/AF3 .sif)}"; : "${AF3_WEIGHTS:?set AF3_WEIGHTS (AF3 parameter directory)}"
W=${PTM_RFD3_WORK:-$HOME/ptm_rfd3_work}
SIF=$AF3_SIF
WEIGHTS=$AF3_WEIGHTS
REPO=${PTM_RFD3_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
mkdir -p "$W"
# inputs: IN_DIR (a folder of JSONs) or IN_TAR (ONE tar of JSONs - hnvme has an inode quota, so pack them)
: "${TAR:?}"
if [ -n "${IN_TAR:-}" ]; then
  IN_DIR=/tmp/ptm_af3_in_${SLURM_JOB_ID:-local}; rm -rf $IN_DIR; mkdir -p $IN_DIR
  tar xf "$IN_TAR" -C $IN_DIR || { echo "ERROR: cannot extract IN_TAR=$IN_TAR"; exit 2; }
  # tar may hold a top-level folder: flatten
  find $IN_DIR -mindepth 2 -name '*.json' -exec mv {} $IN_DIR/ \; 2>/dev/null || true
fi
: "${IN_DIR:?}"
[ "$(find "$IN_DIR" -maxdepth 1 -name '*.json' ! -name 'index*.json' ! -name '*_index.json' ! -name '*_names.json' | wc -l)" -gt 0 ] || { echo "ERROR: no AF3 input JSONs in $IN_DIR"; exit 2; }
echo "### af3 batch host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date +%T) in=$IN_DIR"
apptainer exec --nv --bind "$WEIGHTS":/data/models --bind "$W":"$W" --bind "$REPO":"$REPO" "$SIF" \
    python $REPO/scripts/af3/af3_batch.py --in_dir "$IN_DIR" --tar "$TAR" --samples ${SAMPLES:-5} --recycles ${RECYCLES:-10} \
    --budget_min ${BUDGET_MIN:-95}
rc=$?
echo "### rc=$rc end $(date +%T)"
exit $rc
