#!/bin/bash
#SBATCH --job-name=ptm_rfd4
#SBATCH --partition=barracuda
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx4090|rtx5090|rtxa5000
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# RFD4-Proteina binder design against a PTM-bearing target, in the AIPD cu128 container (A5000 / 4090 / 5090), with optional
# particle guidance (ptm_rfd3.rfd4_particles; PTMSMC_* env vars are forwarded into the container).
# Steps (all inside the container): ptm_rfd3.rfd4_run inputs (same PTM builder as RFD3) -> rfd4-generate through
# ptm_rfd3.rfd4_particles -> ptm_rfd3.rfd4_run collect (RFD3 output layout, binder = chain A) -> $PTM_RFD3_WORK/out/<OUTSUB>/<TAG>.tar,
# which `ptm-rfd3 stage --keep-sequence` reads like an RFD3 tar.
# Placement: GPU - flow-matching sampling is >>2x faster than CPU.
# Required env: SPECS=<PTM spec json> OUTSUB=<name>. Optional: LENGTHS="80,95,110" (fixed binder lengths, default midpoint of the
#   contig range) NREP=8 (designs per length) BS=8 (batch size = particles per SMC population; must divide NREP) SEED=1 TAG=run
#   RFD4_CONTAINER (default AIPD rfd4-proteina-cu128.sif) EXTRA="<more rfd4-generate hydra overrides>"
set -uo pipefail
abspath() { if [ -e "$1" ]; then readlink -f "$1"; else printf '%s' "$1"; fi; }
W=$(abspath "${PTM_RFD3_WORK:-$HOME/ptm_rfd3_work}")
REPO=$(abspath "${PTM_RFD3_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}")
: "${SPECS:?}"; : "${OUTSUB:?}"
SPECS=$(abspath "$SPECS"); [ -f "$SPECS" ] || { echo "ERROR: SPECS=$SPECS not found"; exit 2; }
C=${RFD4_CONTAINER:-/zfs/s01/z02/home/project/AIPD/containers/rfd4-proteina-cu128.sif}
WEIGHTS=${RFD4_WEIGHTS:-/zfs/s01/z02/home/project/AIPD/weights/rfd4-proteina}
S=/apps/firefly/spack/opt/spack/linux-broadwell
AP=${APPTAINER_BIN:-$S/apptainer-1.5.2-747bebqovgvi4xkbnu22yhfnfl42ngd5/bin/apptainer}
export PATH=$(ls -d $S/{conmon,fuse-overlayfs,squashfuse,squashfs,gocryptfs,e2fsprogs,shadow,util-linux-uuid}-*/bin 2>/dev/null | tr '\n' ':')$PATH
export APPTAINER_CACHEDIR=/tmp/apptcache_$USER
NREP=${NREP:-8}; BS=${BS:-8}; SEED=${SEED:-1}; TAG=${TAG:-run}
(( NREP % BS == 0 )) || { echo "ERROR: BS=$BS must divide NREP=$NREP (one example per batch)"; exit 2; }
TAR=$W/out/$OUTSUB/$TAG.tar
mkdir -p $W/out/$OUTSUB
[ -f $TAR ] && { echo "already done: $TAR"; exit 0; }
T=/tmp/ptm_rfd4_${SLURM_JOB_ID:-local}_$TAG; mkdir -p $T/work $T/rfd4_out $T/out
echo "### $TAG host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T)"
# forward PTMSMC_* (particle guidance) through --cleanenv
ENVS="PYTHONPATH=$REPO,HF_HUB_OFFLINE=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,PTMSMC_LOG_DIR=$T/rfd4_out/ptmsmc"
for v in $(env | grep -o '^PTMSMC_[A-Z_]*' | grep -v '^PTMSMC_LOG_DIR$'); do ENVS="$ENVS,$v=${!v}"; done
$AP exec --nv --cleanenv --pwd $T --bind "$REPO,$W,$T,$(dirname $SPECS),$WEIGHTS:/weights/rfd4-proteina:ro" --env "$ENVS" "$C" bash -c "
  set -euo pipefail
  python -m ptm_rfd3.rfd4_run inputs '$SPECS' $T/work --lengths '${LENGTHS:-}'
  source $T/work/rfd4_env.sh; echo \"### PTMSMC_PTM_ATOMS=\$PTMSMC_PTM_ATOMS PTMSMC_SCHEME=\${PTMSMC_SCHEME:-<off>}\"
  python -m ptm_rfd3.rfd4_particles hf_ckpt_path=/weights/rfd4-proteina/bundle inputs=$T/work/rfd4_specs.yaml out_dir=$T/rfd4_out \
      num_replicates=$NREP batch_size=$BS seed=$SEED ${EXTRA:-}
  python -m ptm_rfd3.rfd4_run collect $T/rfd4_out $T/work $T/out --pop-size $BS
  mkdir -p $T/out/ptmsmc; cp -r $T/rfd4_out/ptmsmc/. $T/out/ptmsmc/ 2>/dev/null || true
" 2>&1 | grep -vE "MIRROR|VAR_NAME|env file|^  \(" | tee $T/design.log | tail -60
RC=${PIPESTATUS[0]}
N=$(ls $T/out | grep -c 'cif.gz$')
echo "### design rc=$RC n_cif=$N $(date +%T)"
if [ "$RC" -eq 0 ] && [ "$N" -gt 0 ] && tar cf $TAR.part -C $T/out . && mv $TAR.part $TAR; then
  echo "### wrote $TAR ($(du -h $TAR | cut -f1))"
else
  echo "### INCOMPLETE rc=$RC n=$N: keeping the raw run in $W/out/$OUTSUB/failed_$TAG"
  mkdir -p $W/out/$OUTSUB/failed_$TAG && cp -r $T/. $W/out/$OUTSUB/failed_$TAG/ 2>/dev/null
fi
rm -rf $T
echo "### done $(date +%T)"
