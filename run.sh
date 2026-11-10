#!/usr/bin/env bash
# One-command entry point for the DeepFuse paper.
#
#   ./run.sh tables   Recompute every table from the shipped result files (no GPU, no images).
#   ./run.sh check    Quick end-to-end check on DDI (K=4, 2 seeds, a few minutes on one GPU).
#   ./run.sh all      Re-run every experiment, then the statistics (one GPU, long).
#   ./run.sh splits   Rebuild splits/ from splits/index/ and the ISIC Archive metadata.
#
# Paths are read from config.env (copy config.env.example) or from the environment.
# Jobs always run one after another, never in parallel.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$REPO/config.env" ] && source "$REPO/config.env"
PYTHON="${PYTHON:-python}"
MODE="${1:-}"

ALL="ham pad dermnet bcn hiba mskcc ddi patch16 derm7pt_derm derm7pt_clin"
ARMS="base last1 fixed8 random8 deepfuse"

die() { echo "error: $*" >&2; exit 1; }
log() { echo; echo "=== $(date '+%H:%M:%S') $*"; }
run() { log "$*"; "$PYTHON" "$@"; }

need_gpu_setup() {
  [ -n "${PANDERM_CODE:-}" ] && [ -d "$PANDERM_CODE/classification/models" ] \
    || die "set PANDERM_CODE to a clone of https://github.com/SiyuanYan1/PanDerm"
  [ -n "${PANDERM_CKPT:-}" ] && [ -f "$PANDERM_CKPT" ] \
    || die "set PANDERM_CKPT to panderm_ll_data6_checkpoint-499.pth"
  [ -n "${DEEPFUSE_DATA:-}" ] && [ -d "$DEEPFUSE_DATA" ] \
    || die "set DEEPFUSE_DATA to the folder holding the dataset images (see README, Dataset layout)"
  [ -n "${LIBFEWSHOT:-}" ] && [ -d "$LIBFEWSHOT" ] \
    || die "set LIBFEWSHOT to a clone of https://github.com/RL-VIG/LibFewShot"
  "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    || die "no CUDA GPU visible to $PYTHON"
  export PANDERM_CKPT DEEPFUSE_DATA LIBFEWSHOT
  export PYTHONPATH="$PANDERM_CODE/classification${PYTHONPATH:+:$PYTHONPATH}"
}

tables() {
  cd "$REPO/code"
  run icd_pauc/revision_tables.py --tag trans ind
  run icd_pauc/revision_depth_report.py --tag panderm_trans panderm_ind dinov2_trans dinov2_ind clip_trans clip_ind
  run icd_pauc/revision_stats_backbones.py
  run icd_pauc/revision_stats_clusters.py
  run icd_pauc/class_count_depth.py
}

caches() {  # $@ = datasets
  cd "$REPO/code"
  for ds in "$@"; do
    run -m icd_pauc.model_zoo  --dataset "$ds" --models panderm_lp dinov2_l clip_l
    run -m icd_pauc.deep_cache --dataset "$ds"
  done
  run -m icd_pauc.deep_cache_timm --backbone dinov2_l --datasets "$@"
  run -m icd_pauc.deep_cache_timm --backbone clip_l   --datasets "$@"
}

dermnet_mask() {
  "$PYTHON" -c "import pandas as pd, numpy as np; m = pd.read_csv('$REPO/splits/dermnet_test_contamination.csv'); np.save('$REPO/code/dermnet_keep.npy', ~m['masked'].to_numpy())"
}

check() {
  need_gpu_setup
  caches ddi
  cd "$REPO/code"
  local S="--datasets ddi --Ks 4 --n_seeds 2"
  run -m icd_pauc.revision_main   $S --norm transductive --tag check_trans
  run -m icd_pauc.revision_main   $S --norm inductive    --tag check_ind
  run -m icd_pauc.fsl_heads_run   $S --norm transductive --save_preds --tag _rev_check
  run -m icd_pauc.depth_select    $S --backbone panderm --norm transductive --tag check_panderm
  run -m icd_pauc.revision_main   $S --backbone dinov2_l --norm transductive --arms $ARMS --tag check_abl
  run -m icd_pauc.picks_symmetric $S --tag check
  run -m icd_pauc.picks_ablation  $S --tag check
  run -m icd_pauc.protocol_compare $S --tag _check
  "$PYTHON" "$REPO/scripts/compare_check.py" "$REPO/results"
}

all() {
  need_gpu_setup
  if [ -n "$(ls -A "$REPO/results" 2>/dev/null)" ]; then
    [ -e "$REPO/results_shipped" ] && die "results/ is not empty and results_shipped/ exists; move one of them first"
    log "moving the shipped result files to results_shipped/"
    mv "$REPO/results" "$REPO/results_shipped"; mkdir -p "$REPO/results"
  fi
  caches $ALL
  cd "$REPO/code"
  # Table 2: DeepFuse versus the default protocol
  run -m icd_pauc.revision_main --all --norm transductive --tag trans
  run -m icd_pauc.revision_main --all --norm inductive    --tag ind
  # Table 4: twelve few-shot comparators
  run -m icd_pauc.fsl_heads_run --datasets $ALL --Ks 4 8 16 --norm transductive --save_preds --tag _rev_trans
  run -m icd_pauc.fsl_heads_run --datasets $ALL --Ks 4 8 16 --norm inductive    --save_preds --tag _rev_ind
  # Table 3(a), Fig. 2: extraction depth; Table 3(b): ablation arms
  for bb in panderm dinov2_l clip_l; do
    t=${bb%_l}
    run -m icd_pauc.depth_select  --backbone $bb --all --n_seeds 20 --norm transductive --tag ${t}_trans
    run -m icd_pauc.depth_select  --backbone $bb --all --n_seeds 20 --norm inductive    --tag ${t}_ind
    run -m icd_pauc.revision_main --backbone $bb --all --Ks 1 4 16 --norm transductive --arms $ARMS --tag ${t}_abl_trans
    run -m icd_pauc.revision_main --backbone $bb --all --Ks 1 4 16 --norm inductive    --arms $ARMS --tag ${t}_abl_ind
  done
  # Table 3(b) symmetric row, DermNet de-duplication, budget comparison (rank with seeds 0-4)
  run -m icd_pauc.picks_symmetric --all --tag sym10 --n_seeds 20
  run -m icd_pauc.picks_ablation  --all --tag all10_5arm
  run -m icd_pauc.picks_ablation  --all --inductive --tag all10_ind
  dermnet_mask
  run -m icd_pauc.picks_ablation  --datasets dermnet --test_mask "$REPO/code/dermnet_keep.npy" --tag dermnet_dedup
  run -m icd_pauc.protocol_compare
  run -m icd_pauc.fsl_heads_run   --datasets $ALL --Ks 4 8 16 --tag _libfewshot_audited
  # Statistics
  run -m icd_pauc.revision_clinical --tag trans
  tables
  log "done. New results are in results/, the shipped ones in results_shipped/."
}

splits() {
  [ -n "${ISIC_META:-}" ] && [ -f "$ISIC_META" ] || die "set ISIC_META to the ISIC Archive metadata.csv"
  run "$REPO/scripts/build_splits.py" --isic_meta "$ISIC_META" --out "$REPO/splits"
}

case "$MODE" in
  tables) tables ;;
  check)  check ;;
  all)    all ;;
  splits) splits ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
