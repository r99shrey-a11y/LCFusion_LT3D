#!/bin/bash
# Fine-tune DINO Run A (imbalanced) then Run B (class-balanced) sequentially.
# Each run uses the single GPU; B starts only after A finishes.
#
# Launch detached:
#   nohup bash ~/LCFusion_LT3D/train_dino_chain.sh > ~/LCFusion_LT3D/chain.log 2>&1 &

set -e
PY=/home/batashey/miniconda3/envs/lcfusion/bin/python
CFG=/home/batashey/LCFusion_LT3D/configs
LAUNCH=/home/batashey/LCFusion_LT3D/train_dino.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/batashey/mmdetection3d

echo "=========== RUN A (imbalanced) started $(date) ==========="
$PY $LAUNCH $CFG/dino_nuscenes_runA.py
echo "=========== RUN A finished $(date) ==========="

echo "=========== RUN B (class-balanced) started $(date) ==========="
$PY $LAUNCH $CFG/dino_nuscenes_runB.py
echo "=========== RUN B finished $(date) ==========="

echo "Both runs complete. Checkpoints in work_dirs/dino_nuscenes_run{A,B}/"
