#!/bin/bash
set -e

VIDEO="/data_avik/Ten_new_days/Day_3.mp4"
BASE="/data_avik/Ten_new_days"
TASKS_XLSX="/data_avik/Ten_new_days/Tasks.xlsx"   # <-- adjust path if needed

RUNS=(
    "run_unfreeze_fc"
    "run_unfreeze_layer4_fc"
    "run_unfreeze_layer3_layer4_fc"
    "run_unfreeze_layer2_layer3_layer4_fc"
    "run_unfreeze_layer1_layer2_layer3_layer4_fc"
)

for RUN in "${RUNS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running Part 11 for: $RUN"
    echo "=========================================="
    python3 "$BASE/Part 11(Optional for performance evaluation).py" \
        --model_path "$BASE/$RUN/resnet3d_best_kinetics_2.pth" \
        --mean_path  "$BASE/$RUN/dataset_mean.npy" \
        --std_path   "$BASE/$RUN/dataset_std.npy" \
        --video_path "$VIDEO" \
        --tasks_xlsx "$TASKS_XLSX" \
        --output_dir "$BASE/$RUN/optimization_runs"
done

echo ""
echo "All Part 11 runs complete!"
