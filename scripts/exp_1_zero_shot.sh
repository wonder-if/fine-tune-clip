#!/bin/bash
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-nightly
# 设置默认的 python 执行路径（可选）以及实验主程序路径
PYTHON_BIN=/home/wyh/anaconda3/envs/pytorch-nightly/bin/python
EXPERIMENT=/mnt/local-data/workspace/codes/fine-tune-clip/exps/exp_1_zero_shot_baseline.py

DATASET="dataset.zero_shot.dataset_name"
DOMAIN="dataset.zero_shot.domain_name"

OUTPUT_FILE=outputs/results_zero_shot_qdr.csv
mkdir -p outputs
echo "dataset,domain,accuracy" > $OUTPUT_FILE
DOMAINS=(quickdraw)

for DOMAIN_NAME in "${DOMAINS[@]}"
do
    echo "▶ Running: domain=$DOMAIN_NAME"

    OUTPUT=$($PYTHON_BIN $EXPERIMENT ${DATASET}=domainnet ${DOMAIN}=$DOMAIN_NAME)
    JSON_RESULT=$(echo "$OUTPUT" | sed -n '/ZERO_SHOT_METRICS_START/,/ZERO_SHOT_METRICS_END/p' | grep -v ZERO_SHOT_METRICS)
    if [ -n "$JSON_RESULT" ]; then
        ACCURACY=$(echo "$JSON_RESULT" | jq -r '.zero_shot_eval_overall_accuracy.accuracy')
        echo "domainnet,$DOMAIN_NAME,$ACCURACY" >> $OUTPUT_FILE
    else
        echo "❌ Failed to parse output for $DOMAIN_NAME"
    fi
done
