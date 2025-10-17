#!/bin/bash
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-nightly
# 设置默认的 python 执行路径（可选）以及实验主程序路径
PYTHON_BIN=/home/wyh/anaconda3/envs/pytorch-nightly/bin/python
EXPERIMENT=./exps/exp_1_zero_shot_baseline/main.py

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

# 计算并写入平均值
if [ -s "$OUTPUT_FILE" ]; then
    AVG=$(awk -F, 'NR>1 && $3 != "" {sum+=$3; n++} END{if(n) printf("%.6f", sum/n)}' "$OUTPUT_FILE")
    echo "domainnet,average,$AVG" >> "$OUTPUT_FILE"
fi