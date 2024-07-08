#!/bin/bash

while getopts c:d: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        d) dir=${OPTARG};;
    esac
done

model=gpt-4-1106-preview

# baselining with random and similarity
python scripts/api_prompting_clsf.py \
    GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --train-split train --test-split small_dev \
    --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name $model --max-new-tokens 20 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    { --sampling-strategy _None_ --alternative {model_name}-random-retrieval-{shot}-shot --seed 0 1 2 --shot 5 15 25 } \
    { --sampling-strategy similarity --sentence-model all-mpnet-base-v2 --alternative {model_name}-similarity-retrieval-{shot}-shot --seed 0 --device cuda:$cuda --shot 25 }
# 3 x 3 + 1 = 10 -> $35

# get priors for train and dev for different labels

python scripts/api_prompting_clsf.py \
    GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --test-ids-filename $dir/goemotions/train_useful_ids.txt $dir/goemotions/small_dev_ids.txt  \
    --train-split test --test-split train small_dev \
    --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name $model --max-new-tokens 20 \
    { --seed 0 1 2 } { --seed 0 --label-randomization-seed 1 2 --description same-examples-different-labels } --shot 5 15 25 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --label-mode distribution --alternative {model_name}-distribution-traindev-prior-{shot}-shot
# (3 + 2) x 3 = 15 -> $50

# reinforce priors

## from different examples

for shot in 5 15 25
do
    python scripts/api_prompting_clsf.py \
        GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
        --train-split train --test-split small_dev \
        --train-pred-log-dir logs/GoEmotionsOpenAI/$model-distribution-traindev-prior-$shot-shot_0 \
        --test-pred-log-dir args.train_pred_log_dir \
        --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
        --model-name $model --max-new-tokens 20 \
        --seed 0 --shot $shot { --train-pred-log-index 0 1 2 } { --train-pred-log-index 3 4 --description same-examples-different-labels } --test-pred-log-index args.train_pred_log_index  \
        --logging-level debug --annotation-mode aggregate --text-preprocessor false \
        --label-mode preds --alternative {model_name}-$shot-shot-distribution-prompt-{shot}-shot
done
# 3 x 5 = 15-> $50
