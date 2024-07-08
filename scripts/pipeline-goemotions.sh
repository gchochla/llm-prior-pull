while getopts c:m:d: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        m) model=${OPTARG};;
        d) dir=${OPTARG};;
    esac
done

seeds={0..2}
shots={5..25..10}

# baselining with random and similarity

python scripts/llm_prompting_clsf.py \
    GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --train-split train --test-split small_dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name-or-path $model --label-format json --max-new-tokens 18 --device cuda:$cuda \
    --shot $shots \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
    { --sampling-strategy _None_ --alternative {model_name_or_path}-random-retrieval-{shot}-shot --seed 0 1 2 } \
    { --sampling-strategy similarity --sentence-model all-mpnet-base-v2 --alternative {model_name_or_path}-similarity-retrieval-{shot}-shot --seed 0 }
# 3 x 5 x 2 = 30

# baselining with random and distribution labels, and zero-shot

python scripts/llm_prompting_clsf.py \
    GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --train-split train --test-split small_dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name-or-path $model --label-format json --max-new-tokens 18 --device cuda:$cuda \
    --seed $seeds --shot $shots \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
    --label-mode random distribution --alternative {model_name_or_path}-{label_mode}-prior-{shot}-shot
# 3 x 5 x 2 = 30

### OPTIONAL: can get performance from compare_distributions.py from traindev zero-shot
# python scripts/llm_prompting_clsf.py \
#     GoEmotions --root-dir $dir/goemotions \
#     --test-split small_dev \
#     --system ' ' --instruction $INSTRUCTION --incontext $INCONTEXT \
#     --model-name-or-path $model --max-new-tokens 18 --device cuda:$cuda \
#     --seed 0 --shot 0 \
#     --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
#     --alternative {model_name_or_path}-zero-shot
# 1

# get priors for train and dev ## for different examples and ## for different labels, and zero-shot

python scripts/llm_prompting_clsf.py \
    GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --test-ids-filename $dir/goemotions/train_useful_ids.txt $dir/goemotions/small_dev_ids.txt  \
    --train-split test --test-split train small_dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name-or-path $model --label-format json --max-new-tokens 18 --device cuda:$cuda \
    { --seed $seeds } { --seed 0 --label-randomization-seed 1 2 } --shot $shots \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
    --label-mode distribution --alternative {model_name_or_path}-distribution-traindev-prior-{shot}-shot
# 3 x 5 = 15

python scripts/llm_prompting_clsf.py \
    GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --test-ids-filename $dir/goemotions/train_useful_ids.txt $dir/goemotions/small_dev_ids.txt \
    --test-split train small_dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name-or-path $model --max-new-tokens 18 --device cuda:$cuda \
    --seed 0 --shot 0 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
    --alternative {model_name_or_path}-zero-shot-traindev-prior
# 1

# reinforce priors

model_name=${model//\//--}

## from different examples

for shot in $shots
do
    python scripts/llm_prompting_clsf.py \
        GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
        --train-pred-log-dir logs/GoEmotions/$model_name-distribution-traindev-prior-$shot-shot_0 \
        --test-pred-log-dir args.train_pred_log_dir \
        --train-split train --test-split small_dev \
        --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' --incontext $'Input: {text}\n{label}\n' \
        --model-name-or-path $model --label-format json --max-new-tokens 18 --device cuda:$cuda \
        --seed 0 --shot $shots --test-pred-log-index args.train_pred_log_index \
        { --train-pred-log-index $seeds } { --train-pred-log-index 0 3 4  --description same-examples-different-labels } \
        --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
        --label-mode preds --alternative {model_name_or_path}-$shot-shot-distribution-prompt-{shot}-shot
done
# 5 x 3 x 5 = 75

## from different labels INTEGRATED ABOVE

# for shot in $shots
# do
#     python scripts/llm_prompting_clsf.py \
#         GoEmotions --root-dir $dir/goemotions \
#         --train-pred-log-dir logs/GoEmotions/$model_name-distribution-traindev-prior-$shot-shot_0 \
#         --test-pred-log-dir args.train_pred_log_dir \
#         --train-split train --test-split small_dev --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
#         --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' --incontext $'Input: {text}\n{label}\n' \
#         --model-name-or-path $model --label-format json --max-new-tokens 18 --device cuda:$cuda \
#         --shot $shots --test-pred-log-index args.train_pred_log_index \
#         --seed 0 --train-pred-log-index 0 3 4  \
#         --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
#         --label-mode preds --alternative {model_name_or_path}-$shot-shot-distribution-prompt-{shot}-shot --description same-examples-different-labels
# done
# 5 x 3 x 5 = 75

## from zero-shot

python scripts/llm_prompting_clsf.py \
    GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --train-pred-log-dir logs/GoEmotions/$model_name-zero-shot-traindev-prior_0 \
    --test-pred-log-dir args.train_pred_log_dir \
    --train-split train --test-split small_dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name-or-path $model --label-format json --max-new-tokens 18 --device cuda:$cuda \
    --seed $seeds --shot $shots \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
    --label-mode preds --alternative {model_name_or_path}-zero-shot-prompt-{shot}-shot
# 3 x 5 = 15