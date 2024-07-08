while getopts c:d: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        d) dir=${OPTARG};;
    esac
done

model=meta-llama/Llama-2-70b-chat-hf

seeds={0..2}
shots={5..25..10}

# baselining with random and similarity

python scripts/llm_prompting_clsf.py \
    GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --train-split train --test-split small_dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name-or-path $model --label-format json --max-new-tokens 13 --device cuda:$cuda \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
    { --sampling-strategy _None_ --alternative {model_name_or_path}-random-retrieval-{shot}-shot --seed 0 1 2 --shot $shots } \
    { --sampling-strategy similarity --sentence-model all-mpnet-base-v2 --alternative {model_name_or_path}-similarity-retrieval-{shot}-shot --seed 0 --shot 25 }
# 3 x 5 + 1 = 16

# get priors for train and dev ## for different examples and ## for different labels, and zero-shot

python scripts/llm_prompting_clsf.py \
    GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --test-ids-filename $dir/goemotions/train_useful_ids.txt $dir/goemotions/small_dev_ids.txt  \
    --train-split test --test-split train small_dev \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name-or-path $model --label-format json --max-new-tokens 13 --device cuda:$cuda \
    { --seed $seeds } { --seed 0 --label-randomization-seed 1 2 } --shot $shots \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
    --label-mode distribution --alternative {model_name_or_path}-distribution-traindev-prior-{shot}-shot
# 3 x 5 = 15

# reinforce priors

model_name=${model//\//--}

## from different examples and labels

for shot in $shots
do
    python scripts/llm_prompting_clsf.py \
        GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
        --train-pred-log-dir logs/GoEmotions/$model_name-distribution-traindev-prior-$shot-shot_0 \
        --test-pred-log-dir args.train_pred_log_dir \
        --train-split train --test-split small_dev \
        --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' --incontext $'Input: {text}\n{label}\n' \
        --model-name-or-path $model --label-format json --max-new-tokens 13 --device cuda:$cuda \
        --seed 0 --shot $shot --test-pred-log-index args.train_pred_log_index \
        { --train-pred-log-index $seeds } { --train-pred-log-index 0 3 4  --description same-examples-different-labels } \
        --logging-level debug --annotation-mode aggregate --text-preprocessor false --cache-dir /data/llm-shared --load-in-4bit --trust-remote-code \
        --label-mode preds --alternative {model_name_or_path}-$shot-shot-distribution-prompt-{shot}-shot
done
# 3 x 5 = 15
