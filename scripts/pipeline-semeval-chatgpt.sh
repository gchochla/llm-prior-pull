while getopts c:m:d: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        m) model=${OPTARG};;
        d) dir=${OPTARG};;
    esac
done

seeds={0..2}
shots={5..85..20}


# baselining with random and similarity

python scripts/api_prompting_clsf.py \
    SemEval --root-dir $dir/semeval2018task1 --language English \
    --train-split train --test-split dev \
    --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name $model --max-new-tokens 20 \
    --shot $shots \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    { --sampling-strategy _None_ --alternative {model_name}-random-retrieval-{shot}-shot --seed $seeds } \
    { --sampling-strategy similarity --sentence-model all-mpnet-base-v2 --alternative {model_name}-similarity-retrieval-{shot}-shot --seed 0 --device cuda:$cuda }
# 5 x 3 + 1 = 16


# baselining with random and distribution labels, and zero-shot

python scripts/api_prompting_clsf.py \
    SemEval --root-dir $dir/semeval2018task1 --language English \
    --train-split train --test-split dev \
    --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name $model --max-new-tokens 20 \
    --shot $shots --seed $seeds \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --label-mode random distribution --alternative {model_name}-{label_mode}-prior-{shot}-shot
# 2 x 5 x 3 = 30

### OPTIONAL: can get performance from compare_distributions.py from traindev zero-shot
# python scripts/api_prompting_clsf.py \
#     SemEval --root-dir $dir/semeval2018task1 --language English \
#     --test-split dev \
#     --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
#     --model-name $model --max-new-tokens 20 \
#     --shot 0 --seed 0 \
#     --logging-level debug --annotation-mode aggregate --text-preprocessor false \
#     --alternative {model_name}-zero-shot
# 1

# get priors for train and dev for different labels, and zero-shot

python scripts/api_prompting_clsf.py \
    SemEval --root-dir $dir/semeval2018task1 --language English \
    --test-ids-filename $dir/semeval2018task1/train_useful_ids.txt $dir/semeval2018task1/dev_ids.txt \
    --train-split test --test-split train dev \
    --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name $model --max-new-tokens 20 \
    { --seed $seeds } { --seed 0 --label-randomization-seed 1 2 } --shot $shots \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --label-mode distribution --alternative {model_name}-distribution-traindev-prior-{shot}-shot
# (3 + 2) x 5 = 25

python scripts/api_prompting_clsf.py \
    SemEval --root-dir $dir/semeval2018task1 --language English \
    --test-ids-filename $dir/semeval2018task1/train_useful_ids.txt $dir/semeval2018task1/dev_ids.txt \
    --test-split train dev \
    --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name $model --max-new-tokens 20 \
    --shot 0 --seed 0 \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --alternative {model_name}-zero-shot-traindev
# 1

# reinforce priors

## from different examples

for shot in $shots
do
    python scripts/api_prompting_clsf.py \
        SemEval --root-dir $dir/semeval2018task1 --language English \
        --train-split train --test-split dev \
        --train-pred-log-dir logs/SemEvalOpenAI/$model-distribution-traindev-prior-$shot-shot_0 \
        --test-pred-log-dir args.train_pred_log_dir \
        --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
        --model-name $model --max-new-tokens 20 \
        --seed 0 --shot $shots --train-pred-log-index $seeds --test-pred-log-index args.train_pred_log_index  \
        --logging-level debug --annotation-mode aggregate --text-preprocessor false \
        --label-mode preds --alternative {model_name}-$shot-shot-distribution-prompt-{shot}-shot
done
# 5 x 3 x 5 = 75

## from different labels

for shot in $shots
do
    python scripts/api_prompting_clsf.py \
        SemEval --root-dir $dir/semeval2018task1 --language English \
        --train-split train --test-split dev \
        --train-pred-log-dir logs/SemEvalOpenAI/$model-distribution-traindev-prior-$shot-shot_0 \
        --test-pred-log-dir args.train_pred_log_dir \
        --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
        --model-name $model --max-new-tokens 20 \
        --shot $shots --seed 0 --train-pred-log-index 0 3 4 --test-pred-log-index args.train_pred_log_index \
        --logging-level debug --annotation-mode aggregate --text-preprocessor false \
        --label-mode preds --alternative {model_name}-$shot-shot-distribution-prompt-{shot}-shot --description same-examples-different-labels
done
# 5 x 3 x 5 = 75

## from zero-shot
python scripts/api_prompting_clsf.py \
    SemEval --root-dir $dir/semeval2018task1 --language English \
    --train-split train --test-split dev \
    --train-pred-log-dir logs/SemEvalOpenAI/$model-zero-shot-traindev_0 \
    --test-pred-log-dir args.train_pred_log_dir \
    --instruction $'Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {labels}\n\n' --incontext $'Input: {text}\n{label}\n' \
    --model-name $model --max-new-tokens 20 \
    --shot $shots --seed $seeds \
    --logging-level debug --annotation-mode aggregate --text-preprocessor false \
    --label-mode preds --alternative {model_name}-zero-shot-prompt-{shot}-shot
# 3 x 5 = 15