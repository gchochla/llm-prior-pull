#!/bin/bash

while getopts d: flag
do
    case "${flag}" in
        d) dir=${OPTARG};;
    esac
done

model=gpt-3.5-turbo
python scripts/compare_distributions.py SemEval --root-dir $dir/semeval2018task1 \
    --out logs/analysis/semeval-$model \
    --experiments logs/SemEvalOpenAI/$model-random-retrieval-5-shot_0 \
    logs/SemEvalOpenAI/$model-random-retrieval-15-shot_0 \
    logs/SemEvalOpenAI/$model-random-retrieval-25-shot_0 \
    logs/SemEvalOpenAI/$model-random-retrieval-35-shot_0 \
    logs/SemEvalOpenAI/$model-random-retrieval-45-shot_0 \
    logs/SemEvalOpenAI/$model-similarity-retrieval-5-shot_0 \
    logs/SemEvalOpenAI/$model-similarity-retrieval-15-shot_0 \
    logs/SemEvalOpenAI/$model-similarity-retrieval-25-shot_0 \
    logs/SemEvalOpenAI/$model-similarity-retrieval-35-shot_0 \
    logs/SemEvalOpenAI/$model-similarity-retrieval-55-shot_0 \
    logs/SemEvalOpenAI/$model-similarity-retrieval-75-shot_0 \
    logs/SemEvalOpenAI/$model-similarity-retrieval-85-shot_0 \
    logs/SemEvalOpenAI/$model-zero-shot-traindev-prior_0 \
    logs/SemEvalOpenAI/$model-random-prior-5-shot_0 \
    logs/SemEvalOpenAI/$model-random-prior-15-shot_0 \
    logs/SemEvalOpenAI/$model-random-prior-25-shot_0 \
    logs/SemEvalOpenAI/$model-distribution-prior-5-shot_0 \
    logs/SemEvalOpenAI/$model-distribution-prior-15-shot_0 \
    logs/SemEvalOpenAI/$model-distribution-prior-25-shot_0 \
    logs/SemEvalOpenAI/$model-5-shot-distribution-prompt-5-shot_0 \
    logs/SemEvalOpenAI/$model-5-shot-distribution-prompt-15-shot_0 \
    logs/SemEvalOpenAI/$model-5-shot-distribution-prompt-25-shot_0 \
    logs/SemEvalOpenAI/$model-15-shot-distribution-prompt-5-shot_0 \
    logs/SemEvalOpenAI/$model-15-shot-distribution-prompt-15-shot_0 \
    logs/SemEvalOpenAI/$model-15-shot-distribution-prompt-25-shot_0 \
    logs/SemEvalOpenAI/$model-25-shot-distribution-prompt-5-shot_0 \
    logs/SemEvalOpenAI/$model-25-shot-distribution-prompt-15-shot_0 \
    logs/SemEvalOpenAI/$model-25-shot-distribution-prompt-25-shot_0 \
    logs/SemEvalOpenAI/$model-zero-shot-prompt-5-shot_0 \
    logs/SemEvalOpenAI/$model-zero-shot-prompt-15-shot_0 \
    logs/SemEvalOpenAI/$model-zero-shot-prompt-25-shot_0 \
    logs/SemEvalOpenAI/$model-distribution-traindev-prior-5-shot_0 \
    logs/SemEvalOpenAI/$model-distribution-traindev-prior-15-shot_0 \
    logs/SemEvalOpenAI/$model-distribution-traindev-prior-25-shot_0 \
    --alternative 5s 15s 25s 35s 45s \
    Cossim-5s Cossim-15s Cossim-25s Cossim-35s Cossim-55s Cossim-75s Cossim-85s \
    0s \
    Random-prior-5s Random-prior-15s Random-prior-25s \
    Proxy-prior-5s Proxy-prior-15s Proxy-prior-25s \
    Prior-5s-prompt-5s Prior-5s-prompt-15s Prior-5s-prompt-25s \
    Prior-15s-prompt-5s Prior-15s-prompt-15s Prior-15s-prompt-25s \
    Prior-25s-prompt-5s Prior-25s-prompt-15s Prior-25s-prompt-25s \
    0s-prompt-5s 0s-prompt-15s 0s-prompt-25s \
    Prior-5s-traindev Prior-15s-traindev Prior-25s-traindev

model=gpt-3.5-turbo
python scripts/compare_distributions.py GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --out logs/analysis/goemotions-$model \
    --experiments logs/GoEmotionsOpenAI/$model-random-retrieval-5-shot_0 \
    logs/GoEmotionsOpenAI/$model-random-retrieval-15-shot_0 \
    logs/GoEmotionsOpenAI/$model-random-retrieval-25-shot_0 \
    logs/GoEmotionsOpenAI/$model-similarity-retrieval-5-shot_0 \
    logs/GoEmotionsOpenAI/$model-similarity-retrieval-15-shot_0 \
    logs/GoEmotionsOpenAI/$model-similarity-retrieval-25-shot_0 \
    logs/GoEmotionsOpenAI/$model-zero-shot-traindev-prior_0 \
    logs/GoEmotionsOpenAI/$model-random-prior-5-shot_0 \
    logs/GoEmotionsOpenAI/$model-random-prior-15-shot_0 \
    logs/GoEmotionsOpenAI/$model-random-prior-25-shot_0 \
    logs/GoEmotionsOpenAI/$model-distribution-prior-5-shot_0 \
    logs/GoEmotionsOpenAI/$model-distribution-prior-15-shot_0 \
    logs/GoEmotionsOpenAI/$model-distribution-prior-25-shot_0 \
    logs/GoEmotionsOpenAI/$model-5-shot-distribution-prompt-5-shot_0 \
    logs/GoEmotionsOpenAI/$model-5-shot-distribution-prompt-15-shot_0 \
    logs/GoEmotionsOpenAI/$model-5-shot-distribution-prompt-25-shot_0 \
    logs/GoEmotionsOpenAI/$model-15-shot-distribution-prompt-5-shot_0 \
    logs/GoEmotionsOpenAI/$model-15-shot-distribution-prompt-15-shot_0 \
    logs/GoEmotionsOpenAI/$model-15-shot-distribution-prompt-25-shot_0 \
    logs/GoEmotionsOpenAI/$model-25-shot-distribution-prompt-5-shot_0 \
    logs/GoEmotionsOpenAI/$model-25-shot-distribution-prompt-15-shot_0 \
    logs/GoEmotionsOpenAI/$model-25-shot-distribution-prompt-25-shot_0 \
    logs/GoEmotionsOpenAI/$model-zero-shot-prompt-5-shot_0 \
    logs/GoEmotionsOpenAI/$model-zero-shot-prompt-15-shot_0 \
    logs/GoEmotionsOpenAI/$model-zero-shot-prompt-25-shot_0 \
    logs/GoEmotionsOpenAI/$model-distribution-traindev-prior-5-shot_0 \
    logs/GoEmotionsOpenAI/$model-distribution-traindev-prior-15-shot_0 \
    logs/GoEmotionsOpenAI/$model-distribution-traindev-prior-25-shot_0 \
    --alternative 5s 15s 25s \
    Cossim-5s Cossim-15s Cossim-25s \
    0s \
    Random-prior-5s Random-prior-15s Random-prior-25s \
    Proxy-prior-5s Proxy-prior-15s Proxy-prior-25s \
    Prior-5s-prompt-5s Prior-5s-prompt-15s Prior-5s-prompt-25s \
    Prior-15s-prompt-5s Prior-15s-prompt-15s Prior-15s-prompt-25s \
    Prior-25s-prompt-5s Prior-25s-prompt-15s Prior-25s-prompt-25s \
    0s-prompt-5s 0s-prompt-15s 0s-prompt-25s \
    Prior-5s-traindev Prior-15s-traindev Prior-25s-traindev

model=meta-llama--Llama-2-13b-chat-hf
python scripts/compare_distributions.py SemEval --root-dir $dir/semeval2018task1 \
        --out logs/analysis/semeval-$model \
        --experiments logs/SemEval/$model-random-retrieval-5-shot_0 \
        logs/SemEval/$model-random-retrieval-15-shot_0 \
        logs/SemEval/$model-random-retrieval-25-shot_0 \
        logs/SemEval/$model-similarity-retrieval-5-shot_0 \
        logs/SemEval/$model-similarity-retrieval-15-shot_0 \
        logs/SemEval/$model-similarity-retrieval-25-shot_0 \
        logs/SemEval/$model-similarity-retrieval-35-shot_0 \
        logs/SemEval/$model-similarity-retrieval-45-shot_0 \
        logs/SemEval/$model-similarity-retrieval-55-shot_0 \
        logs/SemEval/$model-similarity-retrieval-65-shot_0 \
        logs/SemEval/$model-zero-shot-traindev-prior_0 \
        logs/SemEval/$model-distribution-prior-5-shot_0 \
        logs/SemEval/$model-distribution-prior-15-shot_0 \
        logs/SemEval/$model-distribution-prior-25-shot_0 \
        logs/SemEval/$model-distribution-prior-35-shot_0 \
        logs/SemEval/$model-distribution-prior-45-shot_0 \
        logs/SemEval/$model-distribution-prior-55-shot_0 \
        logs/SemEval/$model-5-shot-distribution-prompt-5-shot_0 \
        logs/SemEval/$model-5-shot-distribution-prompt-15-shot_0 \
        logs/SemEval/$model-5-shot-distribution-prompt-25-shot_0 \
        logs/SemEval/$model-15-shot-distribution-prompt-5-shot_0 \
        logs/SemEval/$model-15-shot-distribution-prompt-15-shot_0 \
        logs/SemEval/$model-15-shot-distribution-prompt-25-shot_0 \
        logs/SemEval/$model-25-shot-distribution-prompt-5-shot_0 \
        logs/SemEval/$model-25-shot-distribution-prompt-15-shot_0 \
        logs/SemEval/$model-25-shot-distribution-prompt-25-shot_0 \
        logs/SemEval/$model-zero-shot-prompt-5-shot_0 \
        logs/SemEval/$model-zero-shot-prompt-15-shot_0 \
        logs/SemEval/$model-zero-shot-prompt-25-shot_0 \
        logs/SemEval/$model-distribution-traindev-prior-5-shot_0 \
        logs/SemEval/$model-distribution-traindev-prior-15-shot_0 \
        logs/SemEval/$model-distribution-traindev-prior-25-shot_0 \
        --alternative 5s 15s 25s \
        Cossim-5s Cossim-15s Cossim-25s Cossim-35s Cossim-45s Cossim-55s Cossim-65s \
        0s \
        Proxy-prior-5s Proxy-prior-15s Proxy-prior-25s Proxy-prior-35s Proxy-prior-45s Proxy-prior-55 \
        Prior-5s-prompt-5s Prior-5s-prompt-15s Prior-5s-prompt-25s \
        Prior-15s-prompt-5s Prior-15s-prompt-15s Prior-15s-prompt-25s \
        Prior-25s-prompt-5s Prior-25s-prompt-15s Prior-25s-prompt-25s \
        0s-prompt-5s 0s-prompt-15s 0s-prompt-25s \
        Prior-5s-traindev Prior-15s-traindev Prior-25s-traindev

model=meta-llama--Llama-2-70b-chat-hf
python scripts/compare_distributions.py SemEval --root-dir $dir/semeval2018task1 \
    --out logs/analysis/semeval-$model \
    --experiments logs/SemEval/$model-random-retrieval-5-shot_0 \
    logs/SemEval/$model-random-retrieval-15-shot_0 \
    logs/SemEval/$model-random-retrieval-25-shot_0 \
    logs/SemEval/$model-similarity-retrieval-5-shot_0 \
    logs/SemEval/$model-similarity-retrieval-15-shot_0 \
    logs/SemEval/$model-similarity-retrieval-25-shot_0 \
    logs/SemEval/$model-zero-shot-traindev-prior_0 \
    logs/SemEval/$model-random-prior-5-shot_0 \
    logs/SemEval/$model-random-prior-15-shot_0 \
    logs/SemEval/$model-random-prior-25-shot_0 \
    logs/SemEval/$model-distribution-prior-5-shot_0 \
    logs/SemEval/$model-distribution-prior-15-shot_0 \
    logs/SemEval/$model-distribution-prior-25-shot_0 \
    logs/SemEval/$model-5-shot-distribution-prompt-5-shot_0 \
    logs/SemEval/$model-5-shot-distribution-prompt-15-shot_0 \
    logs/SemEval/$model-5-shot-distribution-prompt-25-shot_0 \
    logs/SemEval/$model-15-shot-distribution-prompt-5-shot_0 \
    logs/SemEval/$model-15-shot-distribution-prompt-15-shot_0 \
    logs/SemEval/$model-15-shot-distribution-prompt-25-shot_0 \
    logs/SemEval/$model-25-shot-distribution-prompt-5-shot_0 \
    logs/SemEval/$model-25-shot-distribution-prompt-15-shot_0 \
    logs/SemEval/$model-25-shot-distribution-prompt-25-shot_0 \
    logs/SemEval/$model-zero-shot-prompt-5-shot_0 \
    logs/SemEval/$model-zero-shot-prompt-15-shot_0 \
    logs/SemEval/$model-zero-shot-prompt-25-shot_0 \
    logs/SemEval/$model-distribution-traindev-prior-5-shot_0 \
    logs/SemEval/$model-distribution-traindev-prior-15-shot_0 \
    logs/SemEval/$model-distribution-traindev-prior-25-shot_0 \
    --alternative 5s 15s 25s \
    Cossim-5s Cossim-15s Cossim-25s \
    Random-prior-5s Random-prior-15s Random-prior-25s \
    0s \
    Proxy-prior-5s Proxy-prior-15s Proxy-prior-25s \
    Prior-5s-prompt-5s Prior-5s-prompt-15s Prior-5s-prompt-25s \
    Prior-15s-prompt-5s Prior-15s-prompt-15s Prior-15s-prompt-25s \
    Prior-25s-prompt-5s Prior-25s-prompt-15s Prior-25s-prompt-25s \
    0s-prompt-5s 0s-prompt-15s 0s-prompt-25s \
    Prior-5s-traindev Prior-15s-traindev Prior-25s-traindev

for model in google--gemma-7b allenai--OLMo-7B; do
    python scripts/compare_distributions.py SemEval --root-dir $dir/semeval2018task1 \
        --out logs/analysis/semeval-$model \
        --experiments logs/SemEval/$model-random-retrieval-5-shot_0 \
        logs/SemEval/$model-random-retrieval-15-shot_0 \
        logs/SemEval/$model-random-retrieval-25-shot_0 \
        logs/SemEval/$model-similarity-retrieval-5-shot_0 \
        logs/SemEval/$model-similarity-retrieval-15-shot_0 \
        logs/SemEval/$model-similarity-retrieval-25-shot_0 \
        logs/SemEval/$model-random-prior-5-shot_0 \
        logs/SemEval/$model-random-prior-15-shot_0 \
        logs/SemEval/$model-random-prior-25-shot_0 \
        logs/SemEval/$model-distribution-prior-5-shot_0 \
        logs/SemEval/$model-distribution-prior-15-shot_0 \
        logs/SemEval/$model-distribution-prior-25-shot_0 \
        logs/SemEval/$model-5-shot-distribution-prompt-5-shot_0 \
        logs/SemEval/$model-5-shot-distribution-prompt-15-shot_0 \
        logs/SemEval/$model-5-shot-distribution-prompt-25-shot_0 \
        logs/SemEval/$model-15-shot-distribution-prompt-5-shot_0 \
        logs/SemEval/$model-15-shot-distribution-prompt-15-shot_0 \
        logs/SemEval/$model-15-shot-distribution-prompt-25-shot_0 \
        logs/SemEval/$model-25-shot-distribution-prompt-5-shot_0 \
        logs/SemEval/$model-25-shot-distribution-prompt-15-shot_0 \
        logs/SemEval/$model-25-shot-distribution-prompt-25-shot_0 \
        logs/SemEval/$model-distribution-traindev-prior-5-shot_0 \
        logs/SemEval/$model-distribution-traindev-prior-15-shot_0 \
        logs/SemEval/$model-distribution-traindev-prior-25-shot_0 \
        --alternative 5s 15s 25s \
        Cossim-5s Cossim-15s Cossim-25s \
        Random-prior-5s Random-prior-15s Random-prior-25s \
        Proxy-prior-5s Proxy-prior-15s Proxy-prior-25s \
        Prior-5s-prompt-5s Prior-5s-prompt-15s Prior-5s-prompt-25s \
        Prior-15s-prompt-5s Prior-15s-prompt-15s Prior-15s-prompt-25s \
        Prior-25s-prompt-5s Prior-25s-prompt-15s Prior-25s-prompt-25s \
        Prior-5s-traindev Prior-15s-traindev Prior-25s-traindev
done

for model in google--gemma-7b allenai--OLMo-7B meta-llama--Llama-2-13b-chat-hf; do
    python scripts/compare_distributions.py GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
        --out logs/analysis/goemotions-$model \
        --experiments logs/GoEmotions/$model-random-retrieval-5-shot_0 \
        logs/GoEmotions/$model-random-retrieval-15-shot_0 \
        logs/GoEmotions/$model-random-retrieval-25-shot_0 \
        logs/GoEmotions/$model-similarity-retrieval-5-shot_0 \
        logs/GoEmotions/$model-similarity-retrieval-15-shot_0 \
        logs/GoEmotions/$model-similarity-retrieval-25-shot_0 \
        logs/GoEmotions/$model-random-prior-5-shot_0 \
        logs/GoEmotions/$model-random-prior-15-shot_0 \
        logs/GoEmotions/$model-random-prior-25-shot_0 \
        logs/GoEmotions/$model-distribution-prior-5-shot_0 \
        logs/GoEmotions/$model-distribution-prior-15-shot_0 \
        logs/GoEmotions/$model-distribution-prior-25-shot_0 \
        logs/GoEmotions/$model-5-shot-distribution-prompt-5-shot_0 \
        logs/GoEmotions/$model-5-shot-distribution-prompt-15-shot_0 \
        logs/GoEmotions/$model-5-shot-distribution-prompt-25-shot_0 \
        logs/GoEmotions/$model-15-shot-distribution-prompt-5-shot_0 \
        logs/GoEmotions/$model-15-shot-distribution-prompt-15-shot_0 \
        logs/GoEmotions/$model-15-shot-distribution-prompt-25-shot_0 \
        logs/GoEmotions/$model-25-shot-distribution-prompt-5-shot_0 \
        logs/GoEmotions/$model-25-shot-distribution-prompt-15-shot_0 \
        logs/GoEmotions/$model-25-shot-distribution-prompt-25-shot_0 \
        logs/GoEmotions/$model-distribution-traindev-prior-5-shot_0 \
        logs/GoEmotions/$model-distribution-traindev-prior-15-shot_0 \
        logs/GoEmotions/$model-distribution-traindev-prior-25-shot_0 \
        --alternative 5s 15s 25s \
        Cossim-5s Cossim-15s Cossim-25s \
        Random-prior-5s Random-prior-15s Random-prior-25s \
        Proxy-prior-5s Proxy-prior-15s Proxy-prior-25s \
        Prior-5s-prompt-5s Prior-5s-prompt-15s Prior-5s-prompt-25s \
        Prior-15s-prompt-5s Prior-15s-prompt-15s Prior-15s-prompt-25s \
        Prior-25s-prompt-5s Prior-25s-prompt-15s Prior-25s-prompt-25s \
        Prior-5s-traindev Prior-15s-traindev Prior-25s-traindev
done

model=meta-llama--Llama-2-70b-chat-hf
python scripts/compare_distributions.py GoEmotions --root-dir $dir/goemotions --emotion-clustering-json $dir/goemotions/emotion_clustering.json \
    --out logs/analysis/goemotions-$model \
    --experiments logs/GoEmotions/$model-random-retrieval-5-shot_0 \
    logs/GoEmotions/$model-random-retrieval-15-shot_0 \
    logs/GoEmotions/$model-random-retrieval-25-shot_0 \
    logs/GoEmotions/$model-similarity-retrieval-25-shot_0 \
    logs/GoEmotions/$model-distribution-prior-5-shot_0 \
    logs/GoEmotions/$model-distribution-prior-15-shot_0 \
    logs/GoEmotions/$model-distribution-prior-25-shot_0 \
    logs/GoEmotions/$model-5-shot-distribution-prompt-5-shot_0 \
    logs/GoEmotions/$model-15-shot-distribution-prompt-15-shot_0 \
    logs/GoEmotions/$model-25-shot-distribution-prompt-25-shot_0 \
    logs/GoEmotions/$model-distribution-traindev-prior-5-shot_0 \
    logs/GoEmotions/$model-distribution-traindev-prior-15-shot_0 \
    logs/GoEmotions/$model-distribution-traindev-prior-25-shot_0 \
    --alternative 5s 15s 25s \
    Cossim-25s \
    Proxy-prior-5s Proxy-prior-15s Proxy-prior-25s \
    Prior-5s-prompt-5s \
    Prior-15s-prompt-15s \
    Prior-25s-prompt-25s \
    Prior-5s-traindev Prior-15s-traindev Prior-25s-traindev
