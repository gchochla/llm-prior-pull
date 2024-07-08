# The Strong Pull of Prior Knowledge in Large Language Models and Its Impact on Emotion Recognition

This repo contains the official implementation of [The Strong Pull of Prior Knowledge in Large Language Models and Its Impact on Emotion Recognition](https://arxiv.org/pdf/2403.17125), published in ACII'24.

## Abstract

> In-context Learning (ICL) has emerged as a powerful paradigm for performing natural language tasks with Large Language Models (LLM) without updating the models' parameters, in contrast to the traditional gradient-based finetuning. The promise of ICL is that the LLM can adapt to perform the present task at a competitive or state-of-the-art level at a fraction of the cost. The ability of LLMs to perform tasks in this few-shot manner relies on their background knowledge of the task (or *task priors*).
However, recent work has found that, unlike traditional learning, LLMs are unable to fully integrate information from demonstrations that contrast task priors. This can lead to performance saturation at suboptimal levels, especially for subjective tasks such as emotion recognition, where the mapping from text to emotions can differ widely due to variability in human annotations.
In this work, we design experiments and propose measurements to explicitly quantify the consistency of proxies of LLM priors and their pull on the posteriors. We show that LLMs have strong yet inconsistent priors in emotion recognition that ossify their predictions. We also find that the larger the model, the stronger these effects become.
Our results suggest that caution is needed when using ICL with larger LLMs for affect-centered tasks outside their pre-training domain and when interpreting ICL results.

## Installation

This repo uses `Python 3.9`+ (type hints, for example, won't work with previous versions). After you create and activate your virtual environment (with conda, venv, etc), install local dependencies with:

```bash
pip install -e .[dev]
```

## Run experiments

Experiments are logged with [legm](https://github.com/gchochla/legm), so refer to the documentation there for an interpretation of the resulting `logs` folder, but navigating should be intuitive enough with some trial and error.

To run the GoEmotions experiments, we recommend using the emotion pooling we set up based on the hierarchical clustering (besides, the bash scripts are set up for it). To do so, create the file `emotion_clustering.json` under the root folder of the dataset with the following contents:

```JSON
{
    "joy": [
        "amusement",
        "excitement",
        "joy",
        "love"
    ],
    "optimism": [
        "desire",
        "optimism",
        "caring"
    ],
    "admiration": [
        "pride",
        "admiration",
        "gratitude",
        "relief",
        "approval",
        "realization"
    ],
    "surprise": [
        "surprise",
        "confusion",
        "curiosity"
    ],
    "fear": [
        "fear",
        "nervousness"
    ],
    "sadness": [
        "remorse",
        "embarrassment",
        "disappointment",
        "sadness",
        "grief"
    ],
    "anger": [
        "anger",
        "disgust",
        "annoyance",
        "disapproval"
    ]
}
```

Also, create a subset of the development set of GoEmotions, and name the file `small_dev.tsv`, for example by:

```bash
head -n 800 $dir/goemotions/dev.tsv > $dir/goemotions/small_dev.tsv
```

We also need to extract the IDs in a separate file:

```python
import os
import pandas as pd

root_dir = "/path/to/parent/folder/of/dataset"
df = pd.read_csv(os.path.join(root_dir, "goemotions", "small_dev.tsv"), sep="\t", header=None)
with open(os.path.join(root_dir, "goemotions", "small_dev_ids.txt"), "w") as fp:
    fp.write("\n".join(df[2].values.tolist()))
```

Also, we need an equivalent to create a file containing the subset of example IDs used in the prompts:

```bash
python scripts/get_random_ids_from_train.py GoEmotions --root-dir $dir/goemotions --annotation-mode aggregate --text-preprocessor false --shot 5 15 25 35 45 55 65 75 --seed {0..2} --emotion-clustering-json $dir/goemotions/emotion_clustering.json  --output-filename $dir/goemotions/train_useful_ids.txt
```

To run all experiments (presented in the supplementary materials), you can run the bash scripts in `scripts` starting with `pipeline-`, and providing the cuda ID (`-c 0`), the model path or name (`-m meta-llama/Llama-2-7b-chat-hf` or `-m gpt-3.5-turbo`), and the directory to the parent folder of the dataset (`-d /path/to/parent/folder/of/dataset`).
