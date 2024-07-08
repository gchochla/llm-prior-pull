import os
import langcodes
import json
from typing import Any

import torch
import pandas as pd

from llm_pull.base_datasets import TextDatasetWithPriors


class SemEval2018Task1Ec(TextDatasetWithPriors):
    """Plain text dataset for `SemEval 2018 Task 1: Affect in Tweets`
    (https://competitions.codalab.org/competitions/17751). Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDatasetWithPriors` for attributes.
        language: language to load.
    """

    multilabel = True
    annotator_labels = False
    name = "SemEval 2018 Task 1"
    source_domain = "Twitter"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = TextDatasetWithPriors.argparse_args()
        args.update(
            dict(
                language=dict(
                    type=str, default="english", help="language to load"
                )
            )
        )
        return args

    def __init__(self, language, *args, **kwargs):
        """Initializes dataset.

        Args:
            language: language to load.
            Check `TextDataset` for other arguments.
        """
        self.language = language
        super().__init__(*args, **kwargs)

    def _load_data(
        self,
    ) -> tuple[list[str], list[str], torch.Tensor, list[str]]:
        split_mapping = dict(
            train="train", dev="dev", test="test-gold", smalldev="smalldev"
        )
        filenames = [
            os.path.join(
                self.root_dir,
                self.language.title(),
                "E-c",
                f"2018-E-c-{langcodes.find(self.language.lower()).language.title()}-{split_mapping[s]}.txt",
            )
            for s in self.splits
        ]

        df = pd.concat(
            [pd.read_csv(filename, sep="\t") for filename in filenames]
        )

        emotions = list(df.columns[2:])
        sorted_emotions = sorted(emotions)
        emotion_inds = [emotions.index(e) for e in sorted_emotions]
        texts = df.Tweet.values.tolist()
        ids = df.ID.values.tolist()

        labels = torch.tensor(df.iloc[:, 2:].values[:, emotion_inds]).float()

        return {
            _id: dict(text=self.preprocessor(t), original_text=t, label=l)
            for _id, t, l in zip(ids, texts, labels)
        }, sorted_emotions


class GoEmotions(TextDatasetWithPriors):
    """Plain text dataset for `GoEmotions`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDatasetWithPriors` for attributes.
    """

    multilabel = True
    annotator_labels = True
    name = "GoEmotions"
    source_domain = "Reddit"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = TextDatasetWithPriors.argparse_args() | dict(
            emotion_clustering_json=dict(
                type=str,
                help="JSON file with clustering of emotions",
            )
        )
        return args

    def __init__(self, emotion_clustering_json, *args, **kwargs):
        """Initializes dataset.

        Args:
            emotion_clustering_json: JSON file with clustering of emotions.
            Check `TextDatasetWithPriors` for other arguments.
        """
        self.emotion_clustering_json = emotion_clustering_json
        super().__init__(*args, **kwargs)

    def _multilabel_one_hot(
        self, labels: "np.ndarray", n_classes: int = 27
    ) -> torch.Tensor:
        """GoEmotions-specific label transformer to multilable one-hot,
        neutral emotion is discarded (represented as 0s)."""

        labels = [
            list(filter(lambda x: x < n_classes, map(int, lbl.split(","))))
            for lbl in labels
        ]
        new_labels = [
            torch.nn.functional.one_hot(
                torch.tensor(lbl, dtype=int), n_classes
            ).sum(0)
            for lbl in labels
        ]
        return torch.stack(new_labels)

    def _subset_emotions(
        self,
        annotations: dict[Any, dict[str, str | torch.Tensor]],
        emotions: list[str],
    ) -> list[str]:
        """Transforms emotions to a subset of emotions based on clustering
        in `emotion_clustering_json`. Each new label is union of old labels."""

        if not self.emotion_clustering_json:
            return emotions

        with open(self.emotion_clustering_json) as fp:
            clustering = json.load(fp)

        new_emotions = list(clustering)

        for annotation in annotations.values():
            for worker_id, label in annotation["label"].items():
                new_label = torch.zeros(len(new_emotions))

                for i, emotion in enumerate(new_emotions):
                    for old_emotion in clustering[emotion]:
                        new_label[i] += label[emotions.index(old_emotion)]

                annotation["label"][worker_id] = new_label.clamp(0, 1)

        return new_emotions

    def _load_data(
        self,
    ) -> tuple[dict[Any, dict[str, str | torch.Tensor]], list[str]]:
        ## read emotions from file
        emotion_fn = os.path.join(self.root_dir, "emotions.txt")
        emotions = pd.read_csv(emotion_fn, header=None)[0].values.tolist()[
            :-1
        ]  # gets rid of neutral emotion

        ## read aggregated labels from file
        filenames = [
            os.path.join(self.root_dir, f"{split}.tsv") for split in self.splits
        ]

        df = pd.concat(
            [pd.read_csv(fn, sep="\t", header=None) for fn in filenames]
        )

        ids = df.iloc[:, -1].values.tolist()
        aggr_labels = {
            _id: y
            for _id, y in zip(
                ids,
                self._multilabel_one_hot(
                    df.iloc[:, 1].values, len(emotions)
                ).float(),
            )
        }

        if self.annotation_mode == "aggregate":
            annotations = {
                _id: dict(
                    text=self.preprocessor(text),
                    original_text=text,
                    label={"aggregate": aggr_labels[_id]},
                )
                for _id, text in zip(ids, df.iloc[:, 0].values)
            }
            self.annotators = set()

        else:
            ## read annotator labels from file
            filenames = [
                os.path.join(self.root_dir, f"goemotions_{i}.csv")
                for i in range(1, 4)
            ]
            df = pd.concat([pd.read_csv(fn) for fn in filenames])
            df = df[df["id"].isin(set(ids))]
            df["labels"] = [
                [row[lbl] for lbl in emotions] for _, row in df.iterrows()
            ]

            groupby = df[["text", "rater_id", "id", "labels"]].groupby("id")
            annotations = groupby.agg(
                {
                    "text": lambda x: x.iloc[0],
                    "rater_id": lambda x: x.tolist(),
                    "labels": lambda x: x.tolist(),
                }
            )

            annotations = {
                _id: dict(
                    text=self.preprocessor(text),
                    original_text=text,
                    label={
                        worker_id: torch.tensor(labels).float()
                        for worker_id, labels in zip(rater_ids, label_list)
                    }
                    | {"aggregate": aggr_labels[_id]},
                )
                for _id, text, rater_ids, label_list in annotations.itertuples()
            }

            self.annotators = set(df["rater_id"].unique())

        emotions = self._subset_emotions(annotations, emotions)

        return annotations, emotions
