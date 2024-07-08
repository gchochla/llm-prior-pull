import os
import yaml
from typing import Any, Sequence, Callable, Literal, Mapping
from abc import abstractmethod
from copy import deepcopy

import torch
from ember.dataset import BaseDataset
from transformers import PreTrainedTokenizer, AutoTokenizer


class TextDataset(BaseDataset):
    """Base dataset for text classification.

    Attributes:
        root_dir: path to root directory containing data.
        splits: splits to load.
        ids: list of effective IDs (used in __getitem__, might not correspond
            to dataset directly, e.g. augmented for annotators).
        real_ids: list of real IDs.
        multilabel: whether dataset is multilabel or not.
        annotator_labels: whether dataset has annotator labels or not.
        label_set: set of labels.
        source_domain: source domain of dataset.
        data: dictionary containing data indexed by IDs.
        preprocessor: function to preprocess text.
        annotation_mode: mode to load, one of "aggregate", "annotator",
            "both". If "aggregate", only the aggregated label is returned.
            If "annotator", only the annotator labels are returned. If
            "both", both the aggregated and annotator labels are returned.
        id_separator: separator for IDs of annotator and example.
        annotator2inds: mapping between annotator IDs and indices.
        annotator2ids: mapping between annotator IDs and sample IDs.
        annotator2label_inds: mapping between annotator IDs and per label indices.
    """

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            root_dir=dict(
                type=str,
                help="path to root directory containing data",
            ),
            splits=dict(
                type=str,
                nargs="+",
                splits=["train", "dev", "test"],
                help="train splits to load",
            ),
            ids_filename=dict(
                type=str,
                nargs="+",
                splits=["train", "dev", "test"],
                help="path to file containing subset of IDs to retain",
            ),
            annotation_mode=dict(
                type=str,
                choices=["aggregate", "annotator", "both"],
                default="aggregate",
                help="mode to load, one of 'aggregate', 'annotator', 'both'. "
                "If 'aggregate', only the aggregated label is returned. "
                "If 'annotator', only the annotator labels are returned. "
                "If 'both', both the aggregated and annotator labels are returned. "
                "Error is raised if 'annotator' or 'both' is chosen but the dataset "
                "does not have annotator labels.",
                metadata=dict(name=True),
                searchable=True,
            ),
            debug_len=dict(
                type=int,
                splits=["train", "dev", "test"],
                help="number of examples to load for debugging",
            ),
            text_preprocessor=dict(
                type=bool,
                help="whether to use text preprocessor",
                searchable=True,
            ),
        )

    @property
    @abstractmethod
    def multilabel(self) -> bool:
        """Whether dataset is multilabel or not."""
        pass

    @property
    @abstractmethod
    def annotator_labels(self) -> bool:
        """Whether dataset has annotator labels or not."""
        pass

    @property
    @abstractmethod
    def source_domain(self) -> str:
        """Source domain of dataset."""
        pass

    def __init__(
        self,
        root_dir: str,
        splits: str | Sequence[str],
        text_preprocessor: Callable[[str], str] | None = None,
        annotation_mode: Literal[
            "aggregate", "annotator", "both"
        ] = "aggregate",
        debug_len: int | None = None,
        ids_filename: str | list[str] | None = None,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            root_dir: path to root directory containing data.
            splits: splits to load.
            text_preprocessor: function to preprocess text.
            annotation_mode: mode to load, one of "aggregate", "annotator",
                "both". If "aggregate", only the aggregated label is returned.
                If "annotator", only the annotator labels are returned. If
                "both", both the aggregated and annotator labels are returned.
            debug_len: number of examples to load for debugging.
            ids_filename: path to file containing subset of IDs to retain.
        """

        assert (
            annotation_mode == "aggregate" or self.annotator_labels
        ), f"No annotator labels, but annotation mode was set to {annotation_mode}."

        super().__init__()
        self.id_separator = "__"
        self.root_dir = root_dir
        self.splits = [splits] if isinstance(splits, str) else splits
        self.ids_filename = (
            [ids_filename] if isinstance(ids_filename, str) else ids_filename
        )
        self.preprocessor = text_preprocessor or (lambda x: x)
        self.annotation_mode = annotation_mode
        self.data, self.label_set = self._load_data()
        self._debug_len = debug_len
        self._convert_data()
        self.real_ids = list(self.data)
        (
            self.ids,
            self.annotator2inds,
            self.annotator2ids,
        ) = self._extend_dataset_for_annotators()
        self.annotator2label_inds = self._extend_dataset_for_sampling()

    def _extend_dataset_for_annotators(
        self,
    ) -> tuple[
        list[tuple[Any, str]], dict[str, list[int]], dict[str, list[str]]
    ]:
        """Creates IDs for dataset according to annotation mode,
        mapping between annotator IDs and indices, and mapping between
        annotator IDs and sample IDs."""

        if self.annotation_mode == "aggregate":
            ids = [(_id, "aggregate") for _id in self.data]
            annotator2inds = {"aggregate": list(range(len(ids)))}
            annotator2ids = {"aggregate": list(self.data)}

        else:
            ids = []
            annotator2inds = {}
            annotator2ids = {}
            for _id, datum in self.data.items():
                for worker_id in datum["label"].keys():
                    if (
                        self.annotation_mode == "both"
                        or worker_id != "aggregate"
                    ):
                        annotator2inds.setdefault(worker_id, []).append(
                            len(ids)
                        )
                        annotator2ids.setdefault(worker_id, []).append(_id)
                        ids.append((_id, worker_id))

        return ids, annotator2inds, annotator2ids

    def _extend_dataset_for_sampling(self) -> dict[str, dict[str, list[int]]]:
        """Creates mapping between annotator IDs and per label indices
        for easier sampling."""

        annotator2label_inds = {}
        for annotator, inds in self.annotator2inds.items():
            annotator2label_inds[annotator] = {
                label: [i for i in inds if self[i]["label"][j] == 1]
                for j, label in enumerate(self.label_set)
            }

        return annotator2label_inds

    def _convert_data(self):
        """Converts simple tensor labels to annotator dict format (i.e.
        {"ann1": this label, "ann2": that label, "aggregate": aggregated label})
        by adding "aggregate" key to each label (e.g. {"aggregate": this label}};
        Implemented for datasets without annotator labels, so that returned labels
        in `_load_data` can be simple tensors). Also makes all IDs strings.
        """

        if self.ids_filename:
            ids_to_keep = set()
            for fn in self.ids_filename:
                with open(fn) as fp:
                    ids_to_keep.update(set([l.strip() for l in fp.readlines()]))

            self.data = {k: v for k, v in self.data.items() if k in ids_to_keep}

        ids = list(self.data)
        for i, _id in enumerate(ids):
            # pop because it will be re-inserted with the str key
            datum = self.data.pop(_id)

            if i >= (self._debug_len or float("inf")):
                continue  # not break because we want to pop all

            if torch.is_tensor(datum["label"]):
                # then no aggregate key
                datum["label"] = dict(aggregate=datum["label"])
            else:
                # we have aggregate and annotators, make sure everything a str
                worker_ids = list(datum["label"])
                for worker_id in worker_ids:
                    datum["label"][str(worker_id)] = datum["label"].pop(
                        worker_id
                    )

            self.data[str(_id)] = datum

    def __len__(self):
        """Returns length of dataset."""
        # if self._debug_len is None:
        #     return len(self.ids)

        # return min(self._debug_len, len(self.ids))
        return len(self.ids)

    def __getitem__(self, idx) -> tuple[Any, str, str]:
        """Returns text and label at index `idx`."""
        _id, worker_id = self.ids[idx]
        datum = deepcopy(self.data[_id])
        datum["label"] = datum["label"][worker_id]
        return dict(id=_id + self.id_separator + worker_id, **datum)

    def index_label_set(
        self, label: torch.Tensor | int | list[int]
    ) -> str | list[str]:
        """Returns label names given numerical `label`."""
        if not self.multilabel:
            if torch.is_tensor(label):
                label = int(label.item())
            return self.label_set[label]

        if torch.is_tensor(label):
            label = label.tolist()
        return [self.label_set[i] for i, l in enumerate(label) if l == 1]

    def get_label_from_str(self, label: str | list[str]) -> torch.Tensor:
        """Returns label index given string `label`."""

        multilabel_no_label = torch.zeros(len(self.label_set))

        if isinstance(label, str):
            try:
                label = self.label_set.index(label)
            except ValueError:
                assert self.multilabel, (
                    f"Label {label} not found in label set {self.label_set}. "
                    "Only multilabel datasets can have no label"
                )
                return multilabel_no_label

            if self.multilabel:
                return multilabel_no_label.scatter(0, torch.tensor(label), 1)
            return torch.tensor(label)

        assert (
            self.multilabel
        ), "Cannot convert list of labels to single label for non-multilabel dataset."

        return (
            torch.stack(
                [self.get_label_from_str(l) for l in label]
                or [multilabel_no_label]  # in case of empty list
            )
            .sum(0)
            .clamp_max(1)
        )

    def getitem_by_id(self, _id: Any) -> dict[str, str | torch.Tensor]:
        """Returns item with ID `_id`."""
        if self.id_separator in _id:
            return self.data[_id.split(self.id_separator)[0]]
        return self.data[_id]

    @abstractmethod
    def _load_data(
        self,
    ) -> tuple[
        dict[Any, dict[str, str | torch.Tensor | dict[str, torch.Tensor]]],
        list[str],
    ]:
        """Loads data and returns IDs, texts, labels in a dictionary
        indexed by IDs that contains another dictionary with `"text"`
        (after preprocessing), `"original_text"` (before preprocessing),
        and `"label"` keys, e.g.,
        {
            id1: {"text": "lorem ipsum", "label": torch.tensor([4])},
            ...
        },
        and the label set. Alternatively, if the dataset contains annotator
        labels, the `"label"` key can be a dictionary of annotator labels,
        along with the aggregated label, e.g.,
        {
            id1: {
                "text": "lorem ipsum",
                "label": {
                    "ann1": torch.tensor([4]),
                    "ann2": torch.tensor([3, 4]),
                    "aggregate": torch.tensor([4]),
                },
            },
            ...
        }
        """


class TextDatasetWithPriors(TextDataset):
    """Base dataset for text classification with prior labels
    from model prediction logs using `legm.ExperimentManager`.

    Attributes:
        pred_log_dir: path to experiment directory containing predictions.
        Check `TextDataset` for other attributes.
    """

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = TextDataset.argparse_args()
        args.update(
            dict(
                pred_log_dir=dict(
                    type=str,
                    help="path to experiment directory containing predictions to use as labels",
                    searchable=True,
                    splits=["train", "dev", "test"],
                    metadata=dict(
                        name=True,
                        name_transform=lambda x: (
                            None
                            if x is None or x.lower() == "none"
                            else os.path.split(x)[1]
                        ),
                        parent="label_mode",
                        # parent_active=lambda x: (x in ["preds", "contrastive"]),
                    ),
                ),
                pred_log_index=dict(
                    type=int,
                    help="index of prediction log experiment to use as labels",
                    searchable=True,
                    splits=["train", "dev", "test"],
                    metadata=dict(
                        disable_comparison=True, parent="pred_log_dir"
                    ),
                ),
            )
        )
        return args

    def __init__(
        self,
        pred_log_dir: str | None = None,
        pred_log_index: int | str | None = None,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            pred_label_log_dir: path to experiment directory containing
                predictions.
            Check `TextDataset` for other arguments.
        """

        super().__init__(*args, **kwargs)

        if pred_log_dir:
            pred_labels = self._load_labels_from_logs(
                pred_log_dir, pred_log_index
            )
            for _id in pred_labels:
                if _id in self.data:
                    self.data[_id]["pred_label"] = pred_labels[_id]
        else:
            for _id, datum in self.data.items():
                datum["pred_label"] = {
                    worker_id: None for worker_id in datum["label"]
                }
        self.pred_log_dir = pred_log_dir

    def __getitem__(self, idx) -> tuple[Any, str, str]:
        item = super().__getitem__(idx)
        if "pred_label" in item:
            item["pred_label"] = item["pred_label"][
                item["id"].split(self.id_separator)[1]
            ]
        return item

    def _load_labels_from_logs(
        self, experiment_dir: str, pred_log_index: int | str | None = None
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Loads annotator labels from logs in `experiment_dir` and returns
        a dictionary indexed by IDs that contains another dictionary with
        annotator IDs as keys and their labels as values, e.g.,
        {
            id1: {
                "ann1": torch.tensor([4]),
                "ann2": torch.tensor([3, 4]),
            },
            ...
        }
        """

        labels = {}

        label_fn = os.path.join(experiment_dir, "indexed_metrics.yml")
        with open(label_fn) as fp:
            logs = yaml.safe_load(fp)[f"experiment_{pred_log_index or 0}"]

        if "description" in logs:
            del logs["description"]

        for _id, example_metrics in logs.items():
            labels[_id] = {
                "aggregate": self.get_label_from_str(
                    example_metrics["test_pred"]
                )
            }

        ann_label_fn = os.path.join(experiment_dir, "annotator_preds.yml")
        with open(ann_label_fn) as f:
            ann_logs = yaml.safe_load(f)["experiment_0"]

        for worker_id, annotator_metrics in ann_logs.items():
            for _id, example_metrics in annotator_metrics.items():
                labels[_id][worker_id] = self.get_label_from_str(
                    example_metrics["test_pred"]
                )

        return labels


class TokenizationMixin:
    """Mixin for tokenizing text for the `transformers` library.
    MUST be inherited before any other class because it requires
    init arguments.

    Attributes:
        _tokenization_mixin_data: dictionary containing the tokenizer and
            the maximum tokenization length.
    """

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            max_length=dict(
                type=int,
                help="maximum length of tokenized text",
            )
        )

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        model_name_or_path: str | None = None,
        max_length: int | None = None,
        cache_dir: str | None = None,
        trust_remote_code: bool = False,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            tokenizer: tokenizer to use for tokenizing utterances,
                otherwise provide `model_name_or_path`.
            model_name_or_path: model name or path to load tokenizer from,
                otherwise provide `tokenizer`.
            max_length: maximum length of utterance.
            cache_dir: path to `transformers` cache directory.
            trust_remote_code: whether to trust remote code.

        Raises:
            AssertionError: if neither `tokenizer` nor `model_name_or_path`
                are provided.
        """

        assert (
            tokenizer is not None or model_name_or_path is not None
        ), "Either tokenizer or model_name_or_path must be provided."

        self._tokenization_mixin_data = dict(
            tokenizer=tokenizer
            or AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            ),
            max_length=max_length,
        )
        if self._tokenization_mixin_data["tokenizer"].pad_token is None:
            self._tokenization_mixin_data["tokenizer"].pad_token = getattr(
                self._tokenization_mixin_data["tokenizer"], "unk_token", "[PAD]"
            )
        self._tokenization_mixin_data["tokenizer"].padding_side = "left"

        super().__init__(*args, **kwargs)

    def dict_tokenize(
        self,
        data: dict[Any, dict[str, str | torch.Tensor]],
        text_preprocessor: Callable[[str], str],
    ):
        """Tokenizes text in `data` in-place.

        Args:
            data: dictionary containing text to tokenize.
            text_preprocessor: function to preprocess text.
        """
        for k in data:
            data[k]["encoding"] = self.tokenize(
                text_preprocessor(data[k]["text"])
            )

    def tokenize(self, text: str) -> Mapping[str, torch.Tensor]:
        """Tokenizes text.

        Args:
            text: text to tokenize.

        Returns:
            Tensor of token ids.
        """
        if self._tokenization_mixin_data["max_length"] is None:
            return self._tokenization_mixin_data["tokenizer"](
                text, return_tensors="pt", return_token_type_ids=False
            )

        return self._tokenization_mixin_data["tokenizer"](
            text,
            max_length=self._tokenization_mixin_data["max_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
        )

    def decode(self, input_ids: torch.Tensor) -> str:
        return self._tokenization_mixin_data["tokenizer"].decode(
            input_ids.squeeze()
        )

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenization_mixin_data["tokenizer"]

    def tokenize_conversation(self, conversation: list[dict[str, str]]):
        """Tokenizes a conversation.

        Args:
            conversation: list of dictionaries containing text to tokenize.
                Keys are "role" and "content". "role" is either "user", "system",
                or "assistant".

        Returns:
            List of tokenized texts.
        """

        return self._tokenization_mixin_data["tokenizer"].apply_chat_template(
            conversation, return_tensors="pt"
        )
