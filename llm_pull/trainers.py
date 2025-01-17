from typing import Any

import torch
from torch.utils.data import DataLoader
from ember.trainer import BaseTrainer
from sklearn.metrics import f1_score, jaccard_score


class PromptEvaluator(BaseTrainer):
    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = BaseTrainer.argparse_args()
        args_discard = [
            "save_model",
            "discard_classifier",
            "classifier_layer_name",
            "lr",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "weight_decay",
            "eval_steps",
            "max_steps",
            "num_train_epochs",
            "train_batch_size",
            "eval_batch_size",
            "warmup_ratio",
            "early_stopping_patience",
            "early_stopping_metric",
            "early_stopping_delta",
            "early_stopping_lower_better",
        ]
        for arg in args_discard:
            args.pop(arg)
        return args

    def __init__(self, experiment_manager, *args, **kwargs):
        setattr(experiment_manager, "save_model", False)
        setattr(experiment_manager, "num_train_epochs", None)
        setattr(experiment_manager, "max_steps", -1)
        setattr(experiment_manager, "early_stopping_metric", None)
        setattr(experiment_manager, "eval_batch_size", 1)
        kwargs["experiment_manager"] = experiment_manager
        super().__init__(*args, **kwargs)

    def get_logits_from_model(
        self, return_vals: Any, *args, **kwargs
    ) -> torch.Tensor:

        batch_preds: list[list[str]] = return_vals["preds"]
        if self.any_dataset.test_dataset.multilabel:
            preds = torch.tensor(
                [
                    [
                        int(label in preds)
                        for label in self.any_dataset.label_set
                    ]
                    for preds in batch_preds
                ],
                device=self.exp_manager.device,
            )
        else:
            for pred in batch_preds:
                assert (
                    len(pred) == 1
                ), f"Only one prediction is expected, got {pred}."

            # TODO: does not handle case of no predictions,
            # which may not be handled by label word embeddings
            # bcs of thresholding, implement naive predictor
            preds = torch.tensor(
                [
                    [self.any_dataset.label_set.index(pred[0])]
                    for pred in batch_preds
                ],
                device=self.exp_manager.device,
            )

        return preds

    def get_log_outputs_from_model(
        self, return_vals: dict[str, str | torch.Tensor]
    ) -> list[tuple[str | torch.Tensor, str | torch.Tensor]]:
        if "text" in return_vals:
            outs = return_vals["text"]
            residual_outs = return_vals.get("residual_text", [None] * len(outs))
        else:
            outs = return_vals["ids"]
            residual_outs = return_vals.get("residual_ids", None)
            if residual_outs is not None:
                residual_outs = residual_outs.tolist()
            else:
                residual_outs = [None] * len(outs)

        log_vals = list(zip(outs, residual_outs))
        return log_vals

    def calculate_cls_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        train: bool,
        aggregate: bool = True,
    ) -> torch.Tensor:
        if aggregate:
            return torch.tensor(0.0, device=self.exp_manager.device)
        return torch.zeros(len(labels), device=self.exp_manager.device)

    def input_batch_kwargs(self, batch: dict[str, Any]) -> dict[str, Any]:
        encoding = {k: v[0] for k, v in batch["encoding"].items()}
        return encoding | dict(
            # assumes that the text appears first, which is reasonable for causal LMs
            cutoff_str=self.any_dataset.incontext_prompt.split("{text}")[
                0
            ].strip(),
            label_parser=self.any_dataset.label_parser,
        )

    def batch_labels(self, batch: dict[str, Any]):
        return batch["label"]

    def batch_ids(self, batch: dict[str, Any]):
        return batch["id"]

    def get_eval_preds_from_batch(
        self, logits: torch.Tensor
    ) -> list[list[int]]:
        return logits.tolist()

    def get_eval_scores_from_batch(self, *args, **kwargs): ...

    def train_end(self):
        self.exp_manager.log_metrics()
        self._save_best_model()
        self.exp_manager.aggregate_results()
        self.exp_manager.plot(
            groups=(
                [
                    [f"{clss}_f1" for clss in self.dev_dataset.label_set],
                    [f"{clss}_auc" for clss in self.dev_dataset.label_set],
                ]
                if self.do_eval
                else None
            )
        )

    def evaluation_metrics(
        self,
        eval_ids: list[Any],
        eval_true: list[list[int]],
        eval_preds: list[list[int]],
        eval_scores: list[list[float]],
        eval_losses: list[float],
        eval_reg_losses: list[float],
        data_loader: DataLoader | None = None,
    ) -> dict[str, dict[str, float]]:
        sep = self.any_dataset.any_dataset.id_separator

        annotator_info = {
            _id.split(sep)[1]: {
                "true": [],
                "pred": [],
            }
            for _id in eval_ids
        }
        for _id, true, pred in zip(
            eval_ids,
            eval_true or [None] * len(eval_ids),
            eval_preds or [None] * len(eval_ids),
        ):
            annotator_id = _id.split(sep)[1]
            annotator_info[annotator_id]["true"].append(true)
            annotator_info[annotator_id]["pred"].append(pred)

        results = {}

        for annotator_id, info in annotator_info.items():
            # zero division should 1 for all matching and constant labels and predictions
            # we set it to 0 because that is unlikely to happen
            macro_f1 = f1_score(
                info["true"], info["pred"], average="macro", zero_division=0
            )
            micro_f1 = f1_score(
                info["true"], info["pred"], average="micro", zero_division=0
            )

            js = jaccard_score(
                info["true"], info["pred"], average="samples", zero_division=1
            )

            f1_scores = f1_score(
                info["true"], info["pred"], average=None, zero_division=0
            )

            results[annotator_id] = {
                "jaccard_score": js,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
            } | {
                f"{clss}_f1": f1
                for clss, f1 in zip(data_loader.dataset.label_set, f1_scores)
            }

        return results

    def evaluate(self, *args, **kwargs):
        annotator_results, example_info = super().evaluate(*args, **kwargs)

        # annotator_results:
        #   per annotator results, IDs are annotator IDs or "aggregate"
        #   + some aggregate metrics
        # example_info:
        #   per example info IDs are going to be example_id - annotator_id
        #   the actual example results have annotator_id == "aggregate"

        aggregate_results = annotator_results.pop("aggregate")
        aggregate_results["eval_loss"] = annotator_results.pop("eval_loss")
        aggregate_results["eval_regularization_loss"] = annotator_results.pop(
            "eval_regularization_loss"
        )

        sep = self.any_dataset.any_dataset.id_separator

        aggregate_info = {}
        annotator_info = {}
        for k, v in example_info.items():
            example_id, annotator_id = k.split(sep)

            # log "out"s are tuples
            v["out"], v["residual_out"] = v["out"]
            # add text for easier debugging
            example = self.any_dataset.test_dataset.getitem_by_id(example_id)
            v["text"] = example["text"]
            for kk, vv in v.items():
                if "pred" in kk or "true" in kk:
                    v[kk] = self.any_dataset.any_dataset.index_label_set(vv)

            if annotator_id == "aggregate":
                aggregate_info[example_id] = v
            else:
                annotator_info.setdefault(annotator_id, {})[example_id] = v

        # we have to make numpy scalars into floats manually
        # because we are using manual logging
        annotator_results = {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in annotator_results.items()
        }
        self.exp_manager.set_custom_data(
            annotator_results, "annotator_metrics.yml"
        )

        self.exp_manager.set_custom_data(annotator_info, "annotator_preds.yml")

        return aggregate_results, aggregate_info


class APIPromptEvaluator(PromptEvaluator):
    def input_batch_kwargs(self, batch: dict[str, Any]) -> dict[str, Any]:
        return dict(
            user_prompt=batch["text"][0],
            system_prompt=self.test_dataset.system_prompt,
        )

    def train_end(self):
        self.exp_manager.set_custom_data(
            dict(
                completion_tokens=self.model.completion_tokens,
                prompt_tokens=self.model.prompt_tokens,
            ),
            "tokens.yml",
        )
        return super().train_end()
