from typing import Any

from legm import from_namespace

from llm_pull.base_datasets import TokenizationMixin
from llm_pull.benchmarks import SemEval2018Task1Ec, GoEmotions


class SemEval2018Task1EcDataset(SemEval2018Task1Ec):
    """Plain text dataset for `SemEval 2018 Task 1: Affect in Tweets`
    (https://competitions.codalab.org/competitions/17751). Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `SemEval2018Task1Ec`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SemEval2018Task1EcDatasetForTransformers(
    TokenizationMixin, SemEval2018Task1Ec
):
    """Dataset with encodings for `transformers`
    for `SemEval 2018 Task 1: Affect in Tweets`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            SemEval2018Task1Ec.argparse_args()
            | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.data, self.preprocessor)


class GoEmotionsDataset(GoEmotions):
    """Plain text dataset for `GoEmotions`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `GoEmotions`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GoEmotionsDatasetForTransformers(TokenizationMixin, GoEmotions):
    """Dataset with encodings for `transformers`
    for `GoEmotions`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return GoEmotions.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.data, self.preprocessor)
