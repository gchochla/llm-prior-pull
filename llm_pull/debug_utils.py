import gc
import torch

from llm_pull.models import LMForGeneration
from llm_pull.datasets import SemEval2018Task1Ec
from llm_pull.prompt_dataset import PromptDataset


class TestPrompter:
    """Quick prompts for testing the model."""

    def __init__(
        self,
        model_name="meta-llama/Llama-2-7b-chat-hf",
        cache_dir="/project/shrikann_35/llm-shared/",
        ds_root_dir="/project/shrikann_35/chochlak/semeval2018task1/",
        instruction_prompt="Perform emotion classification in the following examples by selecting none, one, or multiple of the following emotions: {}",
        incontext_prompt="Input: {}\nEmotion(s): {}",
        query_prompt=None,
        device="cuda:0",
        model_dtype="float16",
        load_in_4bit=False,
        load_in_8bit=False,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.model_dtype = model_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

        print("Loading datasets...")
        train_ds = SemEval2018Task1Ec(
            language="English", root_dir=ds_root_dir, splits="train"
        )
        dev_ds = SemEval2018Task1Ec(
            language="English", root_dir=ds_root_dir, splits="dev"
        )
        self.ds = PromptDataset(
            train_dataset=train_ds,
            test_dataset=dev_ds,
            shot=5,
            instruction_prompt=instruction_prompt,
            incontext_prompt=incontext_prompt,
            query_prompt=query_prompt,
            model_name_or_path=model_name,
            cache_dir=cache_dir,
            sampling_strategy="random",
            device="cpu",
            retries=3,
            seed=42,
            sentence_model_name_or_path=None,
        )
        print("Loading model...")
        self.model = LMForGeneration(
            model_name_or_path=model_name,
            cache_dir=self.cache_dir,
            max_new_tokens=50,
            model_dtype=model_dtype,
            device=self.device,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            tokenizer=self.ds.get_tokenizer(),
        )

    def change_prompt(
        self,
        instruction_prompt=None,
        incontext_prompt=None,
        query_prompt=None,
        shot=None,
    ):
        instruction_prompt = instruction_prompt or self.ds.instruction_prompt
        incontext_prompt = incontext_prompt or self.ds.incontext_prompt
        query_prompt = query_prompt or self.ds.query_prompt
        shot = shot or self.ds.shot

        self.ds = PromptDataset(
            train_dataset=self.ds.train_dataset,
            test_dataset=self.ds.test_dataset,
            shot=shot,
            instruction_prompt=instruction_prompt,
            incontext_prompt=incontext_prompt,
            query_prompt=query_prompt,
            model_name_or_path=self.model_name,
            cache_dir=self.cache_dir,
            sampling_strategy="random",
            device="cpu",
            retries=3,
            seed=42,
            sentence_model_name_or_path=None,
        )

    def change_model(
        self,
        model_name=None,
        model_dtype=None,
        load_in_4bit=False,
        load_in_8bit=False,
    ):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        self.model_name = model_name or self.model_name
        self.model_dtype = model_dtype or self.model_dtype
        self.load_in_4bit = load_in_4bit or self.load_in_4bit
        self.load_in_8bit = load_in_8bit or self.load_in_8bit

        self.model = LMForGeneration(
            model_name_or_path=self.model_name,
            cache_dir=self.cache_dir,
            max_new_tokens=50,
            model_dtype=self.model_dtype,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            device=self.device,
            tokenizer=self.ds.get_tokenizer(),
        )
        self.ds = PromptDataset(
            train_dataset=self.ds.train_dataset,
            test_dataset=self.ds.test_dataset,
            shot=self.ds.shot,
            instruction_prompt=self.ds.instruction_prompt,
            incontext_prompt=self.ds.incontext_prompt,
            query_prompt=self.ds.query_prompt,
            model_name_or_path=model_name,
            cache_dir=self.cache_dir,
            sampling_strategy="random",
            device="cpu",
            retries=3,
            seed=42,
            sentence_model_name_or_path=None,
        )

    def forward(self, i=0):
        return self.model(
            **{
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in self.ds[i]["encoding"].items()
            }
        )

    def prompt(self, i=0):
        return dict(
            prompt=self.ds[i]["text"],
            generation=self.forward(i)["text"][0],
        )
