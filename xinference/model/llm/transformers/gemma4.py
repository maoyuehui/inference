# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List, Tuple, Union

from ....core.model import register_batching_multimodal_models
from ...scheduler.request import InferenceRequest
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from .core import PytorchChatModel, register_non_default_model


@register_batching_multimodal_models("gemma-4")
@register_transformer
@register_non_default_model("Gemma4ForConditionalGeneration")
class Gemma4ChatModel(PytorchChatModel):
    GEMMA4_ARCHITECTURES = {"Gemma4ForConditionalGeneration"}

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        result = super().check_lib()
        if result is not True:
            return result

        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("5.5.0"):
            return False, "Gemma-4 requires transformers>=5.5.0"
        return True

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if model_spec.model_format not in ["pytorch", "gptq", "awq", "bnb", "fp4"]:
            return (
                False,
                "Gemma4 transformer supports pytorch/gptq/awq/bnb/fp4 formats only",
            )
        if not model_family.has_architecture(*cls.GEMMA4_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not Gemma-4-it",
            )
        return True

    def _load_model(self, **kwargs):
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
            padding_side="left",
        )
        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Gemma-4 tokenizer requires either a pad token or an EOS token"
                )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **kwargs,
        )
        self._processor = processor
        self._device = model.device
        return model, tokenizer

    def _get_full_prompt(self, messages: List[Dict], tools, generate_config: dict):
        return self._transform_messages(messages)

    def build_prefill_kwargs(
        self, prompts: List, req_list: List[InferenceRequest]
    ) -> Dict:
        inputs = self._processor.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        ).to(self._device)

        for i, r in enumerate(req_list):
            input_ids = inputs["input_ids"][i]
            if "attention_mask" in inputs:
                attention_mask = inputs["attention_mask"][i].bool()
                real_len = int(attention_mask.sum().item())
                r.padding_len = attention_mask.numel() - real_len
                r.extra_kwargs["attention_mask_seq_len"] = real_len
                r.prompt_tokens = input_ids[attention_mask].tolist()
            else:
                r.prompt_tokens = input_ids.tolist()
                r.padding_len = 0

        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        position_ids = self.build_prefill_position_ids(batch_size, seq_len, req_list)

        return {**inputs, "position_ids": position_ids}
