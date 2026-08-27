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

import torch

from .....core.model import XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
from ....scheduler.request import InferenceRequest
from ...llm_family import LLMFamilyV2, PytorchLLMSpecV2
from ..core import NON_DEFAULT_MODEL_LIST
from ..gemma4 import Gemma4ChatModel


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) for k, v in self.items()})


class _Processor:
    def apply_chat_template(self, prompts, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["return_dict"] is True
        assert kwargs["padding"] is True
        assert prompts == [["longer"], ["shorter"]]
        return _Batch(
            {
                "input_ids": torch.tensor([[1, 2, 3, 4], [0, 0, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]]),
            }
        )


def _request():
    return InferenceRequest([], None, True, "chat", None)


def _family(architectures):
    spec = PytorchLLMSpecV2(
        model_format="pytorch",
        model_size_in_billions=4,
        quantization="none",
        model_id="google/gemma-4-E4B-it",
        model_revision=None,
    )
    return LLMFamilyV2(
        version=2,
        context_length=131072,
        model_type="LLM",
        model_name="gemma-4",
        model_lang=["en"],
        model_ability=["chat", "vision"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
        architectures=architectures,
    )


def test_gemma4_registers_transformers_and_batching_support():
    assert "Gemma4ForConditionalGeneration" in NON_DEFAULT_MODEL_LIST
    assert "gemma-4" in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS


def test_gemma4_match_json_accepts_gemma4_conditional_generation():
    family = _family(["Gemma4ForConditionalGeneration"])
    spec = family.model_specs[0]

    assert Gemma4ChatModel.match_json(family, spec, "none") is True


def test_gemma4_prefill_uses_attention_mask_for_left_padding():
    model = Gemma4ChatModel.__new__(Gemma4ChatModel)
    model._processor = _Processor()
    model._device = torch.device("cpu")

    first = _request()
    second = _request()

    kwargs = model.build_prefill_kwargs([["longer"], ["shorter"]], [first, second])

    assert kwargs["input_ids"].tolist() == [[1, 2, 3, 4], [0, 0, 5, 6]]
    assert kwargs["attention_mask"].tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
    assert kwargs["position_ids"].tolist() == [[0, 1, 2, 3], [0, 0, 0, 1]]

    assert first.prompt_tokens == [1, 2, 3, 4]
    assert first.padding_len == 0
    assert first.extra_kwargs["attention_mask_seq_len"] == 4
    assert first.extra_kwargs["max_position_id"] == 3

    assert second.prompt_tokens == [5, 6]
    assert second.padding_len == 2
    assert second.extra_kwargs["attention_mask_seq_len"] == 2
    assert second.extra_kwargs["max_position_id"] == 1
