# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gpt2/modeling_gpt2.py
# Copyright 2023 The vLLM team.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GPT-2 model compatible with HuggingFace weights."""
from typing import (Final, Iterable, List, Literal, Mapping, Optional,
                    Protocol, Set, Tuple, TypedDict, TypeVar, Union)

import numpy as np
import torch
from torch import nn
# from transformers import GPT2Config
from transformers import BatchFeature

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.parallel_state import (
    get_pp_group, get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import get_act_fn
# from vllm.model_executor.layers.linear import (ColumnParallelLinear,
#                                                QKVParallelLinear,
#                                                RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
# from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, ProcessingCache,
                                        PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalInputs, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import MultiModalDataItems

from vllm.model_executor.models.gpt2 import GPT2Block  #, GPT2MLP, GPT2Attention

class TTSProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # return {"audio": 2048}
        return {"image": 2048}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # return {"audio": 2048}
        return {"image": 2048}

class TTSDummyInputsBuilder(BaseDummyInputsBuilder[TTSProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        # num_audios = mm_counts.get("audio", 0)
        return ProcessorInputs(
            prompt_text="", # "0" * num_audios,
            mm_data={
                # "audio": np.zeros((1, 1, 1, 3))
                # "audio": [torch.zeros((1, 1024))] * num_audios
                # "audio": torch.zeros((num_audios, 1024))
            },
        )

class TTSMultiModalProcessor(BaseMultiModalProcessor[TTSProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # print("prompt:", prompt)
        # print("mm_data:", mm_data)
        # print("mm_kwargs:", mm_kwargs)
        # print()
        processed_outputs = BatchFeature()
        processed_outputs["input_ids"] = np.array([[0] * len(prompt)])
        # # processed_outputs["audio_embeds"] = mm_data.get("audio", [torch.zeros((1, 1024))] * len(prompt))
        # processed_outputs["audio_embeds"] = mm_data.get("audio", torch.zeros((len(prompt), 1024)))
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # print("hf_inputs:", hf_inputs)
        # print("hf_processor_mm_kwargs:", hf_processor_mm_kwargs)
        # print()
        return dict(
            # audio_embeds=MultiModalFieldConfig.batched("audio"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        # print("mm_items:", mm_items)
        # print("hf_processor_mm_kwargs:", hf_processor_mm_kwargs)
        # print("out_mm_kwargs:", out_mm_kwargs)
        # print()

        def get_replacement(item_idx: int):
            tokens = [0]
            return tokens
        return [
            PromptReplacement(
                modality="image",
                target=[0],
                replacement=get_replacement,
            ),
        ]

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        # print("mm_items:", mm_items)
        # print("hf_processor_mm_kwargs:", hf_processor_mm_kwargs)
        # print("out_mm_kwargs:", out_mm_kwargs)
        # print()

        def get_replacement(item_idx: int):
            tokens = [0]
            return tokens
        return [
            PromptReplacement(
                modality="image",
                target=[0],
                replacement=get_replacement,
            ),
        ]


# @support_torch_compile
# class GPT2Model(nn.Module):

#     def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
#         super().__init__()

#         config = vllm_config.model_config.hf_config
#         cache_config = vllm_config.cache_config
#         quant_config = vllm_config.quant_config

#         self.config = config
#         assert not config.add_cross_attention
#         assert not config.scale_attn_by_inverse_layer_idx
#         assert not config.reorder_and_upcast_attn
#         self.embed_dim = config.n_embd
#         # self.wte = VocabParallelEmbedding(config.vocab_size,
#         #                                   self.embed_dim,
#         #                                   quant_config=quant_config,
#         #                                   prefix=f"{prefix}.wte")
#         # self.wpe = nn.Embedding(1, self.embed_dim)  # not used
#         self.start_layer, self.end_layer, self.h = make_layers(
#             config.n_layer,
#             lambda prefix: GPT2Block(
#                 config, cache_config, quant_config, prefix=prefix),
#             prefix=f"{prefix}.h")
#         self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
#         self.make_empty_intermediate_tensors = (
#             make_empty_intermediate_tensors_factory(["hidden_states"],
#                                                     config.n_embd))

#     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
#         # return self.wte(input_ids)
#         return torch.zeros((input_ids.shape[0], input_ids.shape[1], self.config.n_embd))

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         position_ids: torch.Tensor,
#         intermediate_tensors: Optional[IntermediateTensors],
#         inputs_embeds: Optional[torch.Tensor],
#     ) -> Union[torch.Tensor, IntermediateTensors]:
#         assert inputs_embeds is not None
#         # if get_pp_group().is_first_rank:
#         #     # position_embeds = self.wpe(position_ids)
#         #     hidden_states = inputs_embeds  # + position_embeds
#         # else:
#         #     assert intermediate_tensors is not None
#         #     hidden_states = intermediate_tensors["hidden_states"]
#         hidden_states = inputs_embeds

#         for i in range(self.start_layer, self.end_layer):
#             layer = self.h[i]
#             hidden_states = layer(hidden_states)

#         if not get_pp_group().is_last_rank:
#             return IntermediateTensors({"hidden_states": hidden_states})

#         hidden_states = self.ln_f(hidden_states)
#         return hidden_states
    
@support_torch_compile
class GPT2Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size
        # self.wte = VocabParallelEmbedding(config.vocab_size,
        #                                   self.embed_dim,
        #                                   quant_config=quant_config,
        #                                   prefix=f"{prefix}.wte")
        # self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: GPT2Block(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h")
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.n_embd))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # if get_pp_group().is_first_rank:
        #     if inputs_embeds is None:
        #         inputs_embeds = self.get_input_embeddings(input_ids)
        #     position_embeds = self.wpe(position_ids)
        #     hidden_states = inputs_embeds + position_embeds
        # else:
        #     assert intermediate_tensors is not None
        #     hidden_states = intermediate_tensors["hidden_states"]
        hidden_states = inputs_embeds

        for layer in self.h[self.start_layer:self.end_layer]:
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            # The HF's GPT-2 implementation uses Conv1D instead of Linear.
            # Because of this, we need to transpose the weights.
            # Note(zhuohan): the logic below might break quantized models.
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
    

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(ind).unsqueeze(0)  # torch.tensor([ind], device=dev)


# from vllm.model_executor.models.llava import (
#     _build_llava_or_pixtral_hf_processor,
#     _build_llava_or_pixtral_hf_info,
#     LlavaDummyInputsBuilder
# )
# @MULTIMODAL_REGISTRY.register_processor(_build_llava_or_pixtral_hf_processor,
#                                         info=_build_llava_or_pixtral_hf_info,
#                                         dummy_inputs=LlavaDummyInputsBuilder)
@MULTIMODAL_REGISTRY.register_processor(TTSMultiModalProcessor,
                                        info=TTSProcessingInfo,
                                        dummy_inputs=TTSDummyInputsBuilder)
class GPT2TTSModel(nn.Module, SupportsPP, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        

        self.transformer = GPT2Model(vllm_config=vllm_config,
                                     prefix=maybe_prefix(
                                         prefix, "transformer"))  # TODO: 参数适配
        self.text_pos_embedding = LearnedPositionEmbeddings(self.config.n_positions, self.config.n_embd)
        self.audio_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.final_norm = nn.LayerNorm(self.config.n_embd, bias=True)
        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.n_embd,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.lm_head",
                                      bias=True)
        # if self.config.tie_word_embeddings:
        #     self.lm_head = self.lm_head.tie_weights(self.transformer.wte)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        # return self.transformer.get_input_embeddings(input_ids)
        return torch.zeros((input_ids.shape[0], input_ids.shape[1], self.config.n_embd))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        audio_embeds = kwargs.pop("image_embeds", None)
        if audio_embeds is not None:
            if not isinstance(audio_embeds, list):
                audio_embeds = audio_embeds.squeeze(1).reshape(-1, audio_embeds.shape[-1])
            else:
                audio_embeds = torch.cat(audio_embeds, dim=1).squeeze(0)
            audio_embeds = audio_embeds.to(dtype=self.audio_emb.weight.dtype)

        if audio_embeds is not None:  # and audio_embeds.shape[0] == input_ids.shape[0]   prefill
            inputs_embeds = audio_embeds
        else:  # decode
            inputs_embeds = self.audio_emb(input_ids)
            # print("positions", positions)
            if torch.cuda.is_current_stream_capturing() or positions[0] > 0:
                pos_emb = [self.text_pos_embedding.get_fixed_embedding(ind, inputs_embeds.device)
                            for ind in positions]  #  - mel_len + 2
                pos_emb = torch.cat(pos_emb, dim=0)
                inputs_embeds = inputs_embeds + pos_emb
        # print("inputs_embeds", inputs_embeds.shape)
        hidden_states = self.transformer(input_ids, positions, intermediate_tensors,
                                         inputs_embeds)  # input_ids no used
        hidden_states = self.final_norm(hidden_states)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue
            # if not name.startswith("transformer.") and not name.startswith(
            #         "lm_head"):
            #     name = "transformer." + name

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            # The HF's GPT-2 implementation uses Conv1D instead of Linear.
            # Because of this, we need to transpose the weights.
            # Note(zhuohan): the logic below might break quantized models.
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
