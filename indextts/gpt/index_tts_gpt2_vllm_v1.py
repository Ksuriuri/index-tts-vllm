from typing import (Any, Dict, Final, Iterable, List, Literal, Mapping, Optional,
                    Protocol, Set, Tuple, TypedDict, TypeVar, Union, Sequence)

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed.parallel_state import (
    get_pp_group, get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
# from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
# from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsPP

from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix,
    _merge_multimodal_embeddings
)

from vllm.model_executor.models.gpt2 import GPT2Block  #, GPT2MLP, GPT2Attention

from vllm.model_executor.models.interfaces import SupportsMultiModal, MultiModalEmbeddings
from vllm.multimodal import MULTIMODAL_REGISTRY, ModalityData
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.processing import (BaseMultiModalProcessor, PromptReplacement,
                                        BaseProcessingInfo, PromptInsertion,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.parse import (AudioItem, MultiModalDataParser, DictEmbeddingItems,
                                   ModalityDataItems, MultiModalDataItems)
# from vllm.model_executor.models.utils import merge_multimodal_embeddings

PLACEHOLDER_TOKEN = "!"
PLACEHOLDER_TOKEN_ID = 0


class GPT2TTSProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # 声明我们支持 'audio' 模态
        return {"audio": None}
    
    def get_data_parser(self) -> MultiModalDataParser:
        return GPT2TTSDataParser()

class GPT2TTSDummyInputsBuilder(BaseDummyInputsBuilder[GPT2TTSProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return PLACEHOLDER_TOKEN * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> Dict[str, Any]:
        num_items = mm_counts.get("audio", 0)
        if num_items == 0:
            return {}
        
        config = self.info.get_hf_config()
        dummy_seq_len = 1024
        dummy_embed = torch.rand(
            (dummy_seq_len, config.n_embd),
            dtype=torch.float16,
        )
        
        return {"audio": {"audio_embeds": [dummy_embed] * num_items}}

class GPT2TTSDataParser(MultiModalDataParser):
    """
    这个解析器重写了处理 'audio' 模态的方法。
    """
    def _parse_audio_data(
        self,
        data: ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        """
        当 vLLM 看到 "audio" 这个 key 时，会调用这个函数。
        'data' 参数是 "audio" key 对应的值。
        """
        # 期望的值是一个字典，例如 {"audio_embeds": tensor}
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                # 这个工厂函数告诉 vLLM 如何将字典里的键映射到模型 forward 函数的参数
                # 这里将 "audio_embeds" 映射到名为 "audio_embeds" 的参数
                fields_factory=lambda hf_inputs: dict(
                    audio_embeds=MultiModalFieldConfig.batched("audio")
                ),
            )
        
        # 如果传入了 "audio" 但不是期望的字典格式，就报错
        raise TypeError(f"For 'audio' modality, expected a dict like {'{'} 'audio_embeds': tensor {'}'}, but got {type(data)}")

class GPT2TTSMultiModalProcessor(BaseMultiModalProcessor[GPT2TTSProcessingInfo]):
    
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_embeds=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: "MultiModalDataItems",
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> List[PromptUpdate]:
        out_mm_data = out_mm_kwargs.get_data()
        
        def get_replacement(item_idx: int):
            # 从处理过的数据中根据 'audio_embeds' 键获取 embedding
            embeds = out_mm_data["audio_embeds"][item_idx]
            num_features = embeds.shape[0]  # 获取序列长度
            
            # 创建一个假的 token 序列，长度必须正确
            return PromptUpdateDetails.select_token_id(
                [PLACEHOLDER_TOKEN_ID] * num_features, PLACEHOLDER_TOKEN_ID
            )

        return [
            PromptReplacement(
                modality="audio",
                target=PLACEHOLDER_TOKEN,  # [PLACEHOLDER_TOKEN_ID],
                replacement=get_replacement,
            )
        ]

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
        self.embed_dim = config.n_embd
        # self.wte = VocabParallelEmbedding(config.vocab_size,
        #                                   self.embed_dim,
        #                                   quant_config=quant_config,
        #                                   prefix=f"{prefix}.wte")
        # self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.start_layer, self.end_layer, self.h = make_layers(
            config.n_layer,
            lambda prefix: GPT2Block(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h")
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.n_embd))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
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
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


@MULTIMODAL_REGISTRY.register_processor(GPT2TTSMultiModalProcessor,
                                        info=GPT2TTSProcessingInfo,
                                        dummy_inputs=GPT2TTSDummyInputsBuilder)
class GPT2TTSModel(nn.Module, SupportsPP, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        
        self.transformer = GPT2Model(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "transformer"))
        self.text_pos_embedding = LearnedPositionEmbeddings(self.config.n_positions, self.config.n_embd)
        with torch.no_grad():
            self.text_pos_embedding.emb.weight[0].zero_()
        self.audio_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.final_norm = nn.LayerNorm(self.config.n_embd, bias=True)
        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.n_embd,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.lm_head",
                                      bias=True)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        # self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors)

    # 实现 SupportsMultiModal 接口方法
    def get_language_model(self) -> torch.nn.Module:
        return self.transformer

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings | None:
        # 从 kwargs 中提取我们的 embedding
        audio_embeds = kwargs.get("audio_embeds")

        processed_embeds = []
        for embed in audio_embeds:
            # 检查是否是多余的维度为1的3D张量
            if embed.dim() == 3 and embed.shape[0] == 1:
                # 移除多余的批次维度，使其变为 2D
                processed_embeds.append(embed.squeeze(0))
            elif embed.dim() == 2:
                # 如果已经是 2D 张量，直接添加
                processed_embeds.append(embed)
            else:
                # 对于非预期的维度，可以抛出错误以便调试
                raise ValueError(
                    "Expected audio embeddings to be 2D or 3D with a "
                    f"leading dimension of 1, but got shape: {embed.shape}")

        return processed_embeds

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # # 这个方法现在用于合并文本和多模态 embedding
        # # 在我们的 prefill 场景下，input_ids 是假的，我们只关心 multimodal_embeddings
        # if multimodal_embeddings is not None:  #  and len(multimodal_embeddings) > 0
        #     # 假设只有一个多模态输入，并且它就是我们想要的完整 embedding
        #     # 如果有多个，需要将它们拼接起来
        #     # 注意：vLLM 的 merge_multimodal_embeddings 是用于替换占位符 token 的，
        #     # 而我们的场景是整个输入都是 embedding，所以我们直接返回它。
        #     return torch.cat(multimodal_embeddings, dim=0)

        # # 对于 decode 阶段，我们走正常的 embedding lookup
        # return self.audio_emb(input_ids)
        inputs_embeds = self.audio_emb(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds=inputs_embeds, 
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=input_ids == PLACEHOLDER_TOKEN_ID
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # **kwargs 用于接收 get_multimodal_embeddings 的数据
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # assert inputs_embeds is not None
        positions = torch.clamp(positions, min=0)
        pos_emb = self.text_pos_embedding.emb(positions)

        # kusuriuri: 这里必须使用 += ，否则计算结果会错误
        inputs_embeds += pos_emb
        
        transformer_output = self.transformer(
            input_ids=None,
            position_ids=positions, 
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds
        )
        
        # if get_pp_group().is_last_rank:
        transformer_output = self.final_norm(transformer_output)
            
        return transformer_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    # def sample(
    #     self,
    #     logits: torch.Tensor,
    #     sampling_metadata: SamplingMetadata,
    # ) -> Optional[SamplerOutput]:
    #     next_tokens = self.sampler(logits, sampling_metadata)
    #     return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue
            if ".wte" in name:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            # try:
            weight_loader(param, loaded_weight)
            # except:
            #     print("weight_loader", name)
            #     raise AssertionError()
            loaded_params.add(name)
        
        # 确保在加载权重后，第0个位置的embedding仍然是全零向量。
        with torch.no_grad():
            self.text_pos_embedding.emb.weight[0].zero_()
        return loaded_params