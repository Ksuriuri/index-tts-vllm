import torch
import time
from typing import Any, List, Optional, Tuple, Union

from packaging import version
import importlib

vllm_version = version.parse(importlib.import_module("vllm").__version__)

# 在 vllm 中注册自定义的 GPT2TTSModel
from vllm import ModelRegistry
from indextts.gpt.index_tts_gpt2_vllm_v1 import GPT2TTSModel

ModelRegistry.register_model("GPT2InferenceModel", GPT2TTSModel)
print("✅  Registry GPT2TTSModel to vllm")


# 将 position_ids 减去 prefill 的长度再加 1，以便正确计算每一步 decode 的 position embedding
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
import numpy as np
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder


def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
    num_scheduled_tokens: np.ndarray,
) -> tuple[
    torch.Tensor,
    SpecDecodeMetadata | None,
]:
    """
    :return: tuple[
        logits_indices, spec_decode_metadata,
    ]
    """
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit_block_table(num_reqs)

    # Get request indices.
    # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

    # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
    # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

    # Get positions.
    positions_np = self.positions.np[:total_num_scheduled_tokens]
    np.add(
        self.input_batch.num_computed_tokens_cpu[req_indices],
        arange,
        out=positions_np,
    )

    # Calculate M-RoPE positions.
    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        self._calc_mrope_positions(scheduler_output)

    # Calculate XD-RoPE positions.
    # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
    if self.uses_xdrope_dim > 0:
        self._calc_xdrope_positions(scheduler_output)

    # Get token indices.
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
    # where M is the max_model_len.
    token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    token_indices_tensor = torch.from_numpy(token_indices)

    # NOTE(woosuk): We use torch.index_select instead of np.take here
    # because torch.index_select is much faster than np.take for large
    # tensors.
    torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        token_indices_tensor,
        out=self.input_ids.cpu[:total_num_scheduled_tokens],
    )
    if self.enable_prompt_embeds:
        is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
        torch.index_select(
            is_token_ids,
            0,
            token_indices_tensor,
            out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
        )

    # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
    # the InputBatch, we need to fill in the prompt embeds into the expected
    # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
    if self.input_batch.req_prompt_embeds:
        output_idx = 0
        for req_idx in range(num_reqs):
            num_sched = num_scheduled_tokens[req_idx]

            # Skip if this request doesn't have embeddings
            if req_idx not in self.input_batch.req_prompt_embeds:
                output_idx += num_sched
                continue

            # Skip if no tokens scheduled
            if num_sched <= 0:
                output_idx += num_sched
                continue

            req_embeds = self.input_batch.req_prompt_embeds[req_idx]
            start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

            # Skip if trying to read beyond available embeddings
            if start_pos >= req_embeds.shape[0]:
                output_idx += num_sched
                continue

            # Copy available embeddings
            end_pos = start_pos + num_sched
            actual_end = min(end_pos, req_embeds.shape[0])
            actual_num_sched = actual_end - start_pos

            if actual_num_sched > 0:
                self.inputs_embeds.cpu[
                    output_idx : output_idx + actual_num_sched
                ].copy_(req_embeds[start_pos:actual_end])

            output_idx += num_sched

    self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
    self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

    # GPT2TTSModel position ids support
    model = self.get_model()
    if isinstance(model, GPT2TTSModel):
        # req_ids_in_batch = self.input_batch.req_ids[:num_reqs]
        prompt_tokens_offset = []
        for req_id in self.input_batch.req_ids:
            prompt_tokens_offset.append(-(len(self.requests[req_id].prompt_token_ids) - 1))
            # print(f"[{idx}] self.requests[req_id].prompt_token_ids:", len(self.requests[req_id].prompt_token_ids), positions_np)
        np.add(np.array(prompt_tokens_offset)[req_indices],
                positions_np,
                out=positions_np)

    # Prepare the attention metadata.
    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
    # Note: pad query_start_loc to be non-decreasing, as kernels
    # like FlashAttention requires that
    self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
    self.query_start_loc.copy_to_gpu()
    query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

    self.seq_lens.np[:num_reqs] = (
        self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
    )
    # Fill unused with 0 for full cuda graph mode.
    self.seq_lens.np[num_reqs:].fill(0)
    self.seq_lens.copy_to_gpu()

    num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
    num_tokens_np = np.array(num_tokens, dtype=np.int32)

    # Record which requests should not be sampled,
    # so that we could clear the sampled tokens before returning
    self.discard_request_mask.np[:num_reqs] = (
        self.seq_lens.np[:num_reqs] < num_tokens_np
    )
    self.discard_request_mask.copy_to_gpu(num_reqs)

    # Copy the tensors to the GPU.
    self._prepare_input_ids(
        scheduler_output,
        total_num_scheduled_tokens,
        cu_num_tokens,
    )

    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
            self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
            non_blocking=True,
        )
    elif self.uses_xdrope_dim > 0:
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
            self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
            non_blocking=True,
        )
    else:
        # Common case (1D positions)
        self.positions.copy_to_gpu(total_num_scheduled_tokens)

    use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        spec_decode_metadata = None
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
    else:
        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        # For chunked prefills, use -1 as mask rather than 0, as guided
        # decoding may rollback speculative tokens.
        num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
        for (
            req_id,
            draft_token_ids,
        ) in scheduler_output.scheduled_spec_decode_tokens.items():
            req_idx = self.input_batch.req_id_to_index[req_id]
            num_draft_tokens[req_idx] = len(draft_token_ids)
            if (
                self.input_batch.num_computed_tokens_cpu[req_idx]
                >= self.input_batch.num_prompt_tokens[req_idx]
            ):
                num_decode_draft_tokens[req_idx] = len(draft_token_ids)
        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens, cu_num_tokens
        )
        logits_indices = spec_decode_metadata.logits_indices
        num_sampled_tokens = num_draft_tokens + 1
        # For DECODE only cuda graph of some attention backends (e.g., GDN).
        self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
        self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
        self.num_decode_draft_tokens.copy_to_gpu()

    # Hot-Swap lora model
    if self.lora_config:
        assert (
            np.sum(num_sampled_tokens)
            <= self.vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.set_active_loras(
            self.input_batch, num_scheduled_tokens, num_sampled_tokens
        )

    return (
        logits_indices,
        spec_decode_metadata,
    )


GPUModelRunner._prepare_inputs = _prepare_inputs
print("✅  GPUModelRunner._prepare_inputs Patched")
