import os
import re
import time
from subprocess import CalledProcessError
import traceback
from typing import List

import numpy as np
import sentencepiece as spm
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model_vllm import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures

from indextts.utils.front import TextNormalizer, TextTokenizer

import matplotlib.pyplot as plt


# def fade_in_out(wav, fade_in=int(24000*0.05), fade_out=int(24000*0.05)):
#     wav = wav.astype(np.float32)
#     print("wav", np.abs(wav).max(), np.abs(wav).mean(), np.abs(wav).min())
    
#     if fade_in > 0:
#         wav[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    
#     if fade_out > 0:
#         wav[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None]
    
#     wav = np.clip(wav, -32768, 32767).astype(np.int16)
#     wav = np.concatenate([np.zeros((int(0.4 * 24000), 1)), wav], axis=0).astype(np.int16)
#     return wav

def trim_and_pad_silence(wav_data, threshold=1000, min_silence=int(24000*0.4)):
    # # 1. 去除前端静音
    # abs_data = np.abs(wav_data).flatten()
    # first_non_silent = np.argmax(abs_data >= threshold)  # 第一个≥threshold的索引
    # wav_data = wav_data[max(0, first_non_silent-int(24000*0.1)):]  # 切片保留后端
    
    # 2. 处理后端静音
    abs_trimmed = np.abs(wav_data).flatten()
    last_non_silent = len(abs_trimmed) - np.argmax(abs_trimmed[::-1] >= threshold)  # 最后一个≥threshold的索引+1
    
    # 计算后端静音长度
    back_silence_length = len(wav_data) - last_non_silent
    if back_silence_length < min_silence:
        pad_length = min_silence - back_silence_length
        padded = np.vstack([wav_data, np.zeros((pad_length, 1))])  # 补0
    else:
        padded = wav_data
    
    return padded.astype(np.int16)


class IndexTTS:
    def __init__(
        self, model_dir="checkpoints", is_fp16=True, device=None, use_cuda_kernel=None, gpu_memory_utilization=0.25
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        cfg_path = os.path.join(model_dir, "config.yaml")
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        vllm_dir = os.path.join(model_dir, "gpt")
        engine_args = AsyncEngineArgs(
            model=vllm_dir,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            # enforce_eager=True,
        )
        indextts_vllm = AsyncLLM.from_engine_args(engine_args)

        self.gpt = UnifiedVoice(indextts_vllm, **self.cfg.gpt, model_dir=model_dir)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        # if self.is_fp16:
        #     self.gpt.eval().half()
        # else:
        #     self.gpt.eval()
        self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load

                anti_alias_activation_cuda = load.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except Exception as ex:
                traceback.print_exc()
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, "bpe.model")  # self.cfg.dataset["bpe_model"]
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)

        self.speaker_dict = {}
    
    def remove_long_silence(self, codes: list, latent: torch.Tensor, max_consecutive=15, silent_token=52):
        assert latent.dim() == 3 and latent.size(0) == 1, "Latent should be (1, seq_len, dim)"
        seq_len, dim = latent.size(1), latent.size(2)
        # print("latent", latent.shape)
        
        if self.stop_mel_token in codes:
            try:
                stop_idx = codes.index(self.stop_mel_token)
                valid_len = max(stop_idx - 1, 0)  # 保留至停止标记前一位
            except ValueError:
                valid_len = len(codes)
        else:
            valid_len = len(codes)
        
        valid_codes = codes[:min(valid_len, len(codes))]
        valid_latent = latent[0, :seq_len]  # 保持维度兼容性
        
        keep_indices = []
        silence_counter = 0
        
        for idx, token in enumerate(valid_codes):
            if token == silent_token:
                silence_counter += 1
            else:
                silence_counter = 0
            
            if silence_counter <= max_consecutive:
                keep_indices.append(idx)
        
        filtered_latent = valid_latent[keep_indices].unsqueeze(0)  # [1, new_seq, dim]
        # print("filtered_latent", filtered_latent.shape)
        return filtered_latent

    async def infer(self, audio_prompt: List[str], text, output_path=None, verbose=False, seed=None):
        print(">> start inference...")
        start_time = time.perf_counter()

        auto_conditioning = []
        for ap_ in audio_prompt:
            audio, sr = torchaudio.load(ap_)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            # cond_mel_frame = cond_mel.shape[-1]
            auto_conditioning.append(cond_mel)

        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list)
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        bigvgan_time = 0

        speech_conditioning_latent = []
        for cond_mel in auto_conditioning:
            speech_conditioning_latent_ = self.gpt.get_conditioning(
                cond_mel,  # .half()
                torch.tensor([cond_mel.shape[-1]], device=self.device)
            )
            speech_conditioning_latent.append(speech_conditioning_latent_)
        speech_conditioning_latent = torch.stack(speech_conditioning_latent).sum(dim=0)
        speech_conditioning_latent = speech_conditioning_latent / len(auto_conditioning)

        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                # 设置采样参数的seed
                if seed is not None:
                    self.gpt.sampling_params.seed = int(seed)
                else:
                    self.gpt.sampling_params.seed = None
                codes = await self.gpt.inference_speech(
                    speech_conditioning_latent,
                    text_tokens,
                    # cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device)
                )
                gpt_gen_time += time.perf_counter() - m_start_time

                # # remove ultra-long silence if exits
                # # temporarily fix the long silence bug.
                # latent = self.remove_long_silence(codes, latent)

                codes = torch.tensor(codes, dtype=torch.long, device=self.device).unsqueeze(0)
                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                latent = self.gpt(speech_conditioning_latent, text_tokens,
                                torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                code_lens*self.gpt.mel_length_compression,
                                cond_mel_lengths=torch.tensor([speech_conditioning_latent.shape[-1]], device=text_tokens.device),
                                return_latent=True, clip_inputs=False)

                m_start_time = time.perf_counter()
                wav, _ = self.bigvgan(latent, [ap_.transpose(1, 2) for ap_ in auto_conditioning])
                bigvgan_time += time.perf_counter() - m_start_time
                wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
        torch.cuda.empty_cache()
        end_time = time.perf_counter()

        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            wav_data = trim_and_pad_silence(wav_data)
            return (sampling_rate, wav_data)
        
    async def infer_with_ref_audio_embed(self, speaker: str, text):
        start_time = time.perf_counter()
        text = text.replace("嗯", "EN4")
        text = text.replace("嘿", "HEI1")
        text = text.replace("嗨", "HAI4")
        text = text.replace("哈哈", "HA1HA1")
        sampling_rate = 24000

        auto_conditioning = self.speaker_dict[speaker]["auto_conditioning"]

        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list)
        wavs = []
        gpt_gen_time = 0
        bigvgan_time = 0

        speech_conditioning_latent = self.speaker_dict[speaker]["speech_conditioning_latent"]

        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                codes = await self.gpt.inference_speech(
                    speech_conditioning_latent,
                    text_tokens,
                    # cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device)
                )
                gpt_gen_time += time.perf_counter() - m_start_time

                # # remove ultra-long silence if exits
                # # temporarily fix the long silence bug.
                # latent = self.remove_long_silence(codes, latent)

                codes = torch.tensor(codes, dtype=torch.long, device=self.device).unsqueeze(0)
                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                latent = self.gpt(speech_conditioning_latent, text_tokens,
                                torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                code_lens*self.gpt.mel_length_compression,
                                cond_mel_lengths=torch.tensor([speech_conditioning_latent.shape[-1]], device=text_tokens.device),
                                return_latent=True, clip_inputs=False)

                m_start_time = time.perf_counter()
                wav, _ = self.bigvgan(latent, [ap_.transpose(1, 2) for ap_ in auto_conditioning])
                bigvgan_time += time.perf_counter() - m_start_time
                wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                # wavs.append(wav[:, :-512])
                wavs.append(wav)  # to cpu before saving
        torch.cuda.empty_cache()
        end_time = time.perf_counter()

        wav = torch.cat(wavs, dim=1)
        # wav_length = wav.shape[-1] / sampling_rate
        # # print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        # print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        # print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        # print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        # print(f">> Generated audio length: {wav_length:.2f} seconds")
        # print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        wav_data = wav.type(torch.int16)
        wav_data = wav_data.numpy().T
        wav_data = trim_and_pad_silence(wav_data)
        return (sampling_rate, wav_data)
    
    @torch.no_grad()
    def registry_speaker(self, speaker: str, audio_paths: List[str]):
        auto_conditioning = []
        for ap_ in audio_paths:
            audio, sr = torchaudio.load(ap_)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            # cond_mel_frame = cond_mel.shape[-1]
            auto_conditioning.append(cond_mel)

        speech_conditioning_latent = []
        for cond_mel in auto_conditioning:
            speech_conditioning_latent_ = self.gpt.get_conditioning(
                cond_mel,  # .half()
                torch.tensor([cond_mel.shape[-1]], device=self.device)
            )
            speech_conditioning_latent.append(speech_conditioning_latent_)
        speech_conditioning_latent = torch.stack(speech_conditioning_latent).sum(dim=0)
        speech_conditioning_latent = speech_conditioning_latent / len(auto_conditioning)

        self.speaker_dict[speaker] = {
            "auto_conditioning": auto_conditioning,
            "speech_conditioning_latent": speech_conditioning_latent
        }
        print(f"Speaker: {speaker} registered")
