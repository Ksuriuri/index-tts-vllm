<a href="README.md">中文</a> ｜ <a href="README_EN.md">English</a>

<div align="center">

# IndexTTS-vLLM
</div>

## Introduction
This project re-implements the GPT model's inference from [index-tts](https://github.com/index-tts/index-tts) using the vllm library, accelerating the inference process of index-tts.

Inference speed improvement (Index-TTS-v1/v1.5) on a single RTX 4090:
- RTF (Real-Time Factor) for a single request: ≈0.3 -> ≈0.1
- GPT model decode speed for a single request: ≈90 tokens/s -> ≈280 tokens/s
- Concurrency: With `gpu_memory_utilization` set to 0.25 (approx. 5GB VRAM), it can handle a concurrency of around 16 without pressure (refer to `simple_test.py` for the benchmark script).

## Update Log

- **[2025-09-22]** Added support for vllm v1. Compatibility with IndexTTS2 is in progress.

- **[2025-09-28]** Supported web UI inference for IndexTTS2 and organized the weight files for easier deployment! \0.0/ ; However, the current version doesn't seem to accelerate the GPT of IndexTTS2, which is under investigation.

- **[2025-09-29]** Resolved the issue of ineffective GPT model inference acceleration for IndexTTS2.

- **[2025-10-09]** Compatible with IndexTTS2 API calls, please refer to [API](#api); APIs for v1/1.5 and the OpenAI-compatible interfaces may still have bugs, to be fixed later.

- **[2025-10-19]** Supported vllm inference for qwen0.6bemo4-merge.

- **[2026-03-03]** vllm 0.16.0 support for gpt2 inference

## TODO list
- Concurrency optimization for V2 API: Currently, only the gpt2 model inference is parallel, while other modules run serially. The s2mel inference has a large overhead (requiring 25 DiT iterations), which significantly impacts concurrency performance.

- Acceleration of s2mel inference.

## Usage Steps

### 1. Clone this project
```bash
git clone https://github.com/Ksuriuri/index-tts-vllm.git
cd index-tts-vllm
```


### 2. Create and activate a conda environment
```bash
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm
```


### 3. Install dependencies
Install dependencies with forced overrides to resolve the protobuf version conflict between vllm 0.16.0 and descript-audiotools 0.7.2.
```bash
pip install uv
uv pip install -r requirements.txt -c overrides.txt
```


### 4. Download model weights

#### Automatic Download (Recommended)

Download the corresponding version of the model weights to the `checkpoints/` directory:

**From ModelScope (recommended for users in China):**

```bash
# Index-TTS
modelscope download --model kusuriuri/Index-TTS-vLLM --local_dir ./checkpoints/Index-TTS-vLLM

# IndexTTS-1.5
modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ./checkpoints/Index-TTS-1.5-vLLM

# IndexTTS-2
modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM
```

**From Hugging Face:**

```bash
# IndexTTS-2
huggingface-cli download ksuriuri/IndexTTS-2-vLLM --local-dir ./checkpoints/IndexTTS-2-vLLM
```

#### Manual Download

- ModelScope: [Index-TTS](https://www.modelscope.cn/models/kusuriuri/Index-TTS-vLLM) | [IndexTTS-1.5](https://www.modelscope.cn/models/kusuriuri/Index-TTS-1.5-vLLM) | [IndexTTS-2](https://www.modelscope.cn/models/kusuriuri/IndexTTS-2-vLLM)
- Hugging Face: [IndexTTS-2](https://huggingface.co/ksuriuri/IndexTTS-2-vLLM)

#### Convert original weights yourself (Optional, not recommended)

You can use `convert_hf_format.sh` to convert the official weight files yourself:

```bash
bash convert_hf_format.sh /path/to/your/model_dir
```

### 5. Launch the web UI!

Run the corresponding version (the first launch may take longer due to CUDA kernel compilation for bigvgan):

```bash
# Index-TTS 1.0
python webui.py

# IndexTTS-1.5
python webui.py --version 1.5

# IndexTTS-2
python webui_v2.py
```


## API

An API interface is encapsulated using FastAPI. Here is an example of how to start it:

```bash
# Index-TTS-1.0/1.5
python api_server.py

# IndexTTS-2
python api_server_v2.py
```

### Startup Parameters
- `--model_dir`: Required, path to the model weights.
- `--host`: Server IP address, defaults to `0.0.0.0`.
- `--port`: Server port, defaults to `6006`.
- `--gpu_memory_utilization`: vllm GPU memory utilization rate, defaults to `0.25`.

### API Request Examples
- For v1/1.5, please refer to `api_example.py`.
- For v2, please refer to `api_example_v2.py`.

### OpenAI API
- Added `/audio/speech` API path for compatibility with the OpenAI interface.
- Added `/audio/voices` API path to get the list of voices/characters.

For details, see: [createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## New Features
- **v1/v1.5:** Supports multi-character audio mixing: You can input multiple reference audios, and the TTS output voice will be a mix of these reference audios.
- **IndexTTS2 (vLLM):** Inference Optimization & Emotion Reference
  - **Inference Optimization**: Added `precompute_speaker` and `precompute_emotion` interfaces to support feature precomputation and caching for reference audios. In batch synthesis or high-frequency calling scenarios, it avoids redundant feature extraction, significantly reducing first-token latency and RTF.
  - **Emotion Reference**: Allows explicitly passing an `emo_audio_prompt` during inference (via API or `infer` method) and supports dynamic emotion intensity adjustment through `emo_alpha`.
 (Inputting multiple reference audios may lead to an unstable output voice; you can try multiple times to get a satisfactory voice and then use it as a reference audio).

## Performance
Word Error Rate (WER) Results for IndexTTS and Baseline Models on the [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| model                   | zh    | en    |
| ----------------------- | ----- | ----- |
| Human                   | 1.254 | 2.143 |
| index-tts (num_beams=3) | 1.005 | 1.943 |
| index-tts (num_beams=1) | 1.107 | 2.032 |
| index-tts-vllm          | 1.12  | 1.987 |

Maintains the performance of the original project.

## Concurrency Test
Refer to [`simple_test.py`](simple_test.py). The API service must be started first.
