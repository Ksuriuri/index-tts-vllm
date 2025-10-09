<a href="README.md">中文</a> ｜ <a href="README_EN.md">English</a>

<div align="center">

# IndexTTS-vLLM
</div>

## Introduction
This project re-implements the inference of the GPT model using the vLLM library on top of [index-tts](https://github.com/index-tts/index-tts), accelerating the inference process of index-tts.

The inference speed improvement (Index-TTS-v1) on a single RTX 4090 is as follows:
- RTF (Real-Time Factor) for a single request: ≈0.3 -> ≈0.1
- GPT model decode speed for a single request: ≈90 tokens/s -> ≈280 tokens/s
- Concurrency: With `gpu_memory_utilization` set to 0.25 (about 5GB of VRAM), it was tested to handle a concurrency of around 16 without pressure (refer to `simple_test.py` for the speed test script).

## Update Log

- **[2025-08-07]** Added support for fully automated one-click deployment of the API service using Docker: `docker compose up`

- **[2025-08-06]** Added support for calling in the OpenAI API format:
    1. Added `/audio/speech` API path to be compatible with the OpenAI interface.
    2. Added `/audio/voices` API path to get the voice/character list.
    - Corresponds to: [createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

- **[2025-09-22]** Supported vLLM v1. Compatibility for IndexTTS2 is in progress.

- **[2025-09-28]** Supported webui inference for IndexTTS2 and organized the weight files, making deployment more convenient now! \0.0/ ; However, the current version does not seem to have an acceleration effect on the GPT of IndexTTS2, which is under investigation.

- **[2025-09-29]** Solved the issue of no acceleration for IndexTTS2. The reason was that `eos_token_id` was missed when making it compatible with v2. It can now accelerate normally.

## Usage Steps

### 1. Clone this project
```bash
git clone https://github.com/Ksuriuri/index-tts-vllm.git
cd index-tts-vllm
```

### 2. Create and activate a conda environment```bash
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm
```

### 3. Install PyTorch

PyTorch version 2.8.0 (corresponding to vllm 0.10.2) is required. Please refer to the [PyTorch official website](https://pytorch.org/get-started/locally/) for specific installation instructions.

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download model weights

(Recommended) Choose the corresponding version of the model weights and download them to the `checkpoints/` path:

```bash
# Index-TTS
modelscope download --model kusuriuri/Index-TTS-vLLM --local_dir ./checkpoints/Index-TTS-vLLM

# IndexTTS-1.5
modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ./checkpoints/Index-TTS-1.5-vLLM

# IndexTTS-2
modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM
```

(Optional, not recommended) You can also use `convert_hf_format.sh` to convert the official weight files yourself:

```bash
bash convert_hf_format.sh /path/to/your/model_dir
```

### 6. Start the webui!

Run the corresponding version:

```bash
# Index-TTS 1.0
python webui.py

# IndexTTS-1.5
python webui.py --version 1.5

# IndexTTS-2
python webui_v2.py
```
The first startup may take a while because it needs to compile the CUDA kernel for bigvgan.

## API

The API interface is encapsulated using FastAPI and **has been adapted for IndexTTS-2 model**.

The startup example is as follows. Please change `--model_dir` to the actual path of your IndexTTS-2 model:

```bash
python api_server.py --model_dir /your/path/to/IndexTTS-2-vLLM
```

### Startup Parameters
- `--model_dir`: Required, path to the IndexTTS-2 model weights.
- `--host`: Service IP address, defaults to `0.0.0.0`.
- `--port`: Service port, defaults to `6006`.
- `--gpu_memory_utilization`: vLLM VRAM usage rate, defaults to `0.25`.

### Documentation
- 📖 [API Usage Guide](API_USAGE.md) - Complete API documentation and examples
- 📖 [V1 to V2 Migration Guide](MIGRATION_V1_TO_V2.md) - Migration guide from IndexTTS-1.5 to IndexTTS-2

### Request Example
Refer to `api_example.py` or see [API_USAGE.md](API_USAGE.md)

### OpenAI API
- Added `/audio/speech` API path to be compatible with the OpenAI interface.
- Added `/audio/voices` API path to get the voice/character list.

For details, see: [createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## New Features
- **v1/v1.5:** Supports multi-character audio mixing: You can input multiple reference audios, and the voice of the TTS output will be a mixed version of the multiple reference audios (inputting multiple reference audios may cause the output voice to be unstable; you can try multiple times until you get a satisfactory voice to use as a reference audio).

## Performance
Word Error Rate (WER) Results for IndexTTS and Baseline Models on the [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| model                   | zh    | en    |
| ----------------------- | ----- | ----- |
| Human                   | 1.254 | 2.143 |
| index-tts (num_beams=3) | 1.005 | 1.943 |
| index-tts (num_beams=1) | 1.107 | 2.032 |
| index-tts-vllm          | 1.12  | 1.987 |

Basically maintains the performance of the original project.

## Concurrency Test
Refer to [`simple_test.py`](simple_test.py). You need to start the API service first.