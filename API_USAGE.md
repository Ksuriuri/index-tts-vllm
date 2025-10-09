# IndexTTS-2 API Server 使用说明

## 启动服务

```bash
python api_server.py --model_dir /path/to/IndexTTS-2-vLLM --port 6006
```

可选参数：
- `--host`: 服务监听地址，默认 `0.0.0.0`
- `--port`: 服务端口，默认 `6006`
- `--model_dir`: 模型目录路径
- `--gpu_memory_utilization`: GPU内存使用率，默认 `0.25`

## API 端点

### 1. 健康检查

**GET** `/health`

返回服务健康状态。

**响应示例：**
```json
{
  "status": "healthy",
  "message": "Service is running",
  "timestamp": 1234567890.123
}
```

### 2. 使用音频路径生成语音

**POST** `/tts_url`

**请求体：**
```json
{
  "text": "你好，这是一个测试。",
  "audio_path": "/path/to/reference/audio.wav"
}
```

或者（向后兼容）：
```json
{
  "text": "你好，这是一个测试。",
  "audio_paths": ["/path/to/reference/audio.wav"]
}
```

**响应：** 返回 WAV 格式的音频文件

**curl 示例：**
```bash
curl -X POST "http://localhost:6006/tts_url" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "audio_path": "/path/to/audio.wav"}' \
  --output output.wav
```

### 3. 使用预设speaker生成语音

**POST** `/tts`

**请求体：**
```json
{
  "text": "你好，这是一个测试。",
  "character": "speaker_name"
}
```

**说明：** `character` 必须是在 `assets/speaker.json` 中预定义的speaker名称。

**响应：** 返回 WAV 格式的音频文件

**curl 示例：**
```bash
curl -X POST "http://localhost:6006/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "character": "default"}' \
  --output output.wav
```

### 4. OpenAI 兼容 API

**POST** `/audio/speech`

遵循 OpenAI 的 TTS API 格式。

**请求体：**
```json
{
  "model": "tts-1",
  "input": "你好，这是一个测试。",
  "voice": "speaker_name"
}
```

**响应：** 返回 WAV 格式的音频文件

**curl 示例：**
```bash
curl -X POST "http://localhost:6006/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "你好世界",
    "voice": "default"
  }' \
  --output output.wav
```

### 5. 获取可用voices列表

**GET** `/audio/voices`

返回所有可用的speaker列表。

**响应示例：**
```json
{
  "speaker1": ["assets/audio1.wav"],
  "speaker2": ["assets/audio2.wav"]
}
```

## Speaker 配置

在 `assets/speaker.json` 中配置预设的speaker：

```json
{
  "default": ["assets/default_voice.wav"],
  "female1": ["assets/female1.wav"],
  "male1": ["assets/male1.wav"]
}
```

服务启动时会自动加载这些配置。

## 错误处理

所有错误都会返回 JSON 格式的错误信息：

```json
{
  "status": "error",
  "error": "错误详细信息"
}
```

常见错误码：
- `400`: 请求参数错误
- `404`: Speaker 不存在
- `500`: 服务器内部错误
- `503`: 服务不可用

## Python 客户端示例

```python
import requests

# 使用音频路径
response = requests.post(
    "http://localhost:6006/tts_url",
    json={
        "text": "你好世界",
        "audio_path": "/path/to/reference.wav"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)

# 使用预设speaker
response = requests.post(
    "http://localhost:6006/tts",
    json={
        "text": "你好世界",
        "character": "default"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## 注意事项

1. 本版本使用 IndexTTS-2 模型，与 IndexTTS-1.5 不兼容
2. 确保模型目录包含所有必需的文件：
   - `bpe.model`
   - `gpt.pth`
   - `config.yaml`
   - `s2mel.pth`
   - `wav2vec2bert_stats.pt`
   - 以及其他依赖文件
3. 参考音频建议使用清晰、无噪音的音频片段
4. 生成的音频采样率为 22050 Hz

