# API Server V1 到 V2 迁移指南

## 主要变化

### 1. 模型类改变

**V1:**
```python
from indextts.infer_vllm import IndexTTS
tts = IndexTTS(model_dir=model_dir)
```

**V2:**
```python
from indextts.infer_vllm_v2 import IndexTTS2
tts = IndexTTS2(model_dir=model_dir)
```

### 2. API 接口变化

#### `/tts_url` 端点

**V1 请求格式:**
```json
{
  "text": "你好",
  "audio_paths": ["/path/to/audio.wav"],
  "seed": 8
}
```

**V2 请求格式:**
```json
{
  "text": "你好",
  "audio_path": "/path/to/audio.wav"
}
```

**变化说明:**
- `audio_paths` (数组) 改为 `audio_path` (字符串)
- 为保持兼容性，仍然支持 `audio_paths`，会自动取第一个元素
- 移除了 `seed` 参数

#### `/tts` 和 `/audio/speech` 端点

这两个端点的接口格式没有变化，但内部实现已更新为使用 IndexTTS2。

### 3. Speaker 注册机制变化

**V1:**
- `IndexTTS` 类内部有 `registry_speaker()` 方法
- 可以预注册多个参考音频

**V2:**
- `IndexTTS2` 类没有 `registry_speaker()` 方法
- API server 在内部维护 speaker 到音频路径的映射
- 每个 speaker 只使用第一个参考音频

### 4. 推理方法签名变化

**V1:**
```python
sr, wav = await tts.infer(audio_paths, text, seed=seed)
sr, wav = await tts.infer_with_ref_audio_embed(character, text)
```

**V2:**
```python
sr, wav = await tts.infer(
    spk_audio_prompt=audio_path,
    text=text,
    output_path=None,
    emo_audio_prompt=None,
    emo_alpha=1.0,
    ...
)
```

## 兼容性说明

### 向后兼容的部分

1. ✅ `/tts_url` 仍然接受 `audio_paths` 参数
2. ✅ `/tts` 和 `/audio/speech` 的请求格式不变
3. ✅ `/health` 和 `/audio/voices` 端点不变
4. ✅ 响应格式保持一致（WAV格式音频）

### 不兼容的部分

1. ❌ 移除了 `seed` 参数控制
2. ❌ 每个 speaker 只使用第一个参考音频（V1可以使用多个）
3. ❌ 需要使用 IndexTTS-2 模型，不能使用 IndexTTS-1.5 模型

## 模型文件要求

### V1 所需文件
```
model_dir/
├── bpe.model
├── gpt.pth
├── config.yaml
└── bigvgan.pth
```

### V2 所需文件
```
model_dir/
├── bpe.model
├── gpt.pth
├── config.yaml
├── s2mel.pth
├── wav2vec2bert_stats.pt
├── w2v-bert-2.0/
├── semantic_codec/
├── campplus/
├── bigvgan/
└── qwen2.5-0.5b-instruct/
```

## 迁移步骤

### 1. 下载 IndexTTS-2 模型

```bash
# 从 ModelScope 下载
modelscope download --model IndexTeam/IndexTTS-2

# 转换为 vLLM 格式
bash convert_hf_format.sh ~/.cache/modelscope/hub/models/IndexTeam/IndexTTS-2
```

### 2. 更新启动命令

**V1:**
```bash
python api_server.py --model_dir /path/to/IndexTTS-1.5-vLLM
```

**V2:**
```bash
python api_server.py --model_dir /path/to/IndexTTS-2-vLLM
```

### 3. 更新客户端代码（如果使用 `seed` 参数）

**V1:**
```python
response = requests.post(
    "http://localhost:6006/tts_url",
    json={
        "text": "你好",
        "audio_paths": ["/path/to/audio.wav"],
        "seed": 42
    }
)
```

**V2:**
```python
response = requests.post(
    "http://localhost:6006/tts_url",
    json={
        "text": "你好",
        "audio_path": "/path/to/audio.wav"
        # seed 参数已移除
    }
)
```

### 4. 验证迁移

测试健康检查：
```bash
curl http://localhost:6006/health
```

测试生成：
```bash
curl -X POST "http://localhost:6006/tts_url" \
  -H "Content-Type: application/json" \
  -d '{"text": "测试", "audio_path": "/path/to/audio.wav"}' \
  --output test.wav
```

## 新功能（V2）

IndexTTS-2 带来的新功能（虽然当前 API server 暂未完全暴露）：

1. **情感控制**
   - 支持通过情感参考音频控制
   - 支持通过情感向量控制
   - 支持通过文本描述控制情感

2. **更好的音质**
   - 使用新的 s2mel 模块
   - 改进的语音合成质量

3. **更灵活的配置**
   - 可调节的分句参数
   - 更多的生成参数

## 未来计划

考虑在 API server 中增加以下端点来充分利用 V2 的新功能：

```python
@app.post("/tts_emotion")
async def tts_with_emotion(request: Request):
    """使用情感控制的 TTS 接口"""
    data = await request.json()
    text = data["text"]
    audio_path = data["audio_path"]
    emo_audio = data.get("emo_audio")  # 情感参考音频
    emo_weight = data.get("emo_weight", 0.8)  # 情感权重
    # ...
```

## 问题排查

### 1. 启动失败：找不到模型文件

**错误信息:**
```
FileNotFoundError: [Errno 2] No such file or directory: '.../s2mel.pth'
```

**解决方案:**
确保使用的是 IndexTTS-2 模型，并且所有必需文件都已下载。

### 2. 推理错误：参数不匹配

**错误信息:**
```
TypeError: infer() got an unexpected keyword argument 'seed'
```

**解决方案:**
更新客户端代码，移除 `seed` 参数。

### 3. Speaker 未找到

**错误信息:**
```json
{"status": "error", "error": "Speaker 'xxx' not found"}
```

**解决方案:**
检查 `assets/speaker.json` 文件，确保 speaker 已正确配置。

## 获取帮助

如有问题，请访问：
- GitHub Issues: https://github.com/Ksuriuri/index-tts-vllm/issues
- 相关 Issue: #119, #107

