# API Server V2 适配更新日志

## 更新日期
2025-10-09

## 更新内容

### 核心修改

#### 1. `api_server.py` - 主要更新
- ✅ 将 `IndexTTS` 替换为 `IndexTTS2`
- ✅ 添加 `speaker_audio_dict` 全局字典维护speaker到音频路径的映射
- ✅ 更新所有API端点以使用 `IndexTTS2.infer()` 方法
- ✅ 保持向后兼容性（`/tts_url` 仍支持 `audio_paths` 参数）
- ✅ 添加错误处理和验证

**主要变化：**
```python
# V1
from indextts.infer_vllm import IndexTTS
tts = IndexTTS(model_dir=args.model_dir)
sr, wav = await tts.infer(audio_paths, text, seed=seed)

# V2
from indextts.infer_vllm_v2 import IndexTTS2
tts = IndexTTS2(model_dir=args.model_dir)
sr, wav = await tts.infer(
    spk_audio_prompt=audio_path,
    text=text,
    output_path=None
)
```

#### 2. 新增文档

**API_USAGE.md**
- 完整的API使用说明
- 所有端点的详细文档
- curl和Python客户端示例
- 错误处理说明

**MIGRATION_V1_TO_V2.md**
- V1到V2的详细迁移指南
- 兼容性说明
- 常见问题排查
- 迁移步骤

#### 3. 更新示例代码

**api_example.py**
- 更新为 IndexTTS-2 版本
- 添加5个完整示例：
  1. 使用音频路径生成
  2. 使用预设speaker生成
  3. OpenAI兼容API
  4. 获取voices列表
  5. 健康检查
- 添加错误处理和友好的输出

#### 4. 更新README

**README.md & README_EN.md**
- 明确标注已适配 IndexTTS-2
- 更新模型路径示例
- 添加文档链接
- 更新参数说明

## API 端点变化

### `/tts_url` - 使用音频路径生成

**V1 请求：**
```json
{
  "text": "测试文本",
  "audio_paths": ["/path/to/audio.wav"],
  "seed": 8
}
```

**V2 请求（推荐）：**
```json
{
  "text": "测试文本",
  "audio_path": "/path/to/audio.wav"
}
```

**V2 请求（兼容）：**
```json
{
  "text": "测试文本",
  "audio_paths": ["/path/to/audio.wav"]
}
```

### `/tts` - 使用speaker生成

**接口格式不变，但内部实现已更新为 IndexTTS2**

### `/audio/speech` - OpenAI兼容

**接口格式不变，但内部实现已更新为 IndexTTS2**

### `/health` & `/audio/voices`

**无变化**

## 兼容性

### ✅ 保持兼容
- `/tts_url` 仍然接受 `audio_paths` 参数（取第一个）
- `/tts` 和 `/audio/speech` 请求格式不变
- 响应格式保持一致（WAV音频）
- 所有端点路径保持不变

### ❌ 不兼容
- 移除了 `seed` 参数（IndexTTS2 不支持）
- 每个speaker只使用第一个参考音频
- 必须使用 IndexTTS-2 模型（不兼容 IndexTTS-1.5）

## 文件清单

### 修改的文件
- ✏️ `api_server.py` - 适配 IndexTTS2
- ✏️ `api_example.py` - 更新示例代码
- ✏️ `README.md` - 添加v2说明
- ✏️ `README_EN.md` - 添加v2说明

### 新增的文件
- ➕ `API_USAGE.md` - API使用文档
- ➕ `MIGRATION_V1_TO_V2.md` - 迁移指南
- ➕ `CHANGELOG_API_V2.md` - 本文件

## 测试建议

### 1. 基本功能测试
```bash
# 启动服务
python api_server.py --model_dir /path/to/IndexTTS-2-vLLM

# 健康检查
curl http://localhost:6006/health

# 运行示例
python api_example.py
```

### 2. 兼容性测试
```bash
# 测试向后兼容（使用 audio_paths）
curl -X POST "http://localhost:6006/tts_url" \
  -H "Content-Type: application/json" \
  -d '{"text": "测试", "audio_paths": ["/path/to/audio.wav"]}' \
  --output test.wav
```

### 3. OpenAI API测试
```bash
curl -X POST "http://localhost:6006/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "测试", "voice": "default"}' \
  --output test.wav
```

## 设计原则

本次更新遵循以下原则：

### ✅ MVP原则
- 只实现必要功能，保持代码简洁
- 直接使用 `IndexTTS2.infer()` 而不添加额外抽象层
- 内部维护简单的 `speaker_audio_dict` 映射

### ✅ Never Nesting原则
- 所有API端点保持扁平结构
- 使用早期返回处理错误情况
- 嵌套深度不超过3层

### ✅ 解耦原则
- API层与推理层分离
- speaker管理与推理逻辑分离
- 每个端点职责单一明确

### ✅ Let it Crash原则
- 不过度try-catch，让错误自然暴露
- 保留详细的错误堆栈信息
- 使用统一的错误响应格式

## 相关Issue

- [#119](https://github.com/Ksuriuri/index-tts-vllm/issues/119) - indexTTS2 使用 python api_server.py 报错
- [#107](https://github.com/Ksuriuri/index-tts-vllm/issues/107) - V2 模型怎么操作适配？

## 贡献者

感谢社区贡献者在Issue中提供的反馈和建议。

