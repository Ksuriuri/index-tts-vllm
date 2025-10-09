"""
IndexTTS-2 API 使用示例

本示例展示如何调用 IndexTTS-2 的 API 接口进行语音合成。
更多详细信息请参考 API_USAGE.md
"""

import requests

SERVER_PORT = 6006
SERVER_URL = f"http://0.0.0.0:{SERVER_PORT}"

# 示例 1: 使用本地音频文件作为参考
# 将 audio_path 改为你的本地音频文件路径
print("示例 1: 使用音频路径生成语音")
url = f"{SERVER_URL}/tts_url"
data = {
    "text": "还是会想你，还是想登你",
    "audio_path": "assets/jay_promptvn.wav"  # IndexTTS-2 使用单个参考音频
}

response = requests.post(url, json=data)
if response.status_code == 200:
    with open("output_example1.wav", "wb") as f:
        f.write(response.content)
    print("✓ 生成成功: output_example1.wav")
else:
    print(f"✗ 生成失败: {response.json()}")


# 示例 2: 使用预设的speaker/character
# speaker需要在 assets/speaker.json 中预先配置
print("\n示例 2: 使用预设speaker生成语音")
url = f"{SERVER_URL}/tts"
data = {
    "text": "还是会想你，还是想登你",
    "character": "jay_klee"
}

response = requests.post(url, json=data)
if response.status_code == 200:
    with open("output_example2.wav", "wb") as f:
        f.write(response.content)
    print("✓ 生成成功: output_example2.wav")
else:
    print(f"✗ 生成失败: {response.json()}")


# 示例 3: OpenAI 兼容 API
print("\n示例 3: 使用 OpenAI 兼容接口")
url = f"{SERVER_URL}/audio/speech"
data = {
    "model": "tts-1",
    "input": "这是一个使用 OpenAI 兼容接口的示例",
    "voice": "jay_klee"
}

response = requests.post(url, json=data)
if response.status_code == 200:
    with open("output_example3.wav", "wb") as f:
        f.write(response.content)
    print("✓ 生成成功: output_example3.wav")
else:
    print(f"✗ 生成失败: {response.json()}")


# 示例 4: 获取可用的voices列表
print("\n示例 4: 获取可用的voices")
url = f"{SERVER_URL}/audio/voices"
response = requests.get(url)
if response.status_code == 200:
    voices = response.json()
    print(f"✓ 可用的voices: {list(voices.keys())}")
else:
    print(f"✗ 获取失败")


# 示例 5: 健康检查
print("\n示例 5: 健康检查")
url = f"{SERVER_URL}/health"
response = requests.get(url)
if response.status_code == 200:
    health = response.json()
    print(f"✓ 服务状态: {health['status']}")
else:
    print(f"✗ 服务不可用")


print("\n所有示例执行完成！详细API文档请查看 API_USAGE.md")