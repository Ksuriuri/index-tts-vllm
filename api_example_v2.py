from dataclasses import asdict, dataclass
import os
from typing import List, Optional
import requests

SERVER_PORT = 6006
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

url = F"http://0.0.0.0:{SERVER_PORT}/tts_url"


@dataclass
class IndexTTS2RequestData:
    text: str
    spk_audio_path: str
    emo_control_method: int = 0
    emo_ref_path: Optional[str] = None
    emo_weight: float = 1.0
    emo_vec: List[float] = None
    emo_text: Optional[str] = None
    emo_random: bool = False
    max_text_tokens_per_sentence: int = 120

    def __post_init__(self):
        # 保证 emo_vec 默认长度为 8 的 0 向量
        if self.emo_vec is None:
            self.emo_vec = [0.0] * 8

    def to_dict(self) -> str:
        return asdict(self)


# 1. 情感与音色参考音频相同
data = IndexTTS2RequestData(
    text="还是会想你，还是想登你",
    spk_audio_path="assets/jay_promptvn.wav"
)

response = requests.post(url, json=data.to_dict())
with open(os.path.join(output_dir, "output1.wav"), "wb") as f:
    f.write(response.content)



# 2. 使用情感参考音频
data = IndexTTS2RequestData(
    text="还是会想你，还是想登你",
    spk_audio_path="assets/jay_promptvn.wav",
    emo_control_method=1,
    emo_ref_path="assets/vo_card_klee_endOfGame_fail_01.wav",
    emo_weight=0.6
)

response = requests.post(url, json=data.to_dict())
with open(os.path.join(output_dir, "output2.wav"), "wb") as f:
    f.write(response.content)



# 3. 使用情感向量控制
# ["喜", "怒", "哀", "惧", "厌恶", "低落", "惊喜", "平静"]
emo_vec = [0, 0, 0.55, 0, 0, 0, 0, 0]

data = IndexTTS2RequestData(
    text="还是会想你，还是想登你",
    spk_audio_path="assets/jay_promptvn.wav",
    emo_control_method=2,
    emo_vec=emo_vec
)

response = requests.post(url, json=data.to_dict())
with open(os.path.join(output_dir, "output3.wav"), "wb") as f:
    f.write(response.content)



# 4. 使用情感描述文本控制，并且根据'text'另外生成情绪参数
data = IndexTTS2RequestData(
    text="还是会想你，还是想登你",
    spk_audio_path="assets/jay_promptvn.wav",
    emo_control_method=3,
    emo_text="极度悲伤"
)

response = requests.post(url, json=data.to_dict())
with open(os.path.join(output_dir, "output4.wav"), "wb") as f:
    f.write(response.content)




# 5. 使用情感描述文本控制，不会再根据'text'另外生成情绪参数
data = IndexTTS2RequestData(
    text="还是会想你，还是想登你",
    spk_audio_path="assets/jay_promptvn.wav",
    # 这里的4没有匹配，可能需要作者写下匹配
    emo_control_method=4,
    # 指定情绪描述
    emo_text="极度悲伤",
    # 情绪烈度 原项目中的'emo_alpha'
    emo_weight=0.6
)

response = requests.post(url, json=data.to_dict())
with open(os.path.join(output_dir, "output5.wav"), "wb") as f:
    f.write(response.content)
