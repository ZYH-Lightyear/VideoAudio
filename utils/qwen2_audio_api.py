from dashscope import MultiModalConversation
from dotenv import load_dotenv


load_dotenv()

# 请用您的本地音频的绝对路径替换 ABSOLUTE_PATH/welcome.mp3
audio_file_path = "file:///hpc2hdd/home/yzhang679/codes/vid_audio/results/Weapon_23s/Weapon_23s-Scene-004.flac"
messages = [
    {
        "role": "system", 
        "content": [{"text": "You are a helpful assistant."}]},
    {
        "role": "user",
        # "The video depicts the scene where the artillery is firing, please rate this audio for relevance to the video (0-10 points)."
        # 视频描述了火炮在射击的场景，请为这段音频与视频的相关性打分（0-10分）
        "content": [{"audio": audio_file_path}, {"text": "What's that sound?"}],
    }
]

response = MultiModalConversation.call(model="qwen-audio-turbo-latest", messages=messages)
print(response)