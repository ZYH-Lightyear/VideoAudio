import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.tools import load_prompt
from typing import Optional, Dict, Any
import json

load_dotenv()

class LLM_API:
    """A wrapper class to interact with a language model."""

    def __init__(
        self,
        model: str = "qwen-plus",
        api_base: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: Optional[str] = None,
        prompt_path: str = "./prompt/audio_prompt.txt"
    ) -> None:
        """
        Initialize the LLM.
        Args:
            model: 模型名称
            api_base: API基础URL
            api_key: API密钥，如果为None则从环境变量获取Qwen API密钥
            prompt_path: 系统提示词文件路径
        """
        # 设置API密钥
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.model = model
        self.system_prompt = load_prompt(prompt_path)

    def generate(
        self, 
        prompt: str,
        response_format: str = "json_object",
        **kwargs
    ) -> str:
        """
        生成回复
        Args:
            prompt: 用户提示词
            response_format: 返回格式
            **kwargs: 其他参数
        Returns:
            str: 模型生成的回复
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Video Caption: {prompt}"}
                ],
                response_format={"type": response_format},
                **kwargs
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return ""
    
    def parse_json(self, response: str):
        """
        解析JSON格式的回复
        Args:
            response: 模型生成的回复
        Returns:
            Dict[str, Any]: 解析后的回复
        """
        return json.loads(response)

# 使用示例
if __name__ == "__main__":
    # 创建LLM实例
    qwen = LLM_API(model="qwen-plus", api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=os.getenv("DASHSCOPE_API_KEY"))
    
    # 测试视频描述
    video_caption = """
    The video begins with a dark, starry night sky, gradually transitioning to reveal 
    a full moon rising in the upper left corner. The moon continues to move across 
    the sky, eventually disappearing from view as it reaches the right side of the frame. 
    The video ends with the moon no longer visible.
    """
    
    # 生成回复
    response = qwen.generate(video_caption)
    response_dict = qwen.parse_json(response)
    print(response_dict)



