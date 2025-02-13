import traceback
from threading import Thread
 
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import GenerationConfig
from transformers import TextIteratorStreamer
import time
import torch
 
 
def chat(model,tokenizer,streamer,system,message,history):
    try:
        # assistant
        messages = [
            {"role": "system", "content": system},
        ]
        if len(history) > 0:
            for his in history:
                user = his[0]
                assistant = his[1]
 
                user_obj = {"role": "user", "content": user}
                assistant_obj = {"role": "assistant", "content": assistant}
 
                messages.append(user_obj)
                messages.append(assistant_obj)
 
        messages.append( {"role": "user", "content": message})
 
        # print(messages)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )
 

        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    
        generation_kwargs = dict(inputs=model_inputs.input_ids, streamer=streamer)
 
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
 
        thread.start()
 
        for new_text in streamer:
            yield new_text
 
 
    except Exception:
        traceback.print_exc()
 
def generate(model,tokenizer,system,message,history):
    try:
        # assistant
        messages = [
            {"role": "system", "content": system},
        ]
        if len(history) > 0 :
            for his in history:
                user = his[0]
                assistant = his[1]
 
                user_obj = {"role": "user", "content": user}
                assistant_obj = {"role": "assistant", "content": assistant}
 
                messages.append(user_obj)
                messages.append(assistant_obj)
 
        messages.append({"role": "user", "content": message})
 
        print(messages)
 
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
 
        generated_ids = model.generate(
            model_inputs.input_ids
        )
 
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
 
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception:
        traceback.print_exc()
 
def loadTokenizer(modelPath):
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    return tokenizer
 
def getStreamer(tokenizer):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    return streamer
 
def loadModel(config, modelPath):
    model = AutoModelForCausalLM.from_pretrained(modelPath, torch_dtype="auto",device_map="auto")
    model.generation_config = config
    return model
 
if __name__ == '__main__':

    modelPath = "./Qwen/Qwen2.5-7B-Instruct"
    config = GenerationConfig.from_pretrained(modelPath, top_p=0.9, temperature=0.45, repetition_penalty=1.1, do_sample=True, max_new_tokens=8192)
    tokenizer = loadTokenizer(modelPath)
    model = loadModel(config, modelPath)
    streamer = getStreamer(tokenizer)
    start_time = time.time()
    

    # 你是一名有用的助手，你需要像配音师通过对视频片段内容的描述（文字描述），给出应该为这段视频配上的音效描述。你需要对一个片段生成1/2个简单的音频描述（不能超过两个），此外你还需要生成一个negative prompt，避免后续程序生成不符合要求的音频（如music,human-speaking）。
    # 注意：
    # 1. 生成的音频描述要简单明确，易于实现，要符合现实情况，不要出现如“DNA旋转的声音”等不符合现实的描述 
    # 2. 生成的音频描述要与视频内容相关
    # 3. 不要出现模糊的描述，如“嗡嗡声”，“背景音“，“旁白声”等没有具体内容和发声实体的描述
    # 例如：
    # 输入的视频描述：一个人坐着一把带有火箭的椅子飞上天空
    # 输出的音效描述：火箭发射

    system = """
    You are a helpful assistant. Like a voice actor, you need to provide sound effect descriptions for a video clip based on its content (text description). You need to generate 1-2 simple audio descriptions for a clip (no more than two)

    **Note:**
    1. 

    1. The generated audio descriptions should be clear and conform to reality. Do not include unrealistic descriptions such as "the sound of DNA rotating."
    2. The audio content should be reasonable, such as the sound that the object/object movement/environment in the video can make.
    3. Do not include vague descriptions, such as "buzzing", "background sound", "voiceover" etc., which lack specific content and the entity making the sound.
    4. To clarify the sound (e.g. wind, sword, weapon, explosion, etc.), you only need to give a simple description, such as: wind whistling, sword piercing, weapon shooting, rocket launch, etc.

    **Example:**

    Input video description: A person sitting in a chair with a rocket flies into the sky
    Output sound effect description: Rocket launch
    
    """

    system_v1 = """
    You are a helpful assistant. Just like a voice actor, you are required to provide sound effect descriptions for a video clip according to its content (text description). You need to generate one or two simple audio descriptions for a clip (no more than two).

    Think step by step:
    1. Identify the entities in the video (such as people, animals, objects, etc.), and their behaviors in the video. Consider "[Entity] makes [adjective] sound / [Action] makes sound".
    2. Determine whether the environment and climate in the video will produce ambient sounds (such as wind sound, rain sound, thunder sound).
    3. Consider whether the entity + action would make a sound in reality (for example, the sun's movement doesn't make a sound). You have three options:
    Option 1: The entity doesn't produce such a sound and there is no reasonable ambient sound -> Output "audio" as: None.
    Option 2: The entity doesn't make a sound but there is reasonable ambient sound -> Output the reasonable ambient sound.
    Option 3: The entity's sound - making is reasonable and there is reasonable ambient sound -> Output the sound - making entity + ambient sound.

    Note: Each audio prompt in the output should not exceed two words.

    Example output:
    ```json
    {
        "background audio": [audio,...],
        "audio":  [audio,...],
    }
    ```
    Input: {video caption}
    Output: Please analyze the video elements and create audio prompts in JSON format. 
    
    """

    message = "The video begins with a dark, starry night sky, gradually transitioning to reveal a full moon rising in the upper left corner. The moon continues to move across the sky, eventually disappearing from view as it reaches the right side of the frame. The video ends with the moon no longer visible."
    history = []    
    
    # message = "我家在哪里？"  
    # history = [('hi，你好','你好！有什么我可以帮助你的吗？'),('我家在广州，很好玩哦','广州是一个美丽的城市，有很多有趣的地方可以去。'),]
 
    response = chat(model,tokenizer,streamer,system_v1,message,history)
 
    result = []
    for r in response:
        result.append(r)
 
    print("".join(result))
 
    end_time = time.time()
    print("执行耗时: {:.2f}秒".format(end_time-start_time))