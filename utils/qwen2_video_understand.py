from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
from typing import List, Optional
from utils.tools import load_prompt

def init_qwen_vl_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        min_pixels=256*256, 
        max_pixels=640*640
    )
    return model, processor

def qwen_video_inference(
    video_path: str, 
    model: Optional[Qwen2VLForConditionalGeneration] = None,
    processor: Optional[AutoProcessor] = None,
    prompt = None
) -> str:
    try:
        # Initialize model if not provided
        if model is None or processor is None:
            model, processor = init_qwen_vl_model()
        
        # Process video frames
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": video_path,
        #                 "max_pixels": 360 * 420,
        #                 "fps": 1.0,
        #             },
        #             {"type": "text", 
        #              "text": 
        #                     '''
        #                     You are an animation voice actor who analyzes videos and provides descriptions of the sound effects that should be included. 
        #                     1. Ignore the watermark and title in the video. 
        #                     2. The format of audio output should be concise and clear, avoiding being too complex.
        #                     3. Output at most two audio descriptions.
        #                     The output format is as follows:
        #                     ' Audio 1', 'Audio 2'

        #                     For example:
        #                     'Engine rumble', 'Gunfire'
        #                     '''
                            
        #             },
        #         ],
        #     }
        # ]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 2.0,
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            }
        ]
        
        # Process input and run inference
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:0")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)
        
        return output_text[0]
        
    except Exception as e:
        return f"Error during video inference: {str(e)}"

def main():
    path = '/hpc2hdd/home/yzhang679/codes/vid_audio/test_seg/aigc_1/aigc_1-Scene-001.mp4'
    video_path = 'file://' + path
    model, processor = init_qwen_vl_model()
    result = qwen_video_inference(video_path, model, processor)
    
    print(result)
    
if __name__ == "__main__":
    main()