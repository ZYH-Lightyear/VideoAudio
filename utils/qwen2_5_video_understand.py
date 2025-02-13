from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.tools import load_prompt
from typing import List, Optional


def init_qwen_vl_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    min_pixels = 256*28*28
    max_pixels = 512*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    return model, processor

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.

def qwen_video_inference(
    images_path: str, 
    model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
    processor: Optional[AutoProcessor] = None,
    prompt = None
) -> str:
    try:
        # Initialize model if not provided
        if model is None or processor is None:
            model, processor = init_qwen_vl_model()

        user_prompt = load_prompt("./prompt/video_understand.txt")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": images_path,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

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
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
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