import os
from utils.qwen2_video_understand import qwen_video_inference, init_qwen_vl_model
from moviepy import *
from utils.tools import load_prompt
# from utils.qwen2_5_video_understand import qwen_video_inference as qwen_video_inference_2_5
# from utils.qwen2_5_video_understand import init_qwen_vl_model as init_qwen_vl_model_2_5 
from typing import Dict, List

def get_sorted_files(directory):
    # Get absolute paths for all files in directory
    file_paths = [os.path.join(os.path.abspath(directory), file) 
                 for file in os.listdir(directory)]
    # Sort paths
    return sorted(file_paths)

def collect_keyframe_paths(video_path) -> Dict[str, List[str]]:
    """获取指定视频的关键帧图像路径
    Args:
        base_dir: 关键帧根目录 (result_keyframes)
        video_name: 视频名称
    Returns:
        Dict[str, List[str]]: {
            "scene_name": [frame_path1, frame_path2, ...]
        }
    """
    
    scenes = {}
    
    # 获取场景文件夹
    scene_folders = sorted([d for d in os.listdir(video_path) 
                          if os.path.isdir(os.path.join(video_path, d))])
    
    # 处理每个场景文件夹中的关键帧
    for scene in scene_folders:
        scene_path = os.path.join(video_path, scene)
        frame_paths = sorted([
            'file://' + os.path.abspath(os.path.join(scene_path, f))
            for f in os.listdir(scene_path)
            if f.endswith('.jpg')
        ])
        scenes[scene] = frame_paths
    
    return scenes

# def get_video_doc(video_path, seg_output_dir, keyframes_path, use_prompt=True):

#     # 获取视频片段路径
#     video_segments_paths = get_sorted_files(seg_output_dir)

#     print('共分割出%d个视频片段' % len(video_segments_paths))

#     scenes = collect_keyframe_paths(keyframes_path)

#     # 生成视频简单描述文档
#     model, processor = init_qwen_vl_model()
    
#     print('开始生成视频描述文档')
#     video_captions = {}
#     if use_prompt == True:
#         # 读取prompt
#         prompt_path = './prompt/video_understand.txt'
#         prompt = load_prompt(prompt_path)
#         i = 0
#         for scene_name, frames in scenes.items():
#             print(f"\nScene: {scene_name}")
#             print(f"Frame count: {len(frames)}")
#             # print("Example frames:")
#             # for frame in frames[:2]:  # 只打印前两个帧路径作为示例
#             #     print(f"  {frame}")
#             result = qwen_video_inference(frames, model, processor, prompt)

#             print(result)

#             video_captions[video_segments_paths[i]] = result
            
#             i += 1
#     else:
#         video_captions[video_segments_paths[i]] = ""

#     return video_captions
        
    

def get_video_doc(video_path, seg_output_dir, keyframes_path, use_prompt=True):
    # 获取视频片段路径
    video_segments_paths = get_sorted_files(seg_output_dir)
    print('共分割出%d个视频片段' % len(video_segments_paths))

    scenes = collect_keyframe_paths(keyframes_path)

    # 生成视频简单描述文档
    model, processor = init_qwen_vl_model()
    print('开始生成视频描述文档')
    
    video_captions = {}
    if use_prompt:
        prompt = load_prompt('./prompt/video_understand.txt')
        
        # 使用zip同时遍历场景和视频片段路径
        for (scene_name, frames), video_segment in zip(scenes.items(), video_segments_paths):
            print(f"\nScene: {scene_name}")
            print(f"Frame count: {len(frames)}")
            
            result = qwen_video_inference(frames, model, processor, prompt)
            print(result)
            video_captions[video_segment] = result
    else:
        # 如果不使用prompt，为所有片段设置空字符串
        video_captions = {path: "" for path in video_segments_paths}

    return video_captions

if __name__ == "__main__":
    name = 'Weapon_23s'
    result_dir = f'./results/{name}'
    seg_output_dir = f'./test_seg/{name}'
    video_path = f'./test_video/mutilclips/{name}.mp4'
    keyframes_path = f'./result_keyframes/{name}/'
    
    doc = get_video_doc(
                video_path=video_path, 
                seg_output_dir=seg_output_dir, 
                keyframes_path=keyframes_path,
                use_prompt=True
                )