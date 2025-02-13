# import spaces
import logging
from moviepy import *
from datetime import datetime
from pathlib import Path
from utils.qwen_llm_api import LLM_API
from utils.segment_video import segment_video
from utils.sample_key_frame import process_video_folder
from utils.video_doc_gen import get_video_doc
from utils.tools import get_sorted_files, extend_short_video, get_extended_filename, merge_videos_in_directory
import gradio as gr
import torch
import torchaudio
import os

try:
    import mmaudio
except ImportError:
    os.system("pip install -e .")
    import mmaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils
import tempfile

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

device = 'cuda'
dtype = torch.bfloat16

model: ModelConfig = all_model_cfg['large_44k_v2']
model.download_if_needed()
output_dir = Path('./output/gradio')

setup_eval_logging()


def get_model() -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
    seq_cfg = model.seq_cfg

    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    return net, feature_utils, seq_cfg

net, feature_utils, seq_cfg = get_model()

# @spaces.GPU(duration=120)
@torch.inference_mode()
def video_to_audio(video: gr.Video, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                   cfg_strength: float, duration: float):
    
    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    
    # 获取video文件名
    video_path = Path(video)
    temp_dir = video_path.parent
    
    seg_dir = temp_dir / 'segments'
    keyframes_dir = temp_dir / 'keyframes'
    clip_results_dir = temp_dir / 'clip_results'

    print(f"Video path: {video}")
    print(f"Temp directory: {str(temp_dir)}")
    print(f"Segments directory: {str(seg_dir)}")
    print(f"Keyframes directory: {str(keyframes_dir)}")
    print(f"Result dir: {str(clip_results_dir)}")
    
    # 视频镜头分割
    if not os.path.exists(seg_dir):
        # Create segments directory
        print("Creating segments directory")
        os.makedirs(seg_dir, exist_ok=True)
        # segment video 
        scenes = segment_video(input_path=video, output_dir=seg_dir)

    # 关键帧提取
    results = process_video_folder(input_folder=seg_dir, output_base_dir=keyframes_dir)
    # 生成视频描述
    video_captions = {}
    video_captions = get_video_doc(video_path=video, seg_output_dir=seg_dir, keyframes_path=keyframes_dir, use_prompt=True)
    # print(video_captions)

    # 获取视频片段路径
    video_segments_paths = get_sorted_files(seg_dir)

    # 根据视频内容生成音频prompt
    audio_prompts = {}
    qwen = LLM_API(model="qwen-plus", api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=os.getenv("DASHSCOPE_API_KEY"))
    
    if os.path.exists(clip_results_dir) == False:
        print("Creating clip results directory")
        os.makedirs(clip_results_dir, exist_ok=True)

    for i in range(len(video_segments_paths)):
        audio_prompt = qwen.generate(prompt=video_captions[video_segments_paths[i]])
        audio_prompts[video_segments_paths[i]] = audio_prompt
        print(audio_prompt)
    
    for i in range(len(video_segments_paths)):
        video = VideoFileClip(video_segments_paths[i])  # 替换为你的视频文件路径
        # 获取视频长度（秒）
        video_duration = video.duration
        print(f"视频长度: {video_duration} 秒")
        
        if video.duration < 1.0:
            print(f"视频长度小于1秒，进行时长扩展处理...")
            input_video_path = video_segments_paths[i]
            extended_path = get_extended_filename(input_video_path, seg_dir)
            extended_path = extend_short_video(input_video_path, extended_path)
            
            print(extended_path)
            
            video = VideoFileClip(extended_path)
            video_duration = video.duration + 1

            audio_generation(video=extended_path, prompt=audio_prompts[video_segments_paths[i]], negative_prompt=negative_prompt, duration=video_duration, clip_results_dir=clip_results_dir, num_steps=num_steps, cfg_strength=cfg_strength, rng=rng)
        else:            
            audio_generation(video=video_segments_paths[i], prompt=audio_prompts[video_segments_paths[i]], negative_prompt=negative_prompt, duration=video_duration, clip_results_dir=clip_results_dir, num_steps=num_steps, cfg_strength=cfg_strength, rng=rng)

    # fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    # video_info = load_video(video, duration)
    # clip_frames = video_info.clip_frames
    # sync_frames = video_info.sync_frames
    # duration = video_info.duration_sec
    # clip_frames = clip_frames.unsqueeze(0)
    # sync_frames = sync_frames.unsqueeze(0)
    # seq_cfg.duration = duration
    # net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    # audios = generate(clip_frames,
    #                   sync_frames, [prompt],
    #                   negative_text=[negative_prompt],
    #                   feature_utils=feature_utils,
    #                   net=net,
    #                   fm=fm,
    #                   rng=rng,
    #                   cfg_strength=cfg_strength)
    
    # audio = audios.float().cpu()[0]

    # # current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    # video_save_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    # # output_dir.mkdir(exist_ok=True, parents=True)
    # # video_save_path = output_dir / f'{current_time_string}.mp4'
    # make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
    # log.info(f'Saved video to {video_save_path}')
    video_save_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    merge_videos_in_directory(clip_results_dir, video_save_path)
    log.info(f'Saved video to {video_save_path}')

    os.remove(seg_dir)
    os.remove(keyframes_dir)
    os.remove(clip_results_dir)

    return video_save_path

def audio_generation(video, prompt, negative_prompt, duration, clip_results_dir, num_steps, cfg_strength, rng):

    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    log.info(f'Saved video to {video}')

    video_info = load_video(video, duration)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    
    audio = audios.float().cpu()[0]
    print('audio done')
    video = Path(video)
    video_save_path = clip_results_dir / f'{video.stem}.mp4'

    if not os.path.exists(clip_results_dir):
        print('Save path does not exist')
    else:
        print('Save path exists')

    make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
    # log.info(f'Saved video to {video_save_path}')

## *原版本, 别删*

# @spaces.GPU(duration=120)
# @torch.inference_mode()
# def video_to_audio(video: gr.Video, prompt: str, negative_prompt: str, seed: int, num_steps: int,
#                    cfg_strength: float, duration: float):

#     rng = torch.Generator(device=device)
#     if seed >= 0:
#         rng.manual_seed(seed)
#     else:
#         rng.seed()
#     fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

#     video_info = load_video(video, duration)
#     clip_frames = video_info.clip_frames
#     sync_frames = video_info.sync_frames
#     duration = video_info.duration_sec
#     clip_frames = clip_frames.unsqueeze(0)
#     sync_frames = sync_frames.unsqueeze(0)
#     seq_cfg.duration = duration
#     net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

#     audios = generate(clip_frames,
#                       sync_frames, [prompt],
#                       negative_text=[negative_prompt],
#                       feature_utils=feature_utils,
#                       net=net,
#                       fm=fm,
#                       rng=rng,
#                       cfg_strength=cfg_strength)
#     audio = audios.float().cpu()[0]

#     # current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
#     video_save_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
#     # output_dir.mkdir(exist_ok=True, parents=True)
#     # video_save_path = output_dir / f'{current_time_string}.mp4'
#     make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
#     log.info(f'Saved video to {video_save_path}')
#     return video_save_path


# @spaces.GPU(duration=120)
@torch.inference_mode()
def text_to_audio(prompt: str, negative_prompt: str, seed: int, num_steps: int, cfg_strength: float,
                  duration: float):

    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    clip_frames = sync_frames = None
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]

    audio_save_path = tempfile.NamedTemporaryFile(delete=False, suffix='.flac').name
    torchaudio.save(audio_save_path, audio, seq_cfg.sampling_rate)
    log.info(f'Saved audio to {audio_save_path}')


    
    return audio_save_path


video_to_audio_tab = gr.Interface(
    fn=video_to_audio,
    description="""
    Project page: <a href="https://hkchengrex.com/MMAudio/">https://hkchengrex.com/MMAudio/</a><br>
    Code: <a href="https://github.com/hkchengrex/MMAudio">https://github.com/hkchengrex/MMAudio</a><br>
    """,
    inputs=[
        gr.Video(),
        gr.Text(label='Prompt'),
        gr.Text(label='Negative prompt', value='music'),
        gr.Number(label='Seed (-1: random)', value=-1, precision=0, minimum=-1),
        gr.Number(label='Num steps', value=25, precision=0, minimum=1),
        gr.Number(label='Guidance Strength', value=4.5, minimum=1),
        gr.Number(label='Duration (sec)', value=8, minimum=1),
    ],
    outputs='playable_video',
    cache_examples=False,
    title='Auto Video Audio — Long Video-to-Audio Synthesis',
    examples=[
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_beach.mp4',
            'waves, seagulls',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_serpent.mp4',
            '',
            'music',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_seahorse.mp4',
            'bubbles',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_india.mp4',
            'Indian holy music',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_galloping.mp4',
            'galloping',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_kraken.mp4',
            'waves, storm',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_nyc.mp4',
            '',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/mochi_storm.mp4',
            'storm',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_spring.mp4',
            '',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_typing.mp4',
            'typing',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_wake_up.mp4',
            '',
            '',
            0,
            25,
            4.5,
            10,
        ],
    ])

text_to_audio_tab = gr.Interface(
    fn=text_to_audio,
    inputs=[
        gr.Text(label='Prompt'),
        gr.Text(label='Negative prompt'),
        gr.Number(label='Seed (-1: random)', value=-1, precision=0, minimum=-1),
        gr.Number(label='Num steps', value=25, precision=0, minimum=1),
        gr.Number(label='Guidance Strength', value=4.5, minimum=1),
        gr.Number(label='Duration (sec)', value=8, minimum=1),
    ],
    outputs='audio',
    cache_examples=False,
    title='MMAudio — Text-to-Audio Synthesis',
)

if __name__ == "__main__":
    gr.TabbedInterface([video_to_audio_tab, text_to_audio_tab],
                       ['Video-to-Audio', 'Text-to-Audio']).launch(allowed_paths=[output_dir], debug=True)
