from scenedetect import detect, ContentDetector, split_video_ffmpeg
import os

def segment_video(input_path: str, output_dir: str = './video_segments', save=True) -> list:
    """
    Segment video into scenes based on content detection.
    
    Args:
        input_path (str): Path to input video file
        output_dir (str): Directory to save segmented videos
        
    Returns:
        list: List of scene timestamps (scene_start, scene_end)
    """
    # Validate input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")
    
    # Detect scenes
    scenes = detect(input_path, ContentDetector(), './stats.csv')
    
    # Print scene timestamps
    for (scene_start, scene_end) in scenes:
        print(f'{scene_start}-{scene_end}')
        
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split video into segments
    if save:
        split_video_ffmpeg(
            input_video_path=input_path,
            scene_list=scenes,
            output_dir=output_dir
        )
        
    return scenes
