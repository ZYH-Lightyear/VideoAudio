import ffmpeg
import os
import tempfile
import shutil
from typing import Optional
from moviepy import *


# example 
# input_video = "/hpc2hdd/home/yzhang679/codes/vid_audio/results/aigc_1/aigc_1-Scene-133.mp4"
# convert_flac_to_mp3(input_video, input_video)

def convert_flac_to_mp3(
    input_path: str, 
    output_path: Optional[str] = None
) -> bool:
    """
    Convert FLAC audio codec in MP4 to MP3
    Args:
        input_path: Path to input video file
        output_path: Path for output video file (optional)
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        if output_path is None:
            output_path = input_path
            
        # Create temporary file
        temp_output = os.path.join(
            tempfile.gettempdir(),
            f"temp_{os.path.basename(output_path)}"
        )
        
        # Convert using ffmpeg with temporary file
        (
            ffmpeg
            .input(input_path)
            .output(
                temp_output,
                acodec='mp3',
                vcodec='copy'
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Replace original file with converted file
        shutil.move(temp_output, output_path)
        
        print(f"Successfully converted {input_path}")
        return True
        
    except Exception as e:
        print(f"Error converting video: {str(e)}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False

def concat_video(video_paths, output_path):
    # 加载视频文件
    video_clips = [VideoFileClip(video_path) for video_path in video_paths]

    # 在时间轴上拼接视频
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    # 保存结果
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print("Videos successfully concatenated!")

def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt

def get_sorted_files(directory):
    # Get absolute paths for all files in directory
    file_paths = [os.path.join(os.path.abspath(directory), file) 
                 for file in os.listdir(directory)]
    # Sort paths
    return sorted(file_paths)

def get_extended_filename(original_path, out_dir):
    """Generate extended filename while preserving original name"""
    original_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(original_name)[0]
    ext = os.path.splitext(original_name)[1]
    return os.path.join(out_dir, f"{name_without_ext}_extended{ext}")

def concatenate_video(input_video_path, temp_output_path, target_duration=1.0):
    """Extend video by concatenating it with itself until reaching target duration"""
    try:
        video = VideoFileClip(input_video_path)
        duration = video.duration
        repeats = int(target_duration / duration) + 1
        
        # Create a temporary file for the concat operation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
            # Write the file list for ffmpeg concat
            for _ in range(repeats):
                f.write(f"file '{input_video_path}'\n")
            f.flush()
            
            # Use ffmpeg concat demuxer
            stream = (
                ffmpeg
                .input(f.name, f='concat', safe=0)
                .output(temp_output_path, c='copy')
                .overwrite_output()
            )
            stream.run(quiet=True)
        
        video.close()
        return temp_output_path
        
    except Exception as e:
        print(f"Error in concatenate_video: {str(e)}")
        return None

def extend_short_video(input_video_path, temp_output_path):
    """Extend video duration to 1 second using multiple methods"""
    try:
        # Read video info using moviepy instead of ffprobe
        video = VideoFileClip(input_video_path)
        fps = video.fps
        duration = video.duration
        
        # Calculate the required speed factor to extend to 1 second
        speed_factor = duration  # This will slow down the video to reach 1 second
        
        # Use ffmpeg directly with setpts filter (slower playback)
        stream = (
            ffmpeg
            .input(input_video_path)
            .filter('setpts', f'{1/speed_factor}*PTS')  # Slow down the video
            .output(temp_output_path)
            .overwrite_output()
        )
        
        try:
            stream.run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            print("ffmpeg error occurred:", e.stderr.decode())
            # Fallback method: just copy the video multiple times
            with open(temp_output_path, 'wb') as outfile:
                for _ in range(int(1/duration) + 1):
                    with open(input_video_path, 'rb') as infile:
                        outfile.write(infile.read())
        
        video.close()
        return temp_output_path
        
    except Exception as e:
        print(f"Slow motion method failed: {str(e)}")
        print("Trying concatenation method...")
        
        # Try concatenation method
        result = concatenate_video(input_video_path, temp_output_path)
        if result:
            return result
            
        # If all methods fail, fall back to simple copy
        print("All extension methods failed, falling back to simple copy...")
        import shutil
        shutil.copy2(input_video_path, temp_output_path)
        return temp_output_path

def merge_videos_in_directory(input_dir, output_path):
    """
    Merge all mp4 videos in a directory in alphabetical order
    Args:
        input_dir: Directory containing mp4 files
        output_path: Path for the merged video file
    """
    try:
        # Get all mp4 files and sort them
        video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        video_files.sort()  # Sort alphabetically
        
        if not video_files:
            print("No mp4 files found in directory")
            return None
            
        # Create temporary file for concat operation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
            # Write the file list for ffmpeg concat
            for video in video_files:
                full_path = os.path.join(input_dir, video)
                f.write(f"file '{os.path.abspath(full_path)}'\n")
            f.flush()
            
            # Use ffmpeg concat demuxer
            try:
                stream = (
                    ffmpeg
                    .input(f.name, f='concat', safe=0)
                    .output(output_path, c='copy')
                    .overwrite_output()
                )
                stream.run(quiet=True)
                print(f"Successfully merged {len(video_files)} videos into {output_path}")
                return output_path
                
            except ffmpeg.Error as e:
                print(f"FFmpeg error during merge: {str(e)}")
                return None
                
    except Exception as e:
        print(f"Error in merge_videos: {str(e)}")
        return None

if __name__ == '__main__':
    video_path = ['/hpc2hdd/home/yzhang679/codes/vid_audio/test_seg/Weapon_23s/Weapon_23s-Scene-007.mp4',
                '/hpc2hdd/home/yzhang679/codes/vid_audio/test_seg/Weapon_23s/Weapon_23s-Scene-008.mp4',
                '/hpc2hdd/home/yzhang679/codes/vid_audio/test_seg/Weapon_23s/Weapon_23s-Scene-009.mp4']

    concat_video(video_path, 'temp.mp4')