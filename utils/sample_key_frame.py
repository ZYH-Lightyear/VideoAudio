import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from typing import List, Dict

class KeyFrameExtractor:
    def __init__(self, video_path: str, base_dir: str = "./results"):
        """
        初始化关键帧提取器
        Args:
            video_path: 输入视频路径
            base_dir: 基础输出目录
        """
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 设置输出目录结构
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, self.video_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化存储
        self.frames = []
        self.shot_boundaries = []
        self.keyframes = []

    def preprocess_video(self):
        """预处理视频,提取所有帧"""
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()
        return self

    def detect_shot_boundaries(self, threshold=10):
        """使用帧差法检测镜头边界"""
        for i in range(1, len(self.frames)):
            prev_frame = cv2.cvtColor(self.frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
            diff = np.mean(np.abs(curr_frame.astype(int) - prev_frame.astype(int)))
            if diff > threshold:
                self.shot_boundaries.append(i)
        return self

    def cal_frame_nums(self) -> int:
        """
        根据视频时长计算应该提取的关键帧数量
        Returns:
            int: 应提取的关键帧数量
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps  # 视频时长(秒)
        cap.release()
        
        # 计算关键帧数量
        if duration <= 1.5:
            return 3
        else:
            # 对于更长的视频，每增加1秒增加1帧
            return 3 + int((duration - 1.5) // 1.0)

    def _sample_frames(self, frames, target_fps=8):
        """
        按照目标FPS采样帧
        Args:
            frames: 原始帧列表
            target_fps: 目标FPS，默认8
        Returns:
            采样后的帧列表和对应的原始索引
        """
        cap = cv2.VideoCapture(self.video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # 如果原始FPS小于目标FPS，返回所有帧
        if original_fps <= target_fps:
            return frames, list(range(len(frames)))

        # 计算采样间隔
        step = int(original_fps / target_fps)
        sampled_frames = []
        sampled_indices = []
        
        for i in range(0, len(frames), step):
            sampled_frames.append(frames[i])
            sampled_indices.append(i)
            
        return sampled_frames, sampled_indices

    def extract_keyframes(self):
        """从每个镜头中提取关键帧"""
        # 采样帧
        shot_frames, original_indices = self._sample_frames(self.frames[2:], target_fps=8)
        n_clusters = self.cal_frame_nums()  # 获取应提取的关键帧数量
        
        # 使用K-means聚类选择关键帧
        frame_features = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten() 
                                 for frame in shot_frames])
        
        # 确保聚类数不超过帧数
        n_clusters = min(n_clusters, len(frame_features))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(frame_features)
        
        # 对每个聚类中心找到最接近的帧
        for cluster_idx in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[cluster_idx]
            distances = np.sum((frame_features - cluster_center) ** 2, axis=1)
            frame_idx = np.argmin(distances)
            
            self.keyframes.append({
                'frame': shot_frames[frame_idx],
                'frame_idx': original_indices[frame_idx],
            })
        
        # 按帧索引排序关键帧
        self.keyframes.sort(key=lambda x: x['frame_idx'])
        return self

    def save_keyframes(self):
        """保存关键帧和元数据"""
        # 保存关键帧图像
        metadata = []
        for i, kf in enumerate(self.keyframes):
            # 生成文件名
            filename = f'keyframe_{i:04d}_frame_{kf["frame_idx"]}.jpg'
            filepath = os.path.join(self.output_dir, filename)
            
            # 保存图像
            cv2.imwrite(filepath, kf['frame'])
            
            # 收集元数据
            metadata.append({
                'keyframe_id': i,
                # 'shot_index': kf['shot_idx'],
                'frame_index': kf['frame_idx'],
                'filename': filename
            })
        
        print(f"已保存 {len(self.keyframes)} 个关键帧到 {self.output_dir}")
        return metadata

    def process(self):
        """执行完整的关键帧提取流程"""
        return (self.preprocess_video()
                .extract_keyframes()
                .save_keyframes())

def process_single_video(video_path: str, output_base_dir: str = "./results") -> List[Dict]:
    """
    处理单个视频的关键帧提取
    Args:
        video_path: 视频文件路径
        output_base_dir: 输出根目录
    Returns:
        List[Dict]: 提取的关键帧元数据列表
    """
    try:
        print(f"\n开始处理视频: {os.path.basename(video_path)}")
        extractor = KeyFrameExtractor(video_path, output_base_dir)
        metadata = extractor.process()
        print(f"视频处理完成，共提取 {len(metadata)} 个关键帧")
        return metadata
    except Exception as e:
        print(f"处理视频时出错: {str(e)}")
        return []

def process_video_folder(input_folder: str, output_base_dir: str = "./results") -> Dict[str, List]:
    """
    批量处理文件夹中的所有视频
    Args:
        input_folder: 输入视频文件夹路径
        output_base_dir: 输出根目录
    Returns:
        Dict[str, List]: 所有视频的处理结果，key为视频名，value为关键帧元数据
    """

    # 获取所有视频文件
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(input_folder) 
                  if os.path.splitext(f)[1].lower() in video_extensions]
    video_files.sort()  # 确保处理顺序一致
    
    results = {}
    total_videos = len(video_files)
    print(f"\n找到 {total_videos} 个视频文件")
    
    os.makedirs(output_base_dir, exist_ok=True)

    # 处理每个视频文件并显示进度
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(input_folder, video_file)
        print(f"\n处理视频 [{i}/{total_videos}]: {video_file}")
        try:
            # 处理单个视频
            extractor = KeyFrameExtractor(video_path, output_base_dir)
            metadata = extractor.process()
            results[video_file] = metadata
            
        except Exception as e:
            print(f"处理视频 {video_file} 时出错: {str(e)}")
            results[video_file] = []
            continue
    
    # 打印总结
    print("\n处理完成:")
    for video_file, metadata in results.items():
        print(f"- {video_file}: 提取了 {len(metadata)} 个关键帧")
    
    return results

if __name__ == "__main__":
    # # 示例1：处理单个视频
    # single_video_path = "/hpc2hdd/home/yzhang679/codes/vid_audio/test_seg/Weapon_23s/Weapon_23s-Scene-008.mp4"
    output_dir = "./result_keyframes/Weapon_23s"
    
    # 处理单个视频
    # metadata = process_single_video(single_video_path, output_dir)
    
    # 示例2：批量处理文件夹
    input_folder = "/hpc2hdd/home/yzhang679/codes/vid_audio/test_seg/"
    results = process_video_folder(input_folder, output_dir)