import os

from datasets.model_transforms import init_transform_dict
from visualization.video.video_processor import VideoProcessor
import cv2
import base64
from PIL import Image
from io import BytesIO


class VideoSet:
    def __init__(self,video_path,frame_num,input_res):
        self.frame_num = frame_num
        self.video_path = video_path
        self.frame_idx = []
        self.frames_origin = []
        self.frames_converted = []
        self.video_name = []
        self.img_transforms = init_transform_dict(input_res)['clip_test']
        # 获取目录下的所有文件和目录列表
        entries = os.listdir(video_path)
        # 过滤出所有的文件
        file_names = [entry for entry in entries if os.path.isfile(os.path.join(video_path, entry))]
        for video_name in file_names:
            video = os.path.join(video_path, video_name)
            frame_idxs, frames_origin, frames_converted = VideoProcessor.load_frames_from_video(video,self.frame_num)
            self.video_name.append(video_name)
            self.frame_idx.append(frame_idxs)
            self.frames_origin.append(VideoSet.toBase64(frames_origin))
            self.frames_converted.append(self.img_transforms(frames_converted))
    @staticmethod
    def toBase64(frames):
        base64_imgs = []
        for frame in frames:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            base64_imgs.append(img_str)
        return base64_imgs