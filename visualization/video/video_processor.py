import cv2
import random
import numpy as np
import torch
import base64
from PIL import Image
from io import BytesIO


class VideoProcessor:
    @staticmethod
    def getBase64(frame):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    @staticmethod
    def load_frames_from_video(video_path,num_frames):
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened()), video_path
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #获取采样帧间隔
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []

        #选择采样帧
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames_origin = []
        frames_converted = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break
            if ret:
                frames_origin.append(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames_converted.append(frame)
            else:
                raise ValueError

        while len(frames_origin) < num_frames:
            frames_origin.append(frames_origin[-1].clone())
            frames_converted.append(frames_origin[-1].clone())

        frames_converted = torch.stack(frames_converted).float() / 255
        cap.release()
        return frame_idxs,frames_origin,frames_converted