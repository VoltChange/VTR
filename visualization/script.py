import argparse

import numpy
import torch

from collator import Collator
from video.video_set import VideoSet
class Config():
    def __init__(self):
        self.videos_dir = 'videodata/'
        self.num_frames = 12
        self.video_sample_type = 'uniform'
        self.input_res = 224

        self.model_path = 'modeldata/model.pth'
        self.load_epoch = -1
        self.metric = 't2v'

        self.huggingface = True
        self.arch = 'clip_transformer'
        self.embed_dim = 512
        self.text_pooling_type = 'mean-pooling'
        self.num_mha_heads = 1
        self.transformer_dropout = 0.4
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str)
    args = parser.parse_args()
    input = args.input
    config = Config()
    collator = Collator(config)
    video_set = VideoSet(config.videos_dir,config.num_frames,config.input_res)
    rank_list = collator.sort(input,video_set.frames_converted)
    topK = min(3, len(rank_list))
    topF = 3
    video_idx = rank_list[0:topK].tolist()
    video_name = [video_set.video_name[idx] for idx in video_idx]
    frame_num = [video_set.video_frame_num[idx] for idx in video_idx]
    frames_origin = [video_set.frames_origin[idx] for idx in video_idx]
    frames_converted = [video_set.frames_converted[idx] for idx in video_idx]
    attention_weights = collator.get_attention_weights(input, frames_converted)
    html='<div style="display:flex">'
    for video_idx in range(topK):
        video_box = '<div><p>'+video_name[video_idx]+'</p>'
        weights = attention_weights[video_idx].squeeze()
        weights_rank = numpy.argsort(-weights).tolist()[0:topF]
        top_frames = []
        for idx in weights_rank:
            img_tag = f'<p><img src="data:image/jpeg;base64,{frames_origin[video_idx][idx]}"/>{float(weights[idx])}<p>'
            video_box += img_tag
        video_box +='</div>'
        html += video_box
    html+='</div>'
    # 将HTML字符串写入到一个HTML文件中
    with open('frame.html', 'w') as f:
        f.write(html)