import os
import io
import json

import numpy
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

from collator import Collator
from video.video_set import VideoSet


# 载入模型
class Config():
    def __init__(self):
        self.videos_dir = 'static/videodata/'
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


config = Config()
collator = Collator(config)
video_set = VideoSet(config.videos_dir, config.num_frames, config.input_res)
app = Flask(__name__)
app.static_folder = 'static'
CORS(app)  # 解决跨域问题


@app.route("/search", methods=["POST"])
@torch.no_grad()
def search():
    topF = 3
    result = {'topVideo': [], 'topFrame': [], 'frameNum': []}
    user_input = request.json['user_input']
    rank_list = collator.sort(user_input, video_set.frames_converted)
    topK = min(6, len(rank_list))
    video_idx = rank_list[0:topK].tolist()
    video_name = [video_set.video_name[idx] for idx in video_idx]
    frame_num = [video_set.video_frame_num[idx] for idx in video_idx]
    frames_origin = [video_set.frames_origin[idx] for idx in video_idx]
    frames_converted = [video_set.frames_converted[idx] for idx in video_idx]
    attention_weights = collator.get_attention_weights(user_input, frames_converted)
    result['topVideo'] = video_name
    result['frameNum'] = frame_num
    frame_idxs = [video_set.frame_idx[idx] for idx in video_idx]
    for video_idx in range(topK):
        weights = attention_weights[video_idx].squeeze()
        weights_rank = numpy.argsort(-weights).tolist()[0:topF]
        top_frames = []
        for idx in weights_rank:
            frame = {'base64': frames_origin[video_idx][idx], 'weight': float(weights[idx]),
                     'position': int(frame_idxs[video_idx][idx])}
            top_frames.append(frame)
        result['topFrame'].append(top_frames)
    return jsonify(result)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


@app.route('/videodata/<path:filename>')
def serve_video(filename):
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(root_dir, 'static', 'videos'), filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
