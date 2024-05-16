import torch

from visualization.collator import Collator
from visualization.video.video_set import VideoSet
class Config():
    def __init__(self):
        self.videos_dir = 'videodata/'
        self.num_frames = 12
        self.video_sample_type = 'uniform'
        self.input_res = 224

        self.model_path = 'model/model.pth'
        self.load_epoch = -1
        self.metric = 't2v'

        self.huggingface = True
        self.arch = 'clip_transformer'
        self.embed_dim = 512
        self.text_pooling_type = 'mean-pooling'
        self.num_mha_heads = 1
        self.transformer_dropout = 0.4
if __name__ == '__main__':
    config = Config()
    collator = Collator(config)
    video_set = VideoSet(config.videos_dir,config.num_frames,config.input_res)
    rank_list = collator.sort("a cat play with a monkey",video_set.frames_converted)
    for idx in rank_list:
        print(video_set.video_name[idx])
    top_idx = rank_list[0]
    attention_weights = collator.get_attention_weights("a cat play with a monkey",[video_set.frames_converted[top_idx]])
    print(attention_weights)
    attention_weights = torch.squeeze(attention_weights)
    print(attention_weights)
    html = ''
    for idx,frame in enumerate(video_set.frames_origin[top_idx]):
        # 在HTML中使用<img>标签显示图像
        img_tag = f'<p><img src="data:image/jpeg;base64,{frame}"/>{str(attention_weights[idx])}<p>'
        html += img_tag
    # 将HTML字符串写入到一个HTML文件中
    with open('frame.html', 'w') as f:
        f.write(html)