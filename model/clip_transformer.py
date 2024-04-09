import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)


    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        topic_text_data = data['topic_text'] if 'topic_text' in data else None
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        topic_text_features = None
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
            if topic_text_data is not None:
                topic_features = [self.clip.get_text_features(**texts) for texts in topic_text_data]
                topic_features = [item.mean(dim=0) for item in topic_features]
                topic_features = torch.stack(topic_features, dim=0)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)
            if topic_text_data is not None:
                topic_features = [self.clip.encode_text(texts) for texts in topic_text_data]
        if topic_text_data is not None:
            text_features = text_features + topic_features*0.01
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(text_features, video_features)
            
        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled
