import torch
import torch.nn as nn
from config.base_config import Config
from modules.cluster import get_topic_aggregated_embeddings
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


    def forward(self, data, return_all_frames=False, is_list = False,n_clusters = None):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        use_gpu = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        if self.config.huggingface:
            if is_list:
                text_features_list = []
                for texts in text_data:
                    text_features_list.append(self.clip.get_text_features(**texts))
            else:
                text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
        else:
            if is_list:
                text_features_list = []
                for texts in text_data:
                    text_features_list.append(self.clip.get_text_features(**texts))
            else:
                text_features = self.clip.encode_text(**text_data)
            video_features = self.clip.encode_image(video_data)
        if is_list:
            text_features_list = [text.cpu() for text in text_features_list]
            for idx,texts in enumerate(text_features_list):
                text_features_list[idx] = get_topic_aggregated_embeddings(texts, "agg", n_clusters)
            # [batch_size,n_clusters,dim]
            text_features = torch.stack(text_features_list).to(device)
            # [batch_size*n_clusters,dim]
            text_features = text_features.view(batch_size*n_clusters,-1)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(text_features, video_features)
            
        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled
