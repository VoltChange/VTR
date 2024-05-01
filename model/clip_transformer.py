import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)
        if self.config.text_pooling_type == 'attention':
            self.attn = nn.MultiheadAttention(config.embed_dim, 8, dropout=self.config.topic_dropout)
        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)


    def forward(self, data, return_all_frames=False,lambda_coef=0.01):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        topic_text_data = data['topic_text'] if 'topic_text' in data else None
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
            if topic_text_data is not None:
                if self.config.text_pooling_type == 'attention':
                    topic_text_features = [self.clip.get_text_features(**texts) for texts in topic_text_data]
                    batch_size = topic_text_features.__len__()
                    embed_dim = self.config.embed_dim
                    # 找出最大长度
                    max_len = max(t.shape[0] for t in topic_text_features)
                    # 使用padding填充tensor并创建key_padding_mask
                    key_padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool).to(self.device)
                    padded_tensors = []
                    for i, tensor in enumerate(topic_text_features):
                        length = tensor.size(0)
                        # 使用0填充每个tensor到max_len
                        padding = torch.zeros([max_len - length, embed_dim], dtype=tensor.dtype,device=self.device)
                        padded_tensor = torch.cat([tensor, padding], dim=0)
                        padded_tensors.append(padded_tensor)
                        # 设置key_padding_mask的padding部分为True
                        key_padding_mask[i, length:] = True
                    padded_batch = torch.stack(padded_tensors, dim=0)
                    # 执行MultiheadAttention; 注意: 期望输入形状是(seq_len, batch_size, embed_dim)
                    # 我们首先将padded_batch转置来匹配这种形状
                    keys = padded_batch.transpose(0, 1)  # 维度变成(max_len, batch_size, embed_dim)
                    queries = text_features.unsqueeze(0)
                    attn_output, attn_output_weights = self.attn(queries,keys,keys,key_padding_mask)
                    topic_features = attn_output.view(batch_size,embed_dim)
                else:
                    topic_features = [self.clip.clip.get_text_features(**texts).mean(dim=0) for texts in topic_text_data]
                    topic_features = torch.stack(topic_features, dim=0)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)
            if topic_text_data is not None:
                topic_features = [self.clip.encode_text(**texts).mean(dim=0) for texts in topic_text_data]
                topic_features = torch.stack(topic_features, dim=0)
        if topic_text_data is not None:
            text_features = text_features + topic_features * lambda_coef
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(text_features, video_features)
            
        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled
