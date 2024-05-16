import torch
import math
from model.model_factory import ModelFactory
from transformers import CLIPTokenizer
import torch.nn.functional as F
from modules.metrics import sim_matrix_training


class Collator:
    def __init__(self,config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = ModelFactory.get_model(config).to(self.device)
        self.model.eval()
        self.config = config
        checkpoint_path = config.model_path
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    def sort(self,text,video_imgs):
        text_data = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        video_data = torch.stack(video_imgs).to(self.device)
        data ={'text':text_data,'video':video_data}
        text_embed, vid_embed, vid_embed_pooled = self.model(data, return_all_frames=True)
        sims_batch = sim_matrix_training(text_embed, vid_embed_pooled, 'transformer')
        sims_batch = sims_batch.squeeze().cpu()
        print(sims_batch)
        rank_list = torch.argsort(sims_batch,descending=True)
        return rank_list
    @staticmethod
    def attention_weights(attention,text_embeds, video_embeds):
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = attention.q_proj(text_embeds)
        q = q.reshape(num_texts, attention.num_heads, attention.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = attention.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, attention.num_heads, attention.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = attention.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, attention.num_heads, attention.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(attention.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)
        return attention_weights
    def get_attention_weights(self,text,video_imgs):
        video_data = torch.stack(video_imgs).to(self.device)
        text_data = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        batch_size = video_data.shape[0]

        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        text_features = self.model.clip.get_text_features(**text_data)
        video_features = self.model.clip.get_image_features(video_data)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        transformer = self.model.pool_frames
        text_embeds = transformer.layer_norm1(text_features)
        video_embeds = transformer.layer_norm1(video_features)

        attention = transformer.cross_attn

        attention_weights = Collator.attention_weights(attention,text_embeds,video_embeds)
        return attention_weights