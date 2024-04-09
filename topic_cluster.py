
import random

import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPTokenizer
import numpy as np
import ujson as json
from collections import defaultdict
import pandas as pd
import argparse
from tqdm import tqdm
from modules.cluster import get_cluster_labels, get_clusters


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data,f)
class DataJson:
    def __init__(self, dataset):
        self.dataset = dataset
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        if self.dataset == 'MSRVTT':
            self.db = load_json('data/MSRVTT/MSRVTT_data.json')
            self.video_ids = [item['video_id'] for item in self.db['videos']]
            self.vid2caption = defaultdict(list)
            for annotation in self.db['sentences']:
                caption = annotation['caption']
                vid = annotation['video_id']
                self.vid2caption[vid].append(caption)
        else:
            self.db = load_json('data/MSVD/captions_msvd.json')
            self.video_ids = [key for key in self.db.keys()]
    def cluster(self):
        for video_id, captions in tqdm(self.vid2caption.items()):
            method = 'agg'
            texts_tokenized = self.tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
            text_features = self.model.get_text_features(**texts_tokenized)
            text_features = text_features.detach().numpy()
            clusters_labels = get_cluster_labels(text_features, method, 3)
            clusters = []
            for c in np.unique(clusters_labels):
                clusters.append([item for idx,item in enumerate(captions) if clusters_labels[idx]==c])
            self.vid2caption[video_id] = clusters
    def save(self):
        save_json('data/'+self.dataset+'/topic_captions.json', self.vid2caption)

if __name__ == '__main__':
    # 选择数据集
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT')
    args = parser.parse_args()
    dataset = args.dataset
    # 样本筛选的标准倍数
    # 使用预训练模型
    data = DataJson(dataset)
    data.cluster()
    data.save()