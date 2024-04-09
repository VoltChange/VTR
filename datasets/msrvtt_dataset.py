import os
import cv2
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class MSRVTTDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        db_file = 'data/MSRVTT/MSRVTT_data.json'
        test_csv = 'data/MSRVTT/MSRVTT_JSFUSION_test.csv'
        # 主题聚类测试数据集
        if split_type == 'topic_test':
            test_csv = 'data/MSRVTT/cluster_'+str(config.samples)+'_'+config.topic+'.csv'
        if config.msrvtt_train_file == '7k':
            train_csv = 'data/MSRVTT/MSRVTT_train.7k.csv'
        else:
            train_csv = 'data/MSRVTT/MSRVTT_train.9k.csv'
        db_file = 'data/MSRVTT/topic_captions.json'
        self.vid2caption = load_json(db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df['video_id'].unique()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(test_csv)

            
    def __getitem__(self, index):
        video_path, caption, video_id ,topic_text= self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)
        if self.split_type =='train':
            return {
                'video_id': video_id,
                'video': imgs,
                'text': caption,
                'topic_text': topic_text
            }
        else:
            return {
                'video_id': video_id,
                'video': imgs,
                'text': caption,
            }
    
    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)


    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            vid, caption_with_topic = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            topic_text = self.vid2caption[vid][caption_with_topic['topic']]
            return video_path, caption_with_topic['caption'], vid , topic_text
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            caption = self.test_df.iloc[index].sentence
            return video_path, caption, vid,None


    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for topic_idx,captions in enumerate(self.vid2caption[vid]):
                    for caption in captions:
                        self.all_train_pairs.append([vid,{'caption':caption,'topic':topic_idx}])
