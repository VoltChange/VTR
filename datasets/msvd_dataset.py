import os

import pandas as pd

from modules.basic_utils import load_json, read_lines
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class MSVDDataset(Dataset):
    """
        videos_dir: directory where all videos are stored
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
        Notes: for test split, we return one video, caption pair for each caption belonging to that video
               so when we run test inference for t2v task we simply average on all these pairs.
    """

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        db_file = 'data/MSVD/captions_msvd.json'
        test_file = 'data/MSVD/test_list.txt'
        train_file = 'data/MSVD/train_list.txt'
        self.vid2caption = load_json(db_file)

        if split_type == 'train':
            self.train_vids = read_lines(train_file)
            self._construct_all_train_pairs()
        elif split_type == 'test':
            self.test_vids = read_lines(test_file)
            self._construct_all_test_pairs()
        else:
            test_csv = 'data/MSVD/cluster_' + str(config.samples) + '_' + config.topic + '.csv'
            self.test_df = pd.read_csv(test_csv)

    def __getitem__(self, index):
        if self.split_type == 'train':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_train(index)
        elif self.split_type == 'test':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_test(index)
        else:
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_topic_test(index)

        imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                         self.config.num_frames,
                                                         self.config.video_sample_type)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        ret = {
            'video_id': video_id,
            'video': imgs,
            'text': caption
        }

        return ret

    def _get_vidpath_and_caption_by_index_train(self, index):
        vid = self.train_vids[index]
        captions = self.vid2caption[vid]
        caption = ""
        for idx,item in enumerate(captions):
            if idx == 0:
                caption = item
            else:
                caption = caption + '<sep>' + item
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        return video_path, caption, vid

    def _get_vidpath_and_caption_by_index_test(self, index):
        vid, caption = self.all_test_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        return video_path, caption, vid

    def _get_vidpath_and_caption_by_index_topic_test(self, index):
        vid = self.test_df.iloc[index].video_id
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        caption = self.test_df.iloc[index].sentence
        return video_path, caption, vid

    def __len__(self):
        if self.split_type == 'train':
            return len(self.train_vids)
        elif self.split_type == 'test':
            return len(self.all_test_pairs)
        return len(self.test_df)

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        for vid in self.train_vids:
            for caption in self.vid2caption[vid]:
                self.all_train_pairs.append([vid, caption])

    def _construct_all_test_pairs(self):
        self.all_test_pairs = []
        for vid in self.test_vids:
            for caption in self.vid2caption[vid]:
                self.all_test_pairs.append([vid, caption])