import random

import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPTokenizer
import numpy as np
import ujson as json
from collections import defaultdict
import pandas as pd
import argparse
from tqdm import tqdm
from modules.cluster import get_cluster_labels


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def random_pick_out(list):
    item = random.choice(list)
    list.remove(item)
    return item


def plot_figure(data1, data2, dataset, sample_num):
    plt.rcParams['font.sans-serif'] = ['FangSong']
    # 绘制堆叠柱状图
    x = np.arange(len(data1))
    width = 0.6
    fig, ax = plt.subplots(figsize=(12, 7))
    # 使用zip结合两个列表，然后根据a进行主要排序，根据b进行次要排序
    combined = sorted(zip(data1, data2), key=lambda x: (x[0]+x[1], x[0]),reverse=True)
    # 解压排序后的列表
    a_sorted, b_sorted = zip(*combined)
    ax.bar(x, a_sorted, width, label='次要主题文本数')
    ax.bar(x, b_sorted, width, bottom=a_sorted, label='主要主题文本数')
    ax.set_xlabel('视频序号', fontsize=28)
    ax.set_ylabel('文本数', fontsize=28)
    ax.set_title(dataset + '数据集' + str(sample_num) + '视频样本规模数据分布', fontsize=28)
    ax.legend(fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # 显示图形
    plt.show()


class DataJson:
    def __init__(self, dataset):
        self.dataset = dataset
        if self.dataset == 'MSRVTT':
            self.db = load_json('data/MSRVTT/MSRVTT_data.json')
            self.video_ids = [item['video_id'] for item in self.db['videos']]
        else:
            self.db = load_json('data/MSVD/captions_msvd.json')
            self.video_ids = [key for key in self.db.keys()]

    def get_video_ids(self):
        return self.video_ids

    def get_texts(self, video_id):
        if self.dataset == 'MSRVTT':
            return [item['caption'] for item in self.db['sentences'] if item['video_id'] == video_id]
        else:
            return self.db[video_id]


if __name__ == '__main__':
    # 选择数据集
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT')
    parser.add_argument('-s', '--samples', type=int, default=500)
    args = parser.parse_args()
    dataset = args.dataset
    samples = args.samples
    method = 'Kmeans'
    # 样本筛选的标准倍数
    filter_multiplier = 1 if dataset == "MSRVTT" else 2
    # 使用预训练模型
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    data = DataJson(dataset)
    video_ids = data.get_video_ids()
    df = pd.DataFrame({'video_id': [], 'a1': [], 'a2': [], 'b1': [], 'b2': []})
    picked_samples = 0
    pbar = tqdm(total=samples)
    main_topic_num = []
    non_main_topic_num = []
    while picked_samples < samples and picked_samples < len(video_ids):
        video_id = random_pick_out(video_ids)
        texts = data.get_texts(video_id)
        texts_tokenized = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        text_features = model.get_text_features(**texts_tokenized)
        text_features = text_features.detach().numpy()
        cluster_labels = get_cluster_labels(text_features, method, 2)
        clusters = defaultdict(list)
        for i, sentence in enumerate(texts):
            cluster_label = cluster_labels[i]
            clusters['topic' + str(cluster_label)].append(sentence)
        if len(clusters['topic0']) > len(clusters['topic1']):
            main_topic = clusters['topic0']
            non_main_topic = clusters['topic1']
        else:
            main_topic = clusters['topic1']
            non_main_topic = clusters['topic0']
        if 2 * filter_multiplier <= len(non_main_topic) <= 6 * filter_multiplier and len(non_main_topic) * 2 <= len(
                main_topic) and (dataset == 'MSRVTT' or 25 <= len(non_main_topic) + len(main_topic) <= 60):
            non_main_topic_num.append(len(non_main_topic))
            main_topic_num.append(len(main_topic))
            dataframe = {'video_id': video_id, 'a1': random_pick_out(main_topic), 'a2': random_pick_out(main_topic),
                         'b1': random_pick_out(non_main_topic), 'b2': random_pick_out(non_main_topic)}
            dataframe = pd.DataFrame([dataframe])
            df = pd.concat([df, dataframe], ignore_index=True)
            picked_samples += 1
            pbar.update(1)
    dir = 'data/' + dataset
    cluster_csv = dir + '/cluster_' + str(samples) + '.csv'
    df.to_csv(cluster_csv)
    plot_figure(non_main_topic_num, main_topic_num, dataset, samples)
