import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering


def get_cluster_labels(texts, method, n_clusters):
    if method == 'Kmeans':
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
        # 对特征向量进行聚类
        kmeans.fit(texts)
        # 获取聚类结果，即每个文本样本所属的类别标签
        return kmeans.labels_
    else:
        agg_clust = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        return agg_clust.fit_predict(texts)


def get_clusters(texts, method, n_clusters):
    labels = get_cluster_labels(texts, method, n_clusters)
    cluster_arrays = []
    for c in np.unique(labels):
        cluster_arrays.append(texts[labels == c])
    return cluster_arrays
def aggregate_embeddings(texts):
    return texts.mean(axis=0)

def get_topic_aggregated_embeddings(texts, method, n_clusters):
    labels = get_cluster_labels(texts, method, n_clusters)
    topic_aggregated_embeddings_arrays = []
    for c in np.unique(labels):
        topic_aggregated_embeddings_arrays.append(texts[labels == c].mean(axis=0))
    return topic_aggregated_embeddings_arrays


