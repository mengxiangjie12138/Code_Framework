import numpy as np
from sklearn.neighbors import kneighbors_graph


def distance(features_1, features_2):
    dist = np.sqrt(np.sum(np.square(features_1 - features_2)))
    return dist


def from_a_get_d(adj):
    A = np.asmatrix(adj)
    I = np.eye(len(A))
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = D_hat**0.5
    D_hat = np.matrix(np.diag(D_hat))
    D_hat = D_hat**-1
    return A_hat, D_hat


def knn_graph(features, k, include_self=True):
    return kneighbors_graph(features, k, include_self=include_self)


# 限制ϵ半径建图算法中的ϵ
def from_labels_get_radius(features, labels, class_number):
    distance_list = np.zeros(class_number)
    for index1, feature1 in enumerate(features):
        for index2, feature2 in enumerate(features):
            if labels[index1] == labels[index2]:
                distance_list[labels[index1]] += distance(features[index1], features[index2])
    count = (len(features) * (len(features) - 1)) / 2
    radius_limit = np.mean(distance_list / count)
    return radius_limit


# ϵ半径建图
def radius_graph(features, dis, labels=None, class_number=None, include_self=True, limit_radius=False):
    if limit_radius:
        dis = from_labels_get_radius(features, labels, class_number)
    adjacency_matrix = np.zeros((len(features), len(features)))
    for index1, feature1 in enumerate(features):
        for index2, feature2 in enumerate(features):
            if dis >= distance(feature1, feature2) and index1 != index2:
                adjacency_matrix[index1, index2] = 1
            elif dis >= distance(feature1, feature2) and index1 == index2 and include_self:
                adjacency_matrix[index1, index2] = 1
    return adjacency_matrix


# knn与ϵ半径组合建图
def knn_and_radius_graph(features, k, dis, labels=None, class_number=None, include_self=True, limit_radius=False):
    adjacency_matrix_knn = knn_graph(features, k, include_self)
    if limit_radius:
        dis = from_labels_get_radius(features, labels, class_number)
    adjacency_matrix_radius = radius_graph(features, dis, include_self)
    adjacency_matrix = np.zeros((len(features), len(features)))
    for index in range(len(features)):
        radius_node_number = np.sum(adjacency_matrix_radius[index])
        if radius_node_number > k:
            adjacency_matrix[index] = adjacency_matrix_radius[index]
        else:
            adjacency_matrix[index] = adjacency_matrix_knn[index]

    return adjacency_matrix


# 全连接建图
def fully_connected_graph(features_len):
    return np.ones((features_len, features_len))


# k-means建图
def k_means_graph(centers, features, labels):
    adjacency_matrix = np.zeros((len(labels), len(labels)))
    centers_index = np.zeros(len(centers))
    centers_label = np.zeros(len(centers))
    centers_distance = np.zeros(len(centers)) + 100000
    for center_index, center in enumerate(centers):
        for index, feature in enumerate(features):
            dis = distance(center, feature)
            if centers_distance[center_index] > dis:
                centers_distance[center_index] = dis
                centers_index[center_index] = index
                centers_label[center_index] = labels[index]

    for center_index in range(len(centers)):
        for index, label in enumerate(labels):
            if label == centers_label[center_index]:
                adjacency_matrix[centers_index[center_index], index] = 1
                adjacency_matrix[index, centers_index[center_index]] = 1

    return adjacency_matrix













