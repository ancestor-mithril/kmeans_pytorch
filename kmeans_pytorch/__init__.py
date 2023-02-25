import numpy as np
import torch
from .soft_dtw_cuda import SoftDTW


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        max_iterations = 10,
        gamma_for_soft_dtw=0.001
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    match distance:
        case 'euclidean':
            pairwise_distance_function = pairwise_distance
        case 'cosine':
            pairwise_distance_function = pairwise_cosine
        case 'soft_dtw':
            sdtw = SoftDTW(use_cuda=(x.device.type == 'cuda'), gamma=gamma_for_soft_dtw)
            pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw)
        case _:
            raise NotImplementedError

    # initialize
    initial_state = initialize(X, num_clusters)

    iteration = 0
    while iteration < max_iterations:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        # increment iteration
        iteration = iteration + 1
    
    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        gamma_for_soft_dtw=0.001
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor) cluster ids
    """
    match distance:
        case 'euclidean':
            pairwise_distance_function = pairwise_distance
        case 'cosine':
            pairwise_distance_function = pairwise_cosine
        case 'soft_dtw':
            sdtw = SoftDTW(use_cuda=(x.device.type == 'cuda'), gamma=gamma_for_soft_dtw)
            pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw)
        case _:
            raise NotImplementedError

    # convert to float
    X = X.float()

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2):

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2):

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

