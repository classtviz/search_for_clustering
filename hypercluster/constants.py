import numpy as np
from sklearn.metrics.pairwise import (
    cosine_distances,
    euclidean_distances,
    manhattan_distances,
    additive_chi2_kernel,
    chi2_kernel,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    laplacian_kernel,
    sigmoid_kernel,
    cosine_similarity,
)
from tslearn.metrics import (
    cdist_dtw,
    cdist_ctw,
    cdist_soft_dtw_normalized,
    cdist_gak,
)
from scipy.spatial.distance import correlation, pdist, squareform


def pdist_corr(d1):
    return squareform(pdist(d1, metric=correlation))


param_delim = ";"
val_delim = "-"

slow = ["AffinityPropagation", "MeanShift"]
fast = ["KMeans", "OPTICS", "HDBSCAN"]
fastest = ["MiniBatchKMeans"]
partitioners = ["AffinityPropagation", "MeanShift", "KMeans", "MiniBatchKMeans"]
clusterers = ["OPTICS", "HDBSCAN"]
categories = {
    "slow": slow,
    "fast": fast,
    "fastest": fastest,
    "partitioning": partitioners,
    "clustering": clusterers,
}

min_cluster_size = [i for i in range(2, 150, 3)]
n_clusters = [i for i in range(2, 50)]
damping = [i / 100 for i in range(55, 95, 5)]
resolutions = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
knn = [20, 30, 60]

PAIRWISE_DISTANCE_FUNCTIONS = {
    "additive_chi2": lambda x: -additive_chi2_kernel(x),
    "cosine_dist": cosine_distances,
    "euclidean": euclidean_distances,
    "manhattan": manhattan_distances,
    "cdist_dtw": cdist_dtw,
    "cdist_ctw": cdist_ctw,
    "cdist_soft_dtw_normalized": cdist_soft_dtw_normalized,
    "correlation": pdist_corr,
}

PAIRWISE_KERNEL_FUNCTIONS = {
    "chi2": chi2_kernel,
    "linear": linear_kernel,
    "polynomial": polynomial_kernel,
    "rbf": rbf_kernel,
    "laplacian": laplacian_kernel,
    "sigmoid": sigmoid_kernel,
    "cosine_sim": cosine_similarity,
    "gak": cdist_gak,
}

distance_metrics = [
    "additive_chi2",
    "cosine_dist",
    "euclidean",
    "manhattan",
    "cdist_dtw",
    "cdist_ctw",
    "cdist_soft_dtw_normalized",
    "correlation",
]

kernel_metrics = [
    "chi2",
    "linear",
    "polynomial",
    "rbf",
    "laplacian",
    "sigmoid",
    "cosine_sim",
    "gak",
]

import hdbscan

hdbscan.HDBSCAN()

variables_to_optimize = {
    "HDBSCAN": dict(
        min_cluster_size=min_cluster_size,
        min_samples=list(range(2, 30, 3)),
    ),
    "AffinityPropagation": dict(damping=damping, affinity=kernel_metrics),
    "AgglomerativeClustering": dict(
        n_clusters=n_clusters,
        affinity=distance_metrics,
        linkage=["average", "single", "complete", "ward"],
    ),
    "Birch": dict(
        threshold=np.linspace(0.1, 0.99, num=10),
        branching_factor=list(range(10, 101, 10)),
        n_clusters=n_clusters,
    ),
    "DBSCAN": dict(
        eps=np.linspace(0.01, 1.0, num=10),
        metric=distance_metrics,
    ),
    "KMeans": dict(n_clusters=n_clusters),
    "MiniBatchKMeans": dict(n_clusters=n_clusters),
    "BisectingKMeans": dict(n_clusters=n_clusters),
    "MeanShift": dict(cluster_all=[False]),
    "OPTICS": dict(
        min_samples=min_cluster_size,
        metric=distance_metrics,
    ),
    "NMFCluster": dict(n_clusters=n_clusters),
    "SpectralClustering": dict(
        n_clusters=n_clusters,
        affinity=[k for k in kernel_metrics if k != "additive_chi2"],
    ),
    "KShape": dict(n_clusters=n_clusters),
    "KernelKMeans": dict(
        n_clusters=n_clusters,
        kernel=[k if k != "cosine_sim" else "cosine" for k in kernel_metrics],
    ),
    "TimeSeriesKMeans": dict(
        n_clusters=n_clusters, metric=["euclidean", "dtw", "softdtw"]
    ),
}


clusterers_w_precomputed = {
    "AffinityPropagation",
    "AgglomerativeClustering",
    "DBSCAN",
    "OPTICS",
    "SpectralClustering",
}


need_ground_truth = [
    "adjusted_rand_score",
    "adjusted_mutual_info_score",
    "homogeneity_score",
    "completeness_score",
    "fowlkes_mallows_score",
    "mutual_info_score",
    "v_measure_score",
]

inherent_metrics = [
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "smallest_largest_clusters_ratio",
    "number_of_clusters",
    "smallest_cluster_size",
    "largest_cluster_size",
    "fraction_clustered",
]

min_or_max = {
    "adjusted_rand_score": "max",
    "adjusted_mutual_info_score": "max",
    "homogeneity_score": "max",
    "completeness_score": "max",
    "fowlkes_mallows_score": "max",
    "silhouette_score": "max",
    "calinski_harabasz_score": "max",
    "davies_bouldin_score": "min",
    "mutual_info_score": "max",
    "v_measure_score": "max",
}

pdist_adjacency_methods = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulsinski",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]


valid_partition_types = [
    "RBConfigurationVertexPartition",
    "ModularityVertexPartition",
    "RBERVertexPartition",
    "CPMVertexPartition",
    "SignificanceVertexPartition",
    "SurpriseVertexPartition",
]
