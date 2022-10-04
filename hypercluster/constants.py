import numpy as np
from sklearn.cluster import *
from sklearn.metrics.pairwise import *
from tslearn.metrics import (
    cdist_dtw,
    cdist_ctw,
    cdist_soft_dtw,
    cdist_soft_dtw_normalized,
    cdist_gak,
)
from tslearn.clustering import *
from .additional_clusterers import *
from .additional_metrics import *


__doc__ = """
Attributes: 
    param_delim: delimiter between hyperparameters for snakemake file labels and labels DataFrame \
    columns.  
    val_delim: delimiter between hyperparameter label and value for snakemake file labels and \
    labels DataFrame columns.  
    categories: Convenient groups of clusterers to use. If all samples need to be clustered, \
    'partitioners' is a good choice. If there are millions of samples, 'fastest' might be a good \
    choice.    
    variables_to_optimize: Some default hyperparameters to optimize and value ranges for a \
    selection of commonly used clustering algoirthms from sklearn. Used as deafults for \
    clustering.AutoClusterer and clustering.optimize_clustering.    
    need_ground_truth: list of sklearn metrics that need ground truth labeling. \
    "adjusted_rand_score", "adjusted_mutual_info_score", "homogeneity_score", \
    "completeness_score", "fowlkes_mallows_score", "mutual_info_score", "v_measure_score"    
    inherent_metrics: list of sklearn metrics that need original data for calculation. \
    "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", \
    "smallest_largest_clusters_ratio", "number_of_clusters", "smallest_cluster_size", \
    "largest_cluster_size"  
    min_or_max: establishing whether each sklearn metric is better when minimized or maximized for \
    clustering.pick_best_labels.  
"""
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

min_cluster_size = [i for i in range(2, 50, 1)]
n_clusters = [i for i in range(2, 50)]
damping = [i / 100 for i in range(55, 95, 5)]
resolutions = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
knn = [20, 30, 60]

PAIRWISE_DISTANCE_FUNCTIONS = {
    "cityblock": manhattan_distances,
    "cosine": cosine_distances,
    "euclidean": euclidean_distances,
    "haversine": haversine_distances,
    "l2": euclidean_distances,
    "l1": manhattan_distances,
    "manhattan": manhattan_distances,
    "cdist_dtw": cdist_dtw,
    "cdist_ctw": cdist_ctw,
    "cdist_soft_dtw": cdist_soft_dtw,
    "cdist_soft_dtw_normalized": cdist_soft_dtw_normalized,
}

PAIRWISE_KERNEL_FUNCTIONS = {
    "additive_chi2": additive_chi2_kernel,
    "chi2": chi2_kernel,
    "linear": linear_kernel,
    "polynomial": polynomial_kernel,
    "poly": polynomial_kernel,
    "rbf": rbf_kernel,
    "laplacian": laplacian_kernel,
    "sigmoid": sigmoid_kernel,
    "cosine": cosine_similarity,
    "gak": cdist_gak,
}

distance = [
    "cityblock",
    "cosine",
    "euclidean",
    "l1",
    "l2",
    "manhattan",
    "cdist_dtw",
    "cdist_ctw",
    "cdist_soft_dtw",
    "cdist_soft_dtw_normalized",
]

kernel = [
    "additive_chi2",
    "chi2",
    "linear",
    "poly",
    "polynomial",
    "rbf",
    "laplacian",
    "sigmoid",
    "cosine",
    "gak",
]

variables_to_optimize = {
    "AffinityPropagation": dict(
        damping=damping, affinity=kernel, func_dict=PAIRWISE_KERNEL_FUNCTIONS
    ),
    "AgglomerativeClustering": dict(
        n_clusters=n_clusters,
        affinity=distance,
        linkage=["average", "single", "complete", "ward"],
        func_dict=PAIRWISE_DISTANCE_FUNCTIONS,
    ),
    "Birch": dict(
        threshold=np.linspace(0.1, 0.99, num=10),
        branching_factor=list(range(10, 101, 10)),
        n_clusters=n_clusters,
    ),
    "DBSCAN": dict(
        eps=np.linspace(0.01, 1.0, num=10),
        metric=distance,
        func_dict=PAIRWISE_DISTANCE_FUNCTIONS,
    ),
    "KMeans": dict(n_clusters=n_clusters),
    "MiniBatchKMeans": dict(n_clusters=n_clusters),
    "BisectingKMeans": dict(n_clusters=n_clusters),
    "MeanShift": dict(cluster_all=[False]),
    "OPTICS": dict(
        min_samples=min_cluster_size,
        metric=distance,
        func_dict=PAIRWISE_DISTANCE_FUNCTIONS,
    ),
    "NMFCluster": dict(n_clusters=n_clusters),
    "SpectralClustering": dict(
        n_clusters=n_clusters, affinity=kernel, func_dict=PAIRWISE_KERNEL_FUNCTIONS
    ),
    "SpectralBiclustering": dict(
        n_clusters=n_clusters, method=["bistochastic", "scale", "log"]
    ),
    "SpectralCoclustering": dict(n_clusters=n_clusters),
    "KShape": dict(n_clusters=n_clusters),
    "KernelKMeans": dict(n_clusters=n_clusters, kernel=kernel + ["gak"]),
    "TimeSeriesKMeans": dict(
        n_clusters=n_clusters, metric=["euclidean", "dtw", "softdtw"]
    ),
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
