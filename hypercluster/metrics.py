from typing import Iterable
from collections import Counter

__doc__ = (
    "More functions for evaluating clustering results. Additional metric evaluations can "
    "be added here, as long as the second argument is the labels to evaluate"
)


def number_clustered(_, labels: Iterable) -> float:
    """

    Args:
        labels: Vector of sample labels.

    Returns: The number of clustered labels.

    """
    return (labels != -1).sum()


def smallest_largest_clusters_ratio(_, labels: Iterable) -> float:
    """

    Args:
        labels: Vector of sample labels.

    Returns:  Ratio of number of members in smallest over largest cluster.

    """
    counts = Counter(labels)
    counts.pop(-1, None)
    return min(counts.values()) / max(counts.values())


def smallest_cluster_ratio(_, labels: Iterable) -> float:
    """

    Args:
        labels: Vector of sample labels.

    Returns:  Ratio of number of members in smallest over all samples.

    """
    counts = Counter(labels)
    counts.pop(-1, None)
    return min(counts.values()) / len(labels)


def number_of_clusters(_, labels: Iterable) -> float:
    return len(Counter(labels))


def smallest_cluster_size(_, labels: Iterable) -> float:
    return min(Counter(labels).values())


def largest_cluster_size(_, labels: Iterable) -> float:
    return max(Counter(labels).values())
