import random
from itertools import product
from sys import stderr
from typing import Any, List
from multiprocessing import Pool

import ClusterEnsembles as CE
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN,
    MeanShift,
    FeatureAgglomeration,
)
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    Normalizer,
    MinMaxScaler,
    MaxAbsScaler,
)
from umap import UMAP

RANDOM_STATE = snakemake.params.get("random_state", 608429167)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def eprint(*args, **kwargs):
    kwargs["file"] = stderr
    print(*args, **kwargs)


# fake tuple for additional params, otherwise pandas multiindex slicing/loc/xs breaks
class T(object):
    def __init__(self, values):
        self._values = values

    def __str__(self):
        return str(self._values)

    def __repr__(self):
        return repr(self._values)

    def __hash__(self):
        return hash(self._values)

    def __eq__(self, other):
        return self._values == other._values


def scores(x: ArrayLike, labels):
    s = []
    score_options = (silhouette_score, calinski_harabasz_score, davies_bouldin_score)
    for score in score_options:
        try:
            s.append(score(x, np.array(labels)))
        except Exception as e:
            s.append(np.nan)
            eprint(e)
    return s


def cluster(x: ArrayLike, samples: List[str], max_n_clusters=7) -> Any:
    all_labels = pd.DataFrame(
        columns=["cluster_method", "n_clusters", "additional_params"] + samples
    )
    all_labels.set_index(
        ["cluster_method", "n_clusters", "additional_params"], inplace=True
    )
    n_cluster_options = [None] + list(range(2, max_n_clusters + 1))
    for n_clusters in n_cluster_options:
        # KMeans
        try:
            labels = KMeans(
                n_clusters=n_clusters, random_state=RANDOM_STATE
            ).fit_predict(x)
            all_labels.loc[("kmeans", n_clusters, T(()))] = labels
        except Exception as e:
            eprint(e)

        # Spectral clustering
        try:
            # remove duplicate rows
            xx = np.unique(x, axis=0)
            labels = SpectralClustering(n_clusters=n_clusters).fit_predict(xx)
            all_labels.loc[("spectral", n_clusters if n_clusters else 0, ())] = labels
        except Exception as e:
            eprint(e)

        # hierarchical
        linkages = ["ward", "complete", "average", "single"]
        affinities = ["euclidean", "l1", "l2", "manhattan", "cosine"]
        for linkage, affinity in product(linkages, affinities):
            try:
                if n_clusters:
                    labels = AgglomerativeClustering(
                        n_clusters=n_clusters, affinity=affinity, linkage=linkage
                    ).fit_predict(x)
                    all_labels.loc[
                        (
                            "agglomerative",
                            n_clusters if n_clusters else 0,
                            T((linkage, affinity)),
                        )
                    ] = labels
            except Exception as e:
                eprint(e)

    # AffinityPropagation
    try:
        labels = AffinityPropagation(random_state=RANDOM_STATE).fit_predict(x)
        all_labels.loc[("affinitypropagation", 0, T(()))] = labels
    except Exception as e:
        eprint(e)

    # MeanShift
    try:
        labels = MeanShift().fit_predict(x)
        all_labels.loc[("meanshift", 0, T(()))] = labels
    except Exception as e:
        eprint(e)

    # DBSCAN
    try:
        labels = DBSCAN().fit_predict(x)
        all_labels.loc[("dbscan", 0, T(()))] = labels
    except Exception as e:
        eprint(e)

    return all_labels


def get_input():
    matrices = {
        path: pd.read_csv(path, sep="\t", index_col=[0, 1])
        for path in snakemake.input.data_matrices
    }

    samples = {key: list(mat.columns) for key, mat in matrices.items()}
    features = {key: list(mat.index) for key, mat in matrices.items()}

    for key, mat in matrices.items():
        mat = mat.transpose()
        mat = mat.values

        # remove rows with singular values
        mat = mat[~np.all(mat[:, 1:] == mat[:, :-1], axis=1), :]

        if np.isnan(mat).any() or np.isinf(mat).any():
            new_min = np.nanmin(mat[np.isfinite(mat)])
            new_max = np.nanmax(mat)
            mat = np.nan_to_num(mat, neginf=new_min, posinf=new_max)

        mat = pd.DataFrame(data=mat, columns=features[key], index=samples[key])
        eprint(mat)
        matrices[key] = mat

    return matrices


def get_transformations():
    transformation_names = snakemake.params.get(
        "transformations",
        ["pca", "nmf", "ica", "umap", "normalize", "power", "robust", "quantile"],
    )
    transformations = []
    for t in transformation_names:
        t = t.lower()
        if t == "pca":
            transformations.append(PCA)
        elif t == "nmf":
            transformations.append(NMF)
        elif t == "ica" or t == "fastica":
            transformations.append(FastICA)
        elif t == "umap":
            transformations.append(UMAP)
        elif t == "standardize" or t == "standardscaler":
            transformations.append(StandardScaler)
        elif t == "normalize" or t == "normalizer":
            transformations.append(Normalizer)
        elif t == "power" or t == "powertransform":
            transformations.append(PowerTransformer)
        elif t == "robust" or t == "robustscaler":
            transformations.append(RobustScaler)
        elif t == "quantile" or t == "quantiletransformer":
            transformations.append(QuantileTransformer)
        elif t == "minmax" or t == "minmaxscaler":
            transformations.append(MinMaxScaler)
        elif t == "maxabs" or t == "maxabsscaler":
            transformations.append(MaxAbsScaler)
        else:
            raise ValueError(f"Unknown transformation: {t}")

    return transformations


def do_clustering(matrix, Transform, n_components, max_n_clusters):
    _path, matrix = matrix
    x = matrix.values
    samples = list(matrix.index)
    xx = x
    try:
        transform = Transform(n_components=n_components)
    except:
        if Transform is FeatureAgglomeration:
            # if the number of features is very high, feature agglomeration is costly (memory-wise)
            # we're selecting the most variable features here, which is biased,
            # perhaps consider doing multiple random subsamples instead?
            num_selected_features = 16536
            xx = xx[:, np.argsort(-np.std(xx, axis=0))[:num_selected_features]]
            transform = Transform(
                n_clusters=max(2, int(np.ceil(np.log2(num_selected_features)))),
                memory="/tmp/consensus",
            )
        else:
            transform = Transform()
    try:
        xx = transform.fit_transform(xx)
        cluster_df = cluster(xx, samples, max_n_clusters=max_n_clusters)
        # TODO: score on original data (x) or transformed data (xx)?
        score_df = cluster_df.apply(lambda lbls: scores(xx, lbls), axis=1)
        score_df = pd.DataFrame(
            index=score_df.index,
            columns=["silhouette", "calinski_harabasz", "davies_bouldin"],
            data=score_df.tolist(),
        )
        return cluster_df, score_df
    except Exception as e:
        eprint(e)
        return None, None


def main():
    matrices = get_input()

    transformations = get_transformations()
    n_components_options = list(
        range(2, snakemake.params.get("dim_reduction_max_dim", 2) + 1)
    )
    max_n_clusters = snakemake.params.get("max_n_clusters", 2) + 1

    options = product(
        matrices.items(), transformations, n_components_options, [max_n_clusters]
    )
    with Pool(snakemake.threads) as pool:
        results = [
            (a, b)
            for a, b in pool.starmap(do_clustering, options)
            if a is not None and b is not None
        ]

    all_clusters, all_scores = zip(*results)
    all_clusters = pd.concat(all_clusters)
    all_scores = pd.concat(all_scores)

    # drop entries with score of nan
    all_scores.dropna(axis=0, inplace=True)
    # and keep only thos with silhouette larger than 0
    all_scores = all_scores.query("silhouette > 0.0")
    all_clusters = all_clusters.loc[all_scores.index]

    all_clusters.to_csv(snakemake.output.labels, sep="\t", index=True)

    all_consensus_labels = []
    index = []
    for nclass in [None] + list(range(2, max_n_clusters)):
        labels = all_clusters.values.astype(dtype=int)
        consensus_labels = CE.cluster_ensembles(
            labels, nclass=nclass, random_state=RANDOM_STATE, verbose=True
        )
        all_consensus_labels.append(consensus_labels)
        index.append(f"{len(set(consensus_labels))}_clusters")

    samples = list(all_clusters.columns)
    consensus_df = pd.DataFrame(columns=samples, data=all_consensus_labels, index=index)
    consensus_df.to_csv(snakemake.output.clusters, sep="\t", index=True)


if __name__ == "__main__":
    main()
