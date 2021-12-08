from itertools import product
from typing import Tuple, Any

import pandas as pd
import numpy as np
import random

from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN,
    MeanShift,
    FeatureAgglomeration,
)
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from umap import UMAP
from sys import stderr

import ClusterEnsembles as CE

RANDOM_STATE = 608429167
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def eprint(*args, **kwargs):
    kwargs["file"] = stderr
    print(*args, **kwargs)


def scores(x: ArrayLike, labels):
    s = []
    for score in (silhouette_score, calinski_harabasz_score, davies_bouldin_score):
        try:
            s.append(score(x, np.array(labels)))
        except Exception as e:
            s.append(np.nan)
            eprint(e)
    return s


def cluster(x: ArrayLike, max_n_clusters=7) -> Any:
    all_labels = dict()
    n_cluster_options = [None] + list(range(2, max_n_clusters + 1))
    for n_clusters in n_cluster_options:
        eprint(n_clusters)
        # KMeans
        try:
            name = f"kmeans_{n_clusters}"
            eprint(name)
            labels = KMeans(
                n_clusters=n_clusters, random_state=RANDOM_STATE
            ).fit_predict(x)
            all_labels[name] = labels
        except Exception as e:
            eprint(e)

        # Spectral clustering
        try:
            eprint(f"spectral_{n_clusters if n_clusters else 0}")

            # remove duplicate rows
            xx = np.unique(x, axis=0)
            labels = SpectralClustering(n_clusters=n_clusters).fit_predict(xx)
            all_labels[f"spectral_{n_clusters if n_clusters else 0}"] = labels
        except Exception as e:
            eprint(e)

        # hierarchical
        linkages = ["ward", "complete", "average", "single"]
        affinities = ["euclidean", "l1", "l2", "manhattan", "cosine"]
        for linkage, affinity in product(linkages, affinities):
            try:
                eprint(
                    f"agglomerative_{n_clusters if n_clusters else 0}_{linkage}_{affinity}"
                )
                if n_clusters:
                    labels = AgglomerativeClustering(
                        n_clusters=n_clusters, affinity=affinity, linkage=linkage
                    ).fit_predict(x)
                else:
                    labels = AgglomerativeClustering(
                        n_clusters=None,
                        affinity=affinity,
                        linkage=linkage,
                        distance_threshold=0.5,
                        compute_full_tree=True,
                    ).fit_predict(x)
                all_labels[
                    f"agglomerative_{n_clusters if n_clusters else 0}_{linkage}_{affinity}"
                ] = labels
            except Exception as e:
                eprint(e)

    # AffinityPropagation
    try:
        eprint(f"affinitypropagation")
        labels = AffinityPropagation(random_state=RANDOM_STATE).fit_predict(x)
        all_labels[f"affinitypropagation"] = labels
    except Exception as e:
        eprint(e)

    # MeanShift
    try:
        eprint(f"meanshift")
        labels = MeanShift().fit_predict(x)
        all_labels[f"meanshift"] = labels
    except Exception as e:
        eprint(e)

    # DBSCAN
    try:
        eprint(f"dbscan")
        labels = DBSCAN().fit_predict(x)
        all_labels[f"dbscan"] = labels
    except Exception as e:
        eprint(e)

    return all_labels


def get_input() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "snakemake" in locals() or "snakemake" in globals():
        # beta values between 0 (unmethylated) and 1 (methylated))
        beta = pd.read_csv(snakemake.input.beta, sep="\t", index_col=0)

        # m values between 0 and infinity
        m = pd.read_csv(snakemake.input.m, sep="\t", index_col=0)
    else:
        # beta values between 0 (unmethylated) and 1 (methylated))
        beta = pd.read_csv("beta.tsv", sep="\t", index_col=0)

        # m values between 0 and infinity
        m = pd.read_csv("m.tsv", sep="\t", index_col=0)

    exclude = ["RB_E_061", "RB_E_066", "RB_E_067", "RB_E_069_FR"]
    beta = beta[[c for c in beta.columns if c.startswith("RB")]]
    beta.drop(columns=exclude, inplace=True)
    eprint(beta)
    samples = list(beta.columns)
    features = list(beta.index)
    beta = beta.transpose()
    beta = beta.values

    m = m[[c for c in m.columns if c.startswith("RB")]]
    m.drop(columns=exclude, inplace=True)
    m = m.transpose()
    m = m.values

    # remove rows with singular values
    beta = beta[~np.all(beta[:, 1:] == beta[:, :-1], axis=1), :]

    # remove rows with singular values
    m = m[~np.all(m[:, 1:] == m[:, :-1], axis=1), :]
    new_min = np.nanmin(m[np.isfinite(m)])
    new_max = np.nanmax(m)
    m = np.nan_to_num(m, neginf=new_min, posinf=new_max)

    return samples, features, beta, m


def main():
    samples, features, beta, m = get_input()

    all_labels = []
    transformations = [FeatureAgglomeration, PCA, NMF, FastICA, UMAP, StandardScaler]
    n_components_options = list(range(2, snakemake.params.dim_reduction_max_dim + 1))
    max_n_clusters = snakemake.params.max_n_clusters + 1

    for x in (m, beta):
        for Transform in transformations:
            for n_components in n_components_options:
                eprint(f"{Transform.__name__}\t{n_components}")
                xx = x
                try:
                    transform = Transform(n_components=n_components)
                except:
                    if Transform is FeatureAgglomeration:
                        xx = xx[:, np.argsort(-np.std(xx, axis=0))[:20000]]
                        print(xx.shape)
                        transform = Transform(
                            n_clusters=17,
                            memory="/tmp/consensus",
                        )  # 17 has highest silhouette score
                    else:
                        transform = Transform()
                try:
                    xx = transform.fit_transform(xx)
                    eprint(xx.shape)
                    clusters = cluster(xx, max_n_clusters=max_n_clusters)
                    cluster_df = pd.DataFrame.from_dict(clusters)
                    eprint(cluster_df)
                    # TODO: score on original data (x) or transformed data (xx)?
                    score_df = cluster_df.apply(lambda lbls: scores(xx, lbls))
                    eprint(score_df)
                    score_df["score"] = [
                        "silhouette",
                        "calinski_harabasz",
                        "davies_bouldin",
                    ]
                    score_df.set_index("score", inplace=True)
                    score_df.dropna(axis=1, inplace=True)
                    eprint(score_df)
                    keep = (
                        score_df.transpose()
                        .query(
                            "silhouette > 0 & davies_bouldin < 3 & calinski_harabasz > 3"
                        )
                        .index
                    )
                    cluster_df = cluster_df[keep]
                    label_array = np.array(list(cluster_df.transpose().values))
                    # remove results with only a singular cluster
                    label_array = label_array[
                        ~np.all(label_array[:, 1:] == label_array[:, :-1], axis=1), :
                    ]
                    eprint(label_array)
                    all_labels.append(label_array)
                except Exception as e:
                    eprint(e)

    all_labels = np.vstack(all_labels)

    label_df = pd.DataFrame(columns=samples, data=all_labels)
    label_df.to_csv(snakemake.output.labels, sep="\t", index=False)

    all_consensus_labels = []
    for nclass in [None] + list(range(2, max_n_clusters)):
        consensus_labels = CE.cluster_ensembles(
            all_labels, nclass=nclass, random_state=RANDOM_STATE, verbose=True
        )
        all_consensus_labels.append(consensus_labels)
    eprint(all_consensus_labels)
    consensus_df = pd.DataFrame(columns=samples, data=all_consensus_labels)
    consensus_df.to_csv(snakemake.output.clusters, sep="\t", index=False)


if __name__ == "__main__":
    main()
