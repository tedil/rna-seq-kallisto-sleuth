
rule threshold_graph:
    input:
        consensus="results/cluster/{model}.consensus_clusters.tsv",
        clusterings="results/cluster/{model}.cluster_labels.tsv",
    output:
        plot=report(
            "results/cluster/{model}.consensus_threshold_graph.html",
            category="Cluster",
        ),
    # params:
    #    metadata="",
    conda:
        "../envs/clusters.yml"
    script:
        "../scripts/threshold_graph.py"


rule consensus_clusters:
    input:
        data_matrices=["results/tables/logcount-matrix/{model}.logcount-matrix.tsv"],
    output:
        clusters="results/cluster/{model}.consensus_clusters.tsv",
        labels="results/cluster/{model}.cluster_labels.tsv",
    threads: 24
    params:
        # for algorithms that support it, generate clusterings with n_clusters in range(2, max_n_clusters + 1)
        max_n_clusters=5,
        # clustering is done on original data and transformed data, e.g. using PCA to reduce dimensions to
        # options in range(2, dim_reduction_max_dim + 1)
        dim_reduction_max_dim=3,
        # transformations to apply to data before applying clustering algorithms
        # (might be split into preprocessing and dimensionality reduction in the future
        # such that linear preprocessing sklearn pipelines can be defined)
        transformations=[
            "pca",
            "nmf",
            "ica",
            "umap",
            "normalize",
            "power",
            "robust",
            "quantile",
        ],
        # 32bit random state used both for random.seed, np.random.seed and algorithms
        # which support a seed or random_state
        random_state=608429167,
    conda:
        "../envs/clusters.yml"
    script:
        "../scripts/consensus_clusters.py"
