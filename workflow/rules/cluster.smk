rule consensus_clusters:
    input:
        data_matrices=["results/Tumor_methylation_beta.matrix.tsv"],
    output:
        clusters="results/cluster/consensus_clusters.tsv",
        labels="results/cluster/cluster_labels.tsv",
        plot=report("results/cluster/consensus_threshold_graph.html", category="Plots")
    params:
        # for algorithms that support it, generate clusterings with n_clusters in range(2, max_n_clusters + 1)
        max_n_clusters=5,
        # clustering is done on original data and transformed data, e.g. using PCA to reduce dimensions to
        # options in range(2, dim_reduction_max_dim + 1)
        dim_reduction_max_dim=3,
    conda: "envs/clusters.yml"
    script: "scripts/consensus_clusters.py"

