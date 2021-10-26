Dendrogram of logcounts for model ``{{ snakemake.wildcards.model }} ``, only displaying genes from gene set ``{{ snakemake.wildcards.signature }} ``.
Transcripts were aggregated to gene level by `{{ snakemake.params.aggregate_counts }}`.
For hierarchical clustering, the linkage function used is ``{{ snakemake.params.linkage }}`` with metric ``{{ snakemake.params.metric }} ``.
