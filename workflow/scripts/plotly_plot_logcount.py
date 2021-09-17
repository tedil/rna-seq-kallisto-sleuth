import pandas as pd
import plotly.express as px
import os


def main(snakemake):
    genes_of_interest = set(snakemake.params.genes_of_interest)

    diffexp = pd.read_csv(snakemake.input.diffexp, sep="\t")
    diffexp_genes = set(
        diffexp.query(f"qval < {snakemake.params.sig_level}")["ext_gene"]
    )

    counts = pd.read_csv(snakemake.input.logcounts, sep="\t")
    counts = counts.set_index("gene")
    counts = counts.loc[
        set(counts.index) & (diffexp_genes | genes_of_interest)
    ].reset_index()

    value_vars = list(counts.columns[2:])
    id_vars = ["transcript", "gene"]
    counts = counts.reindex(columns=id_vars + value_vars)
    counts = pd.melt(
        counts, id_vars=id_vars, value_vars=value_vars, value_name="logcount"
    )
    meta = pd.read_csv(snakemake.input.meta, sep="\t").set_index("sample")
    counts["condition"] = counts["variable"].apply(
        lambda s: meta.loc[s][snakemake.params.primary_variable]
    )
    counts["gene"].fillna(
        counts["transcript"].str.split(".", n=1, expand=True)[0], inplace=True
    )
    print(counts)

    os.makedirs(snakemake.output.plots)
    for gene, group in counts.groupby("gene"):
        fig = px.box(
            group,
            x="condition",
            y="logcount",
            color="transcript",
            points="all",
            title=gene,
            template="plotly_white",
        )
        fig.write_html(os.path.join(snakemake.output.plots, f"{gene}.html"))


if __name__ == "__main__":
    main(snakemake)