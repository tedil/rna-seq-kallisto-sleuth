from collections import namedtuple
from os import PathLike
import os
from typing import List, Union

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch

Signature = namedtuple("Signature", "name description genes")
Path = Union[PathLike, str]


def main(snakemake):
    tumor_counts: pd.DataFrame = read_logcounts(snakemake.input.logcounts)
    observed_genes = set(tumor_counts.reset_index()["gene"])

    sig_path = snakemake.input.signature
    files = [sig_path]
    gene_sets = [gmt for f in files for gmt in read_gmt(f)]
    gene_sets = {g.name: g for g in gene_sets}

    aggregate_counts = snakemake.params.get("aggregate_counts", None)
    if aggregate_counts:
        grouped_counts = (
            tumor_counts.reset_index().drop(columns=["transcript"]).groupby("gene")
        )
        if aggregate_counts == "max":
            tumor_counts = grouped_counts.max()
        elif aggregate_counts == "min":
            tumor_counts = grouped_counts.min()
        elif aggregate_counts == "median":
            tumor_counts = grouped_counts.median()
        elif aggregate_counts == "mean":
            tumor_counts = grouped_counts.mean()
        tumor_counts["transcript"] = ""

    os.makedirs(snakemake.output.plots, exist_ok=True)
    for name, gene_set in gene_sets.items():
        gene_counts = tumor_counts.loc[observed_genes & gene_set.genes, :]
        gene_counts = gene_counts.reset_index()
        gene_counts["gene"] = gene_counts["gene"].fillna("")
        gene_counts["feature"] = (
            gene_counts["gene"] + " (" + gene_counts["transcript"] + ")"
        )
        gene_counts = gene_counts.drop(columns=["gene", "transcript"])
        gene_counts = gene_counts.set_index("feature")

        if snakemake.params.get("normalize", False):
            from sklearn.preprocessing import StandardScaler  # , Normalizer

            X = gene_counts.values
            X = StandardScaler().fit_transform(X)
            gene_counts = pd.DataFrame(
                data=X, index=gene_counts.index, columns=gene_counts.columns
            )

        fig = dendromap(
            gene_counts,
            linkage=snakemake.params.get("linkage", "ward"),
            metric=snakemake.params.get("metric", "euclidean"),
        )
        fig.layout.title = name
        fig.write_html(os.path.join(snakemake.output.plots, name + ".html"))


def read_gmt_line(s: str) -> Signature:
    name, description, *genes = s.strip().split("\t")
    return Signature(name, description, set(genes))


def read_gmt(path: Path) -> List[Signature]:
    with open(path, "rt") as f:
        sets = [read_gmt_line(l.strip()) for l in f.readlines() if l.strip()]
    return sets


def read_logcounts(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=[1, 0])


def dendromap(data, linkage="ward", metric="euclidean", heatmap_opts: dict = dict()):
    data = data.T

    def distfun(x, metric=metric):
        return sch.distance.pdist(x, metric=metric)

    def linkagefun(x):
        return sch.linkage(x, linkage)

    figure = ff.create_dendrogram(
        data.values,
        orientation="bottom",
        labels=data.index.values,
        linkagefun=linkagefun,
        distfun=distfun,
    )
    for i in range(len(figure["data"])):
        figure["data"][i]["yaxis"] = "y2"
        figure["data"][i]["xaxis"] = "x1"
    dendro_side = ff.create_dendrogram(
        data.values.T,
        orientation="right",
        labels=data.columns.values,
        linkagefun=linkagefun,
        distfun=distfun,
    )
    for i in range(len(dendro_side["data"])):
        dendro_side["data"][i]["xaxis"] = "x2"
        dendro_side["data"][i]["yaxis"] = "y1"
        figure.add_trace(dendro_side["data"][i])

    Y = linkagefun(data)
    X = linkagefun(data.T)
    denD1 = sch.dendrogram(
        X, orientation="right", link_color_func=lambda k: "black", no_plot=True
    )
    denD2 = sch.dendrogram(
        Y, orientation="bottom", link_color_func=lambda k: "black", no_plot=True
    )
    data: pd.DataFrame = data
    d_colindex = data.columns.values[denD1["leaves"]]
    d_rowindex = data.index.values[denD2["leaves"]]
    data = data.reindex(index=d_rowindex, columns=d_colindex).T

    dendro_side["layout"]["yaxis"]["ticktext"] = d_colindex
    figure["layout"]["yaxis"]["ticktext"] = dendro_side["layout"]["yaxis"]["ticktext"]
    figure["layout"]["yaxis"]["tickvals"] = dendro_side["layout"]["yaxis"]["tickvals"]
    figure["layout"]["yaxis"]["side"] = "right"
    figure["layout"]["xaxis"]["ticktext"] = d_rowindex

    # D = data[denD1['leaves'], :][:, denD1['leaves']]
    # D = D[denD2['leaves'], :][:, denD2['leaves']]
    heat_data = data.values
    import numpy as np

    hovertext = np.empty_like(heat_data, dtype=object)
    for i in range(hovertext.shape[0]):
        for j in range(hovertext.shape[1]):
            hovertext[
                i, j
            ] = f"{data.index[i]} ↔ {data.columns[j]}<br>{heat_data[i, j]}"
    heatmap = [
        go.Heatmap(
            x=data.index.values,
            y=data.columns.values,
            z=heat_data,
            text=hovertext,
            hoverinfo="text",
            colorbar={"x": 1.1, "len": 0.825, "yanchor": "bottom", "y": 0},
            **heatmap_opts,
        )
    ]

    heatmap[0]["x"] = list(range(5, len(data.columns.values) * 10 + 5, 10))
    heatmap[0]["y"] = list(range(5, len(data.index.values) * 10 + 5, 10))

    # Add Heatmap Data to Figure
    # figure['data'].extend(heatmap)
    figure.add_trace(heatmap[0])

    # Edit Layout
    figure["layout"].update({"width": None, "height": None})
    figure["layout"].update(
        {"showlegend": False, "hovermode": "closest", "autosize": True}
    )

    # Edit xaxis (heatmap x)
    figure["layout"]["xaxis"].update(
        {
            "domain": [0.15, 1],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": "",
        }
    )
    # Edit xaxis2 (left hand dendro)
    figure["layout"].update(
        {
            "xaxis2": {
                "domain": [0, 0.15],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": "",
            }
        }
    )

    # Edit yaxis (heatmap y)
    figure["layout"]["yaxis"].update(
        {
            "domain": [0, 0.85],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": True,
        }
    )
    # Edit yaxis2 (top side dendro)
    figure["layout"].update(
        {
            "yaxis2": {
                "domain": [0.85, 1],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": "",
            }
        }
    )
    figure["layout"]["xaxis"]["tickvals"] = list(
        range(5, len(data.columns.values) * 10 + 5, 10)
    )
    figure["layout"]["yaxis"]["tickvals"] = list(
        range(5, len(data.index.values) * 10 + 5, 10)
    )
    return figure


if __name__ == "__main__":
    main(snakemake)
