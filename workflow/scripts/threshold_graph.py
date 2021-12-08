from collections import defaultdict
from itertools import product
from typing import Union

import numpy as np
import pandas as pd
import networkx as nx
from networkx import Graph
from plotly.graph_objects import Scatter, Figure, Layout


def _color_by_supplement_button(supplement: pd.DataFrame):
    if supplement is None or supplement.empty:
        return None
    columns = list(supplement.columns)
    index = list(supplement.index)
    column_types = {
        col: "continuous"
        if np.issubdtype(supplement[col].values.dtype, float)
        else "categorical"
        for col in columns
    }
    column_values = {
        col: list(map(str, supplement[col].fillna("_unknown").values))
        for col in columns
    }
    column_uniques = {
        col: list(sorted(list(np.unique(column_values[col])))) for col in columns
    }
    column_colors = {
        col: {unique: i for i, unique in enumerate(uniques)}
        for col, uniques in column_uniques.items()
    }

    def get_color(i, col):
        if column_uniques[col][i] != "_unknown":
            if column_types[col] == "categorical":
                return f"hsv({int(i / len(column_uniques[col]) * 360)}, 0.75, 0.66)"
            else:
                return f"hsv(0, 0.75, {1 - (float(column_uniques[col][i]) - np.nanmin(supplement[col].values)) / (np.nanmax(supplement[col].values) - np.nanmin(supplement[col].values))})"
        else:
            return "hsva(0, 0, 0.75, 0.33)"

    colormap = {
        col: {i: get_color(i, col) for i in range(len(column_uniques[col]))}
        for col in columns
    }
    column_colors = {
        col: [colormap[col][column_colors[col][v]] for v in column_values[col]]
        for col in columns
    }
    return {
        "buttons": [
            {
                "args": [
                    {
                        "marker": [{"color": column_colors[col], "size": 12}],
                        "text": [
                            list(
                                map(
                                    lambda x: f", {col}: ".join(map(str, x)),
                                    zip(index, column_values[col]),
                                )
                            )
                        ],
                    }
                ],
                "label": col,
                "method": "update",
            }
            for col in columns[::-1]
            if 2 <= len(column_uniques[col]) < len(column_values[col])
        ],
        "direction": "down",
        "showactive": True,
        "xanchor": "left",
        "yanchor": "top",
    }


def _mk_networkx_figure(
    G: Graph, pos, use_weights=True, node_cliques=defaultdict(list)
):
    nodelist = list(G.nodes())
    edgelist = list(G.edges())
    node_xy = np.asarray([pos[v] for v in nodelist])
    edge_xy = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    if use_weights:
        weights = np.asarray([G.get_edge_data(e[0], e[1])["weight"] for e in edgelist])
        weights = (weights - np.min(weights) + 0.05) / (
            np.max(weights) - np.min(weights)
        )
    else:
        weights = np.zeros(edge_xy.shape[0])

    edge_traces = []
    shapes = []

    for i, (((x0, y0), (x1, y1)), w) in enumerate(zip(edge_xy, weights)):
        edge_traces.append(
            Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=w * 4.0, color="rgba(0,0,0,0.33)"),
                name=f"edge_{i}",
            )
        )

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers+text",
        hoverinfo="text",
        textposition="top center",
        marker=dict(size=15, line=dict(width=1)),
        name="nodes",
    )

    clique_map = {}
    i = 0
    for cliques in node_cliques.values():
        for clique in cliques:
            k = tuple(sorted(clique))
            if k not in clique_map:
                clique_map[k] = i
                i += 1
    num_cliques = len(clique_map)
    clique_colors = {
        c: f"hsv({(360 / (num_cliques + 1)) * i}, 0.85, 0.8)"
        for c, i in clique_map.items()
    }
    colors = []
    for i, ((x, y), node) in enumerate(zip(node_xy, nodelist)):
        possible_cliques = node_cliques[node]
        if len(possible_cliques) > 0:
            largest_clique = sorted(
                possible_cliques, key=lambda x: len(x), reverse=True
            )[0]
            colors.append(clique_colors[tuple(sorted(largest_clique))])
        else:
            colors.append("hsv(0, 0, 0)")
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])
        node_trace["text"] += tuple([nodelist[i]])
    node_trace["marker"]["color"] = colors
    return Figure(
        data=edge_traces + [node_trace],
        layout=Layout(
            showlegend=False,
            hovermode="closest",
            shapes=shapes,
            template="plotly_white",
            # margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )


def distance_graph(
    clusterings: pd.DataFrame, metadata: Union[None, pd.DataFrame] = None
):

    graph = nx.Graph()

    if metadata is not None:
        # it is important we add the nodes in the order they appear in the metadata
        # because the dropdown changing the colors and labels relies on the order
        graph.add_nodes_from(list(metadata.index))
    else:
        graph.add_nodes_from(list(clusterings.columns))

    for i, labels in clusterings.iterrows():
        lbls = sorted(set(labels))
        for label in lbls:
            samples_in_same_cluster = sorted(list(labels[labels == label].index))
            for sample_a, sample_b in product(
                samples_in_same_cluster, samples_in_same_cluster
            ):
                if sample_a == sample_b:
                    continue
                if graph.has_edge(sample_a, sample_b):
                    graph[sample_a][sample_b]["weight"] += 1
                    graph[sample_a][sample_b]["count"] += 1
                else:
                    graph.add_edge(sample_a, sample_b, weight=1, count=1)

    edges = list(graph.edges())
    counts = np.asarray([graph.get_edge_data(e[0], e[1])["count"] for e in edges])
    min_count, max_count = np.min(counts), np.max(counts)
    print(np.min(counts), np.median(counts), np.max(counts))
    for edge in edges:
        # graph[edge[0]][edge[1]]['weight'] = (graph[edge[0]][edge[1]]['weight'] - min_weight) / (max_weight - min_weight)
        graph[edge[0]][edge[1]]["weight"] = (
            graph[edge[0]][edge[1]]["weight"] / max_count
        )
    weights = np.asarray([graph.get_edge_data(e[0], e[1])["weight"] for e in edges])

    # edges = [({f}, {t}, [d["count"]]) for (f, t, d) in graph.edges(data=True)]
    # edges = sorted(edges, key=lambda e: (-e[2][0], e[0], e[1]))

    # threshold = -1
    # if threshold == -1:
    #     threshold = np.median(weights) - np.nextafter(0., 1)
    print(np.min(weights), np.median(weights), np.max(weights))

    figures = []
    thresholds = np.arange(0, 1.0, 0.1)
    for threshold in thresholds:
        graph = graph.copy()

        under_threshold_edges = [
            (u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] < threshold
        ]
        graph.remove_edges_from(under_threshold_edges)

        layout = "fruchterman_reingold"  # "kamada_kawai"
        layout_fn = getattr(nx, layout + "_layout", "fruchterman_reingold_layout")
        pos = layout_fn(graph, weight="weight")

        cliques = True
        if cliques:
            cliques = list(nx.find_cliques(graph))
            node_cliques = nx.cliques_containing_node(
                graph, list(graph.nodes()), cliques
            )
        else:
            node_cliques = defaultdict(list)

        fig = _mk_networkx_figure(
            graph, pos, use_weights=True, node_cliques=node_cliques
        )
        figures.append(fig)

    from plotly.graph_objects import Figure, Layout

    fig = Figure(
        data=[],
        layout=Layout(
            showlegend=False,
            hovermode="closest",
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.add_annotation(
        text=f"Nodes: Samples<br>"
        f"Edge from A to B: A and B have clustered together<br>"
        f"Edge weight (width): fraction of times A and B clustered together in all {len(clusterings.index)} clusterings<br>"
        f"Color: Largest clique sample is a member of<br>"
        f"Threshold: Edges with weight < threshold are removed",
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.9,
        y=0.01,
        font=dict(size=14),
        bgcolor="rgba(0, 0, 0, 0.1)",
        bordercolor="black",
        borderwidth=0,
    )

    step_sizes = [0]
    for threshold, subfig in zip(thresholds, figures):
        *edge_traces, node_trace = subfig.data
        for edge_trace in edge_traces:
            edge_trace.visible = False
            fig.add_trace(edge_trace)

        node_trace.visible = False
        fig.add_trace(node_trace)
        step_sizes.append(len(edge_traces) + 1)
    step_sizes = np.cumsum(step_sizes)

    # Create and add slider
    from plotly.graph_objs.layout.slider import Step

    steps = []
    for i, threshold in zip(range(len(fig.data) // 2), thresholds):
        threshold = np.round(threshold, decimals=2)
        step = Step(
            method="update",
            label=f"{threshold}",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"Threshold: {threshold}"},
            ],
        )
        for j in range(step_sizes[i], step_sizes[i + 1]):
            step["args"][0]["visible"][j] = True
        steps.append(step)

    for j in range(step_sizes[0], step_sizes[1]):
        fig.data[j]["visible"] = True

    sliders = [
        dict(
            active=0, currentvalue={"prefix": "Threshold: "}, pad={"t": 50}, steps=steps
        )
    ]

    fig.update_layout(
        sliders=sliders,
        template="plotly_white",
    )

    if metadata is not None:
        color_button = _color_by_supplement_button(metadata)
        fig.update_layout(updatemenus=[color_button])

    fig["layout"]["sliders"][0]["currentvalue"]["prefix"] = "Threshold: "

    return fig


# generate_plots()

clusterings = pd.read_csv(snakemake.input.clusterings, sep="\t").set_index(
    ["cluster_method", "n_clusters", "additional_params"]
)

meta_path = snakemake.input.get("samples", None)
if meta_path:
    metadata = pd.read_csv(meta_path, sep="\t")
    metadata.set_index("sample", inplace=True)
else:
    metadata = None

fig = distance_graph(clusterings, metadata)
fig.write_html(snakemake.output.plot)
