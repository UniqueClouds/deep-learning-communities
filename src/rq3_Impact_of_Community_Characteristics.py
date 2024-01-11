import os
import pandas as pd
import numpy as np
from copy import deepcopy
import sys

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn
import tigramite.data_processing as pp
from statsmodels.tsa.stattools import adfuller
from tigramite.plotting import _check_matrices, _draw_network_with_curved_edges
from matplotlib import pyplot
import networkx as nx

columns = [
    "active_users",
    "core_developers",
    "peripheral_developers",
    "retention rate",
    "inflow rate",
    "dropout rate",
    "core-to-peri rate",
    "stars",
]


# columns = [
#     "active",
#     "stars",
#     "core",
#     "peri",
#     "retention_rate",
#     "inflow_rate",
#     "dropout_rate",
#     "core2peripheral_rate",
# ]
# var_names = ["AU", "S", "C", "P", "R", "I", "D", "C2P"]
var_names = ["AU", "C", "P", "R", "I", "D", "C2P", "S"]


def load_and_process_data(data_path):
    """
    Load and process data from the given path.
    Parameters:
        data_path (str): The path of the data.
    Returns:
        data (pandas.DataFrame): The processed data.
    """
    # data format: [community_name, active_users, core_developers, peripheral_developers, retention_rate, inflow_rate, dropout_rate, core-to-peri rate, stars]
    data = pd.read_csv(data_path)
    data = data.applymap(lambda x: x + 0.0000001 if x == 0 else x)
    # columns = ['stars', 'core', 'peri', 'active', 'retention_rate', 'inflow_rate', 'dropout_rate', 'core2peripheral_rate']
    try:
        data = data[columns]
    except KeyError as error:
        print(f"Error: {error} column(s) not found in the input data.")

    ret = pd.DataFrame()
    for c in data.columns:
        origin = data[c]
        if is_stable(origin[1:]):
            ret[c] = np.array(origin[1:])
        elif is_stable(np.diff(origin)):
            ret[c] = np.diff(origin)
        elif is_stable(np.log(origin[1:])):
            ret[c] = np.log(origin[1:])
        elif is_stable(np.diff(np.log(origin))):
            ret[c] = np.diff(np.log(origin))
    return ret


def is_stable(col):
    """
    This function uses the Augmented Dickey-Fuller (ADF) test to check the stability of a time series data column.
    The code you provided defines a function called `is_stable` that takes a single argument `col`. This function uses the Augmented Dickey-Fuller (ADF) test to check the stability of a time series data column.

    Return:
        a value indicating the level of stability (1, 5, 10) or 0 if the column is not stable.
    """
    adf_res = adfuller(col)
    if adf_res[1] <= 0.001 or adf_res[0] <= adf_res[4]["1%"]:
        return 1
    elif adf_res[1] <= 0.01 or adf_res[0] <= adf_res[4]["5%"]:
        return 5
    elif adf_res[1] <= 0.05 or adf_res[4]["10%"]:
        return 10
    return 0


def get_tau_max(lagged_res, level=0.01):
    """
    Finds the maximum tau value for which there exists an element in the 'p_matrix' below the specified level.

    Parameters:
    lagged_res (dict): A dictionary-like object containing a 'p_matrix' key, representing a 3-dimensional matrix.
    level (float, optional): The threshold level for comparison. Defaults to 0.01.

    Returns:
    int: The maximum tau value for which there exists an element below the specified level. Returns 1 if no such element is found.

    """
    p_matrix = lagged_res["p_matrix"]
    tmax = p_matrix.shape[-1] - 1
    for tau in range(tmax, 1, -1):
        mat = p_matrix[:, :, tau]
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i != j and mat[i][j] < level:
                    return tau
    return 1


def causal_inference(
    data: pd.DataFrame,
    community_name: str,
    tmax: int,
    tmin: int,
    cond_choice: str,
    alpha_level: float,
    save_dir: str,
    tmax_: int = None,
):
    """
    Perform causal inference analysis on the given data.

    Parameters:

    """

    test_lis = {"ParCorr": ParCorr(), "GPDC": GPDC(), "CMI": CMIknn()}
    cond_ind_test = test_lis[cond_choice]
    dataframe = pp.DataFrame(data.values, var_names=data.columns, missing_flag=0)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)

    alpha = None
    if cond_choice != 0:
        alpha = 0.2

    if tmax_ is None:
        lagged_res = pcmci.get_lagged_dependencies(
            alpha_level=alpha_level, tau_min=tmin, tau_max=tmax
        )
        tau_max = get_tau_max(lagged_res, alpha_level)
        print(f"tau_max: {tau_max}")
    else:
        tau_max = tmax_

    results = pcmci.run_pcmci(tau_min=tmin, tau_max=tau_max, pc_alpha=alpha)

    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    pcmci.print_significant_links(
        p_matrix=p_matrix, val_matrix=val_matrix, alpha_level=alpha_level
    )
    graph = pcmci.get_graph_from_pmatrix(
        p_matrix=p_matrix, alpha_level=alpha_level, tau_min=tmin, tau_max=tau_max
    )
    build_causal_graph_info(
        save_dir, alpha_level, community_name, var_names, p_matrix, val_matrix, graph
    )
    mask = graph == "o-o"
    graph[mask] = "<->"
    # print(p_matrix)
    plot_graph(graph=graph,val_matrix=val_matrix, var_names=['AU', 'C', 'P', 'R', 'I', 'D', 'C2P', 'S'], title=community_name, save_name=f"{save_dir}{community_name}_PCMCI.pdf")


def plot_graph(
    graph,
    val_matrix=None,
    var_names=None,
    fig_ax=None,
    figsize=None,
    save_name=None,
    link_colorbar_label="cross-dependency",
    node_colorbar_label="auto-dependency",
    link_width=None,
    link_attribute=None,
    node_pos=None,
    arrow_linewidth=10.0,
    vmin_edges=-1,
    vmax_edges=1.0,
    edge_ticks=0.4,
    cmap_edges="RdBu_r",
    vmin_nodes=0,
    vmax_nodes=1.0,
    node_ticks=0.4,
    cmap_nodes="OrRd",
    node_size=0.3,
    node_aspect=None,
    arrowhead_size=20,
    curved_radius=0.2,
    label_fontsize=10,
    alpha=1.0,
    node_label_size=10,
    link_label_fontsize=10,
    lag_array=None,
    network_lower_bound=0.2,
    show_colorbar=True,
    inner_edge_style="dashed",
    link_matrix=None,
    special_nodes=None,
    title=None,
):
    if link_matrix is not None:
        raise ValueError(
            "link_matrix is deprecated and replaced by graph array"
            " which is now returned by all methods."
        )
    if fig_ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
        # if title is not None:
        #     pyplot.title(title)
    else:
        fig, ax = fig_ax

    graph = graph.squeeze()

    if graph.ndim == 4:
        raise ValueError(
            "Time series graph of shape (N,N,tau_max+1,tau_max+1) cannot be represented by plot_graph,"
            " use plot_time_series_graph instead."
        )

    if graph.ndim == 2:
        # If a non-time series (N,N)-graph is given, insert a dummy dimension
        graph = np.expand_dims(graph, axis=2)

    if val_matrix is None:
        no_coloring = True
        cmap_edges = None
        cmap_nodes = None
    else:
        no_coloring = False

    (graph, val_matrix, link_width, link_attribute) = _check_matrices(
        graph, val_matrix, link_width, link_attribute
    )

    N, N, dummy = graph.shape
    tau_max = dummy - 1
    max_lag = tau_max + 1

    if np.count_nonzero(graph != "") == np.count_nonzero(np.diagonal(graph) != ""):
        diagonal = True
    else:
        diagonal = False

    if np.count_nonzero(graph == "") == graph.size or diagonal:
        graph[0, 1, 0] = "---"
        no_links = True
    else:
        no_links = False

    if var_names is None:
        var_names = range(N)

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    # val_matrix[np.abs(val_matrix) < sig_thres] = 0.

    # Only draw link in one direction among contemp
    # Remove lower triangle
    link_matrix_upper = np.copy(graph)
    link_matrix_upper[:, :, 0] = np.triu(link_matrix_upper[:, :, 0])

    # net = _get_absmax(link_matrix != "")
    net = np.any(link_matrix_upper != "", axis=2)
    G = nx.DiGraph(net)

    # This handels Graphs with no links.
    # nx.draw(G, alpha=0, zorder=-10)

    node_color = list(np.zeros(N))
    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and lagged links are handled separately
    for u, v, dic in G.edges(data=True):
        dic["no_links"] = no_links
        # average lagfunc for link u --> v ANDOR u -- v
        if tau_max > 0:
            # argmax of absolute maximum
            argmax = np.abs(val_matrix[u, v][1:]).argmax() + 1
        else:
            argmax = 0

        if u != v:
            # For contemp links masking or finite samples can lead to different
            # values for u--v and v--u
            # Here we use the  maximum for the width and weight (=color)
            # of the link
            # Draw link if u--v OR v--u at lag 0 is nonzero
            # dic['inner_edge'] = ((np.abs(val_matrix[u, v][0]) >=
            #                       sig_thres[u, v][0]) or
            #                      (np.abs(val_matrix[v, u][0]) >=
            #                       sig_thres[v, u][0]))
            dic["inner_edge"] = link_matrix_upper[u, v, 0]
            dic["inner_edge_type"] = link_matrix_upper[u, v, 0]
            dic["inner_edge_alpha"] = alpha
            if no_coloring:
                dic["inner_edge_color"] = None
            else:
                dic["inner_edge_color"] = val_matrix[u, v, 0]
            # # value at argmax of average
            # if np.abs(val_matrix[u, v][0] - val_matrix[v, u][0]) > .0001:
            #     print("Contemporaneous I(%d; %d)=%.3f != I(%d; %d)=%.3f" % (
            #           u, v, val_matrix[u, v][0], v, u, val_matrix[v, u][0]) +
            #           " due to conditions, finite sample effects or "
            #           "masking, here edge color = "
            #           "larger (absolute) value.")
            # dic['inner_edge_color'] = _get_absmax(
            #     np.array([[[val_matrix[u, v][0],
            #                    val_matrix[v, u][0]]]])).squeeze()

            if link_width is None:
                dic["inner_edge_width"] = arrow_linewidth
            else:
                dic["inner_edge_width"] = (
                    link_width[u, v, 0] / link_width.max() * arrow_linewidth
                )

            if link_attribute is None:
                dic["inner_edge_attribute"] = None
            else:
                dic["inner_edge_attribute"] = link_attribute[u, v, 0]

            #     # fraction of nonzero values
            dic["inner_edge_style"] = "solid"
            # else:
            # dic['inner_edge_style'] = link_style[
            #         u, v, 0]

            all_strengths.append(dic["inner_edge_color"])

            if tau_max > 0:
                # True if ensemble mean at lags > 0 is nonzero
                # dic['outer_edge'] = np.any(
                #     np.abs(val_matrix[u, v][1:]) >= sig_thres[u, v][1:])
                dic["outer_edge"] = np.any(link_matrix_upper[u, v, 1:] != "")
            else:
                dic["outer_edge"] = False

            dic["outer_edge_type"] = link_matrix_upper[u, v, argmax]

            dic["outer_edge_alpha"] = alpha
            if link_width is None:
                # fraction of nonzero values
                dic["outer_edge_width"] = arrow_linewidth
            else:
                dic["outer_edge_width"] = (
                    link_width[u, v, argmax] / link_width.max() * arrow_linewidth
                )

            if link_attribute is None:
                # fraction of nonzero values
                dic["outer_edge_attribute"] = None
            else:
                dic["outer_edge_attribute"] = link_attribute[u, v, argmax]

            # value at argmax of average
            if no_coloring:
                dic["outer_edge_color"] = None
            else:
                dic["outer_edge_color"] = val_matrix[u, v][argmax]
            all_strengths.append(dic["outer_edge_color"])

            # Sorted list of significant lags (only if robust wrt
            # d['min_ensemble_frac'])
            if tau_max > 0:
                lags = np.abs(val_matrix[u, v][1:]).argsort()[::-1] + 1
                sig_lags = (np.where(link_matrix_upper[u, v, 1:] != "")[0] + 1).tolist()
            else:
                lags, sig_lags = [], []
            if lag_array is not None:
                dic["label"] = str([lag_array[l] for l in lags if l in sig_lags])[1:-1]
            else:
                dic["label"] = str([l for l in lags if l in sig_lags])[1:-1]
        else:
            # Node color is max of average autodependency
            if no_coloring:
                node_color[u] = None
            else:
                node_color[u] = val_matrix[u, v][argmax]
            dic["inner_edge_attribute"] = None
            dic["outer_edge_attribute"] = None

        # dic['outer_edge_edge'] = False
        # dic['outer_edge_edgecolor'] = None
        # dic['inner_edge_edge'] = False
        # dic['inner_edge_edgecolor'] = None

    if special_nodes is not None:
        special_nodes_draw = {}
        for node in special_nodes:
            i, tau = node
            if tau >= -tau_max:
                special_nodes_draw[i] = special_nodes[node]
        special_nodes = special_nodes_draw

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.0]

    if node_pos is None:
        pos = nx.circular_layout(deepcopy(G))
    else:
        pos = {}
        for i in range(N):
            pos[i] = (node_pos["x"][i], node_pos["y"][i])

    if cmap_nodes is None:
        node_color = None

    node_rings = {
        0: {
            "sizes": None,
            "color_array": node_color,
            "cmap": cmap_nodes,
            "vmin": vmin_nodes,
            "vmax": vmax_nodes,
            "ticks": node_ticks,
            "label": node_colorbar_label,
            "colorbar": show_colorbar,
        }
    }

    _draw_network_with_curved_edges(
        fig=fig,
        ax=ax,
        G=deepcopy(G),
        pos=pos,
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string,
        node_rings=node_rings,
        # 'vmin':float or None, 'vmax':float or None, 'label':string or None}}
        node_labels=var_names,
        node_label_size=node_label_size,
        node_alpha=alpha,
        standard_size=node_size,
        node_aspect=node_aspect,
        standard_cmap="OrRd",
        standard_color_nodes="lightgrey",
        standard_color_links="black",
        log_sizes=False,
        cmap_links=cmap_edges,
        links_vmin=vmin_edges,
        links_vmax=vmax_edges,
        links_ticks=edge_ticks,
        # cmap_links_edges='YlOrRd', links_edges_vmin=-1., links_edges_vmax=1.,
        # links_edges_ticks=.2, link_edge_colorbar_label='link_edge',
        arrowstyle="simple",
        arrowhead_size=arrowhead_size,
        curved_radius=curved_radius,
        label_fontsize=label_fontsize,
        link_label_fontsize=link_label_fontsize,
        link_colorbar_label=link_colorbar_label,
        network_lower_bound=network_lower_bound,
        show_colorbar=show_colorbar,
        # label_fraction=label_fraction,
        special_nodes=special_nodes,
    )

    if save_name is not None:
        pyplot.savefig(save_name, dpi=300)
    else:
        return fig, ax


def build_causal_graph_info(
    save_dir, alpha_level, community_name, var_names, p_matrix, val_matrix, graph
):
    """
    Build causal graph information based on the given parameters.

    Args:
        alpha_level (float): The significance level for determining significant links.
        community_name (str): The name of the community.
        var_names (list): List of variable names.
        p_matrix (numpy.ndarray): Matrix of p-values.
        val_matrix (numpy.ndarray): Matrix of values.
        graph (numpy.ndarray or None): Graph representation.

    Returns:
        pandas.DataFrame: Causal graph information.

    """

    # Define the columns for the resulting DataFrame
    comparison_column = [
        "community_name",
        "var_i",
        "direction",
        "var_j",
        "p_value",
        "val",
        "time_lag",
    ]

    # Create an empty DataFrame with the defined columns
    df = pd.DataFrame(columns=comparison_column)

    N = p_matrix.shape[0]

    # Determine significant links based on the graph or p-values
    if graph is not None:
        sig_links = (graph != "") * (graph != "<--")
    else:
        sig_links = p_matrix <= alpha_level

    # Iterate over each variable
    for j in range(N):
        # Get the significant links and corresponding values
        links = {
            (p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
            for p in zip(*np.where(sig_links[:, j, :]))
        }

        # Sort the links based on their values
        sorted_links = sorted(links, key=links.get, reverse=True)
        n_links = len(links)
        var_name_j = var_names[j]

        # Iterate over the sorted links
        for p in sorted_links:
            var_name_i = var_names[p[0]]
            direction = "->"
            p_value = p_matrix[p[0], j, abs(p[1])]
            val = val_matrix[p[0], j, abs(p[1])]
            time_lag = abs(p[1])

            
            # Append the link information to the DataFrame
            new_row = {
                "community_name": community_name,
                "var_i": var_name_i,
                "direction": direction,
                "var_j": var_name_j,
                "p_value": p_value,
                "val": val,
                "time_lag": time_lag,
            }
            # Create a new DataFrame with the new row
            new_df = pd.DataFrame([new_row])

            # Concatenate the new DataFrame with the existing DataFrame
            df = pd.concat([df, new_df], ignore_index=True)
            # df = df.append(new_row, ignore_index=True)

    # Function to merge rows with the same identifier
    def merge_rows(group):
        if len(group) == 1:
            return group
        else:
            merged_row = {
                "community_name": group["community_name"].iloc[0],
                "var_i": group["var_i"].iloc[0],
                "var_j": group["var_j"].iloc[0],
                "direction": "<->",  # Combining direction
                "p_value": group["p_value"].iloc[0],
                "val": group["val"].iloc[0],
                "time_lag": group["time_lag"].iloc[0],
            }
            return pd.DataFrame([merged_row])

    # Add an identifier column based on the variables, p-value, value, and time lag
    df["identifier"] = df.apply(
        lambda x: "".join(sorted([x["var_i"], x["var_j"]]))
        + str(x["p_value"])
        + str(x["val"])
        + str(x["time_lag"]),
        axis=1,
    )

    # Merge rows with the same identifier
    df_merged = (
        df.groupby("identifier")
        .apply(merge_rows)
        .reset_index(drop=True)
        .drop("identifier", axis=1)
    )

    # Sort and print the merged DataFrame
    print(df_merged.sort_values(by=["var_i"], ascending=True).reset_index(drop=True))

    # Save the merged DataFrame to a CSV file
    df_merged.sort_values(by=["var_i"], ascending=True).reset_index(drop=True).to_csv(
        f"{save_dir}{community_name}_causal_graph_grouped.csv", index=False
    )

    return df_merged

if __name__ == "__main__":
    data = load_and_process_data(sys.argv[1])
    community_name = sys.argv[2]
    # data = load_and_process_data('./data/observations/pytorch_1m.csv')
    # community_name = 'pytorch'
    # plot_choice = int(sys.argv[3])
    save_dir = "./result/figures/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if community_name in ["tensorflow", "pytorch"]:
        causal_inference(
            data,
            community_name=community_name,
            tmax=12,
            tmin=0,
            cond_choice="ParCorr",
            alpha_level=0.01,
            save_dir=save_dir,
            tmax_=None,
        )
