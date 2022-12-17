import os
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image

from file_manager import load_graph_files


def plotout(map_dir):
    fig = plt.figure(frameon=False, figsize=(12, 9), facecolor='none')
    plt.axis('off')
    map_info = load_graph_files(env_path=map_dir, map_lookup="Std")
    col_map = ["gold"] * len(map_info.n_coord)
    nx.draw_networkx(map_info.g_move, map_info.n_coord, node_color=col_map, node_size=150, font_size=6, edge_color='#806C2A', width=0.5, arrows=True)
    plt.savefig("graph_move.png", dpi=200, transparent=True)
    nx.draw_networkx_edges(map_info.g_view, map_info.n_coord, edge_color="grey", width=0.3, arrows=False)
    plt.savefig("graph_view.png", dpi=200, transparent=True)
    plt.close()

if __name__ == "__main__":
    plotout("../../../")