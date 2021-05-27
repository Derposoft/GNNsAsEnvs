import argparse
import glob
import os
import re

import networkx as nx
import numpy as np
from PIL import Image
import matplotlib.colors as colors
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
from sigma_graph.data.file_manager import check_dir, find_file_in_dir, load_graph_files


def agent_log_parser(line) -> dict:
    agent_info = {}
    # parse team info
    team_red = re.search(r"red:(\d+)", line)
    team_blue = re.search(r"blue:(\d+)", line)
    if team_red is not None:
        agent_info["team"] = "red"
        agent_info["id"] = team_red[1]
    elif team_blue is not None:
        agent_info["team"] = "blue"
        agent_info["id"] = team_blue[1]
    else:
        assert f"[log] Invalid agent team format: {line}"
    # parse agent info
    agent_pos = re.search(r"HP:\s?(\d+) node:(\d+) dir:(\d) pos:\((\d+), (\d+)\)", line)
    if agent_info is not None:
        agent_info["HP"] = int(agent_pos[1])
        agent_info["node"] = int(agent_pos[2])
        agent_info["dir"] = int(agent_pos[3])
        agent_info["pos"] = (int(agent_pos[4]), int(agent_pos[5]))
    else:
        assert f"[log] Invalid agent info format: {line}"
    return agent_info


def list_nums_log_parser(line):
    pass


def log_file_parser(line):
    segments = line.split(" | ")
    step_num = int(re.search(r"Step #\s?(\d+)", segments[0])[1])

    agents = []
    for str_agents in segments[1:-2]:
        agents.append(agent_log_parser(str_agents))
    actions = segments[-2]
    rewards = segments[-1]

    return step_num, agents, actions, rewards[:-1]


def check_log_files(env_dir, log_dir, log_file):
    # generate a subfolder in the log folder for -> animations (and optional pictures for each step)
    log_file_dir = find_file_in_dir(log_dir, log_file)
    fig_file_dir = os.path.join(log_dir, log_file[:-4])
    if not check_dir(fig_file_dir):
        os.mkdir(fig_file_dir)
    return log_file_dir, fig_file_dir


def generate_picture(env_dir, log_dir, log_file, if_froze=False, max_step=40):
    # check file existence
    log_file_dir, fig_folder = check_log_files(env_dir, log_dir, log_file)
    map_info, pat_info = load_graph_files(env_path=env_dir) #, map_lookup="S", route_lookup="[0]", is_pickle_graph=True)
    # load log info
    file = open(log_file_dir, 'r')
    lines = file.readlines()

    # agent color scaled based on health point
    h_range = 10
    h_offset = 0.1
    h_max = 100 + h_offset
    h_min = 100 - h_range + h_offset
    # predetermined colors
    h_colors = 6
    col_map_red = ['#000000', '#200000', '#400000', '#800000', '#BF0000', '#FF0000']
    col_map_blue = ['#000000', '#000020', '#000040', '#000080', '#0000BF', '#0000FF']
    bds = np.append([0], np.linspace(h_min, h_max, num=h_colors))
    norm = colors.BoundaryNorm(boundaries=bds, ncolors=h_colors)

    total_reward = 0
    pause_step = 0

    for i, line in enumerate(lines):
        fig = plt.figure()
        # set figure background opacity (alpha) to 0
        fig.patch.set_alpha(0.)
        fig.tight_layout()
        plt.axis('off')

        if i < max_step:
            idx_step, agents, action, reward = log_file_parser(line)
            text_head = f"#{idx_step:2d}/{max_step} {action} {reward} "
        elif i == max_step:
            # get episode rewards from log
            text_head += line[:-1]
        legend_text = [text_head]
        # set color map for agents and waypoints
        col_map = ["gold"] * len(map_info.n_info)
        for agent in agents:
            legend_text += ["{}_{} HP:{} node:{} dir:{} pos:{}".format(agent["team"], agent["id"], agent["HP"],
                                                                       agent["node"], agent["dir"], agent["pos"])]
            if agent["team"] == 'red':
                col_map[agent['node'] - 1] = col_map_red[norm(agent['HP'])]
            elif agent["team"] == 'blue':
                blue_health = agent['HP']
                col_map[agent['node'] - 1] = col_map_blue[norm(blue_health)]
        # set pause frame number for gif looping
        if if_froze and (not pause_step) and (blue_health < h_min):
            pause_step = i
        # render fig and save to png
        nx.draw_networkx(map_info.g_acs, map_info.n_info, node_color=col_map, edge_color="grey", arrows=True)
        plt.legend(legend_text, bbox_to_anchor=(0.07, 0.95, 0.83, 0.1), loc='lower left', prop={'size': 8},
                   mode="expand", borderaxespad=0.)
        plt.savefig(os.path.join(fig_folder, f"{i:03d}.png"), dpi=100, transparent=True)
        # plt.show()
        plt.close()
    return fig_folder, pause_step


def frame_add_background(img_dir, gif_file, bg_file, fps, stop_frame=0, wait_frame=5):
    img_files = img_dir + "/*.png"
    imgs = []
    frames = 0
    for f in sorted(glob.glob(img_files)):
        foreground = Image.open(f)
        background = Image.open(bg_file)
        background.paste(foreground, (0, 0), foreground)
        imgs.append(background)
        if stop_frame:
            if frames == stop_frame:
                break
            frames += 1
    # set up additional end frames before looping
    if not stop_frame:
        for i in range(wait_frame):
            imgs.append(imgs[-1])
    imgs[0].save(fp=gif_file, format='GIF', append_images=imgs[1:],
                 save_all=True, duration=(1000 // fps), loop=(stop_frame > 0))


def local_run(env_dir, log_dir, prefix, bg_pic, fps, froze, route_only, route_info):
    directory = os.fsencode(log_dir)
    for file in os.listdir(directory):
        log_file = os.fsdecode(file)
        if log_file.endswith(".txt") and log_file.startswith(prefix):
            if route_only:
                # fig_folder = generate_picture_route(env_dir, log_dir, log_file, route_info)
                pause_frame = 0
            else:
                fig_folder, pause_frame = generate_picture(env_dir, log_dir, log_file, froze)
            frame_add_background(fig_folder, os.path.join(log_dir, f"{log_file[:-4]}.gif"), bg_pic, fps,
                                 pause_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_dir', type=str, default='../../', help='path to project root')
    parser.add_argument('--log_dir', type=str, default='../../logs/visuals/demo/', help='path to log file folder')
    parser.add_argument('--prefix', type=str, default='log_', help='log file prefix')
    parser.add_argument('--background', type=str, default='../../logs/visuals/background.png')
    parser.add_argument('--fps', type=int, default=2)  # frame per second in animations

    parser.add_argument('--HP_color_off', action='store_true', default=True, help='gradient colors for HP')
    parser.add_argument('--HP_froze_on', action='store_true', default=False, help='stop animation if agent is dead')

    parser.add_argument('--route_only', type=bool, default=False)  # exclude step info
    parser.add_argument('--route_info', type=str, default='name')  # choose from ['name', 'pos', 'idx']
    args = parser.parse_args()

    local_run(args.env_dir, args.log_dir, args.prefix, args.background, args.fps,
              args.HP_froze_on, args.route_only, args.route_info)
