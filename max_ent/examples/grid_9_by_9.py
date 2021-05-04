#!/usr/bin/env python

from logging import debug, root, DEBUG
from collections import namedtuple, defaultdict
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from max_ent.algorithms.icrl import icrl
from max_ent.gridworld import Directions
import max_ent.gridworld.feature as F
import max_ent.examples.grid_plot as P
from max_ent.algorithms.gridworld_icrl import learn_constraints, setup_mdp, \
    generate_mdft_trajectories, generate_trajectories, MDP, generate_weighted_average_trajectories


Config = namedtuple('Config', ['mdp', 'state_penalties',
                               'action_penalties', 'color_penalties', 'blue', 'green'])


def plot_world(title, game, state_rewards, action_rewards, color_rewards,
               demo, blue_states, green_states, vmin=None, vmax=None):

    fsize = (4.5, 3)
    fig = plt.figure(num=title, figsize=fsize)
    spec = gridspec.GridSpec(ncols=12, nrows=2, figure=fig)
    colored = [(s, 'blue') for s in blue_states]
    colored += [(s, 'green')for s in green_states]

    ax = fig.add_subplot(spec[:, :8])
    p = P.plot_state_values(ax, game.world, state_rewards, game.start,
                            game.terminal, colored, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="5%", pad=0.1)
    cb = plt.colorbar(p, cax=cax)
    cb.ax.yaxis.set_ticks_position("left")

    for t in demo.trajectories[:200]:
        P.plot_trajectory(ax, game.world, t, lw=4,
                          color='white', alpha=0.025)

    ax = fig.add_subplot(spec[0, 9:12])
    P.plot_action_rewards(ax, action_rewards, vmin=vmin, vmax=vmax)

    ax = fig.add_subplot(spec[1, 10])
    P.plot_colors(ax, color_rewards, vmin=vmin, vmax=vmax)
    plt.draw()
    return fig


def config_world(blue, green, constrained_states, constrained_actions, constrained_colors, goal, start=[0]):
    size = 9
    action_penalty, state_penalty, color_penalty = -50, -50, -50
    goal_r, default_reward = 10, -1

    # set-up the mdp
    sf = F.DistinationStateFeature(size**2)
    af = F.DirectionFeature(Directions.ALL_DIRECTIONS)
    cf = F.ColorFeature(['No Color', 'Blue', 'Green'], 0)
    cf.set_states_color(blue, 'Blue')
    cf.set_states_color(green, 'Green')

    # Destination state penalties
    constraints = []
    sp = np.ones(size**2) * default_reward
    sp[goal] = goal_r
    for s in constrained_states:
        constraints.append((sf, s, state_penalty))
        sp[s] = state_penalty

    # Action penalties
    ap = {a: 0 for a in Directions.ALL_DIRECTIONS}
    for a in constrained_actions:
        constraints.append((af, a.idx, action_penalty))
        ap[a] = action_penalty

    # Color
    cp = np.zeros(3)
    for c in constrained_colors:
        constraints.append((cf, c, color_penalty))
        cp[c] = color_penalty

    feature_list = [sf, af, cf]
    mdp = setup_mdp(size, feature_list, constraints, terminal=[goal], terminal_reward=goal_r,
                    default_reward=default_reward, start=[start])

    return Config(mdp, sp, ap, cp, blue, green)


def main():
    np.random.seed(123)
    goal = 8
    blue = [4, 13, 22]  # blue states
    green = [58, 67, 76]  # green states
    cs = [31, 39, 41, 47, 51]  # constrained states
    ca = [Directions.UP_LEFT, Directions.UP_RIGHT]  # constrained actions
    cc = [1, 2]  # constrained colors

    n_cfg = config_world(blue, green, [], [], [], goal)
    c_cfg = config_world(blue, green, cs, ca, cc, goal)
    n, c = n_cfg.mdp, c_cfg.mdp

    # Generate demonstrations and plot the world
    demo = generate_trajectories(n.world, n.reward, n.start, n.terminal)
    vmin = c_cfg.state_penalties.min()
    vmax = c_cfg.state_penalties.max()
    plot_world('Nominal', n, n_cfg.state_penalties,
               n_cfg.action_penalties, n_cfg.color_penalties,
               demo, n_cfg.blue, n_cfg.green, vmin=vmin, vmax=vmax)

    demo = generate_trajectories(c.world, c.reward, c.start, c.terminal)
    plot_world('Original Constrained', c, c_cfg.state_penalties,
               c_cfg.action_penalties, c_cfg.color_penalties,
               demo, c_cfg.blue, c_cfg.green, vmin=vmin, vmax=vmax)
    # plt.show()

    # Learn the constraints
    result = learn_constraints(
        n.reward, c.world, c.terminal, demo.trajectories)

    print("learning finished!")

    learned = MDP(c.world, result.reward, c.terminal, c.start)
    demo = generate_trajectories(
        learned.world, learned.reward, learned.start, learned.terminal)
    plot_world('Learned Constrained', learned, result.state_weights, result.action_weights,
               result.color_weights, demo, c_cfg.blue, c_cfg.green, vmin=vmin, vmax=vmax)

    mdf_demo = generate_mdft_trajectories(
        n.world, n.reward, result.reward, n.start, n.terminal, [0.5, 0.5])

    plot_world('Learned Constrained - MDFT trajectories', learned, result.state_weights, result.action_weights,
               result.color_weights, mdf_demo, c_cfg.blue, c_cfg.green, vmin=vmin, vmax=vmax)

    plt.show()
    debug('Done!')


if __name__ == '__main__':
    main()
