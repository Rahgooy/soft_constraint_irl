from collections import defaultdict, namedtuple
from max_ent.examples.compare_hard_results import main
from max_ent.examples.grid_9_by_9 import config_world, plot_world
import matplotlib.pyplot as plt
import numpy as np
from max_ent.gridworld import Directions
from max_ent.algorithms.gridworld_icrl import ICRL_Result, learn_constraints, convert_constraints_to_probs, generate_trajectories, MDP
import json
import pickle
from scipy import stats
from pathlib import Path

PlotSetting = namedtuple('PlotSetting', [
    'filter', 'type', 'name', 'p_thresholds', 'count', 'mae'])

lens = list(range(1, 10)) + list(range(10, 101, 10))
colors = ['purple', 'red', 'orange', 'green']

plots = [
    # Like false positive
    PlotSetting('zero', 'over', 'fp', True, True, False),
    PlotSetting('non-zero', 'under', 'fn', True,
                True, False),  # Like false negative
]
p_thresholds = [0.01, 0.05, 0.1, 0.2]


def dist(demo):
    dist = np.ones((81, 8)) * 1e-6
    for t in demo:
        for s, a, _ in t.transitions():
            dist[s, a] += 1
    return dist/dist.sum().reshape(-1, 1)


def kl(p, q):
    kl = p * np.log2(p/q)
    return kl.sum()


def draw_metric(x, y, sem, y_label, labels, filename):
    plt.figure()
    lwidth = 0.6
    for i in range(y.shape[0]):
        plt.plot(x, y[i, :], 'k', color=colors[i], marker='o', fillstyle='none',
                 linewidth=lwidth, markersize=1, markeredgewidth=lwidth, label=labels[i])
        # plt.fill_between(x, (y[i, :] - sem[i, :]).clip(0), y[i, :] + sem[i, :], alpha=0.2,
        #                  facecolor=colors[i], linewidth=lwidth, antialiased=True)
    if len(labels) > 1:
        plt.legend()

    plt.xlabel('Training Epochs')
    plt.ylabel(y_label)

    plt.grid(axis='both', which='major', ls='--', lw=0.5)
    plt.savefig(f'./reports/transfer/{filename}_transfer.pdf')
    plt.close()


def draw_result(x, mean_reward, mean_reward_t, kl_list, kl_list_t, exp_name):
    kls = np.stack([kl_list.mean(0), kl_list_t.mean(0)])
    kls_std = np.stack([kl_list.std(0), kl_list_t.std(0)])
    draw_metric(x+1, kls, kls_std, 'KL-Divergence', ['MESC-IRL', 'MESC-IRL + Transfer'], f'{exp_name}_kl')

    m_mean = np.stack([mean_reward.mean(0), mean_reward_t.mean(0)])
    m_std = np.stack([mean_reward.std(0), mean_reward_t.std(0)])
    draw_metric(x+1, m_mean, m_std, 'Mean Reward', ['MESC-IRL', 'MESC-IRL + Transfer'], f'{exp_name}_mean_reward')


def draw_reports(results, name):
    N, epochs = len(results), len(results[1][1])
    mean_reward, mean_reward_t = np.zeros((N, epochs)), np.zeros((N, epochs))
    kl_list, kl_list_t = np.zeros((N, epochs)), np.zeros((N, epochs))
    i = 0
    x = np.arange(0, epochs, 1)
    for rc, learned, transfer in results:
        for j in x:
            l = learned[j] if j < len(learned) else learned[-1]
            l_t = transfer[j] if j < len(transfer) else transfer[-1]
            demo = generate_trajectories(
                rc.world, rc.reward, rc.start, rc.terminal, n_trajectories=50).trajectories
            learned_demo = generate_trajectories(
                rc.world, l['best_reward'], rc.start, rc.terminal, n_trajectories=50).trajectories
            transfer_demo = generate_trajectories(
                rc.world, l_t['best_reward'], rc.start, rc.terminal, n_trajectories=50).trajectories

            mean_reward[i, j] = np.mean(
                [sum([rc.reward[t] for t in traj.transitions()]) for traj in learned_demo])
            mean_reward_t[i, j] = np.mean(
                [sum([rc.reward[t] for t in traj.transitions()]) for traj in transfer_demo])
            kl_list[i, j] = kl(dist(demo), dist(learned_demo))
            kl_list_t[i, j] = kl(dist(demo), dist(transfer_demo))
            print(j)

        i += 1

    draw_result(x, mean_reward[:, x], mean_reward_t[:, x], kl_list[:, x], kl_list_t[:, x], name)
