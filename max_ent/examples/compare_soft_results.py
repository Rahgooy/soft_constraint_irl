from collections import defaultdict
from max_ent.examples.compare_hard_results import main
from max_ent.examples.grid_9_by_9 import config_world, plot_world
import matplotlib.pyplot as plt
import numpy as np
from max_ent.gridworld import Directions
from max_ent.algorithms.gridworld_icrl import ICRL_Result, learn_constraints, convert_constraints_to_probs, generate_trajectories, MDP
import json
import pickle
from scipy import stats

lens = list(range(1, 10)) + list(range(10, 101, 10))

filters = ['all', 'zero', 'non-zero']
types = ['all', 'under', 'over']

error_types = [(f, t) for f in filters for t in types]


def dist(demo):
    dist = np.ones((81, 8)) * 1e-6
    for t in demo:
        for s, a in t.state_actions():
            dist[s, a] += 1
    return dist/dist.sum().reshape(-1, 1)


def kl(true, x):
    true_dist = dist(true)
    x_dist = dist(x)
    kl = true_dist * np.log(true_dist/x_dist)
    return kl.sum()


def mae(true, learned, p_true, p_learned, filter, type, eps=0.2):
    cond = get_condition(true, filter, type, p_true, p_learned, eps)
    if cond.sum() == 0:
        return 0
    return np.abs(true[cond] - learned[cond]).mean()


def count(true, p_true, p_learned, filter, type, eps=0.1):
    cond = get_condition(true, filter, type, p_true, p_learned, eps)
    return cond.sum()


def get_condition(true, filter, type, p_true, p_learned, eps):
    f = np.ones_like(true).astype(np.bool)
    t = np.ones_like(true).astype(np.bool)
    if filter == 'zero':
        f = true == 0
    if filter == 'non-zero':
        f = true != 0

    if type == 'under':
        t = (p_true - p_learned) > eps
    if type == 'over':
        t = (p_learned - p_true) > eps

    cond = f & t
    return cond


def get_probs(reward, result):
    ps, pa, pc = convert_constraints_to_probs(reward, result)
    pa = np.array([pa[a] for a in Directions.ALL_DIRECTIONS])
    p = np.concatenate([ps, pa, pc])
    return p


def get_results(d, r, p_slip):
    np.random.seed(123)
    goal = d['goal']
    blue = d['blue']
    green = d['green']
    cs = np.argwhere(np.array(d['state_reward']) == -50).ravel()
    ca_idx = np.argwhere(np.array(d['action_reward']) == -50).ravel()
    ca = [Directions.ALL_DIRECTIONS[d] for d in ca_idx]
    cc = np.argwhere(np.array(d['color_reward']) == -50).ravel()

    n_cfg = config_world(blue, green, [], [], [], goal,
                         start=d['start'], p_slip=p_slip)
    c_cfg = config_world(blue, green, cs, ca, cc, goal,
                         start=d['start'], p_slip=p_slip)
    n = n_cfg.mdp
    c = c_cfg.mdp

    omega = np.zeros(81 + 8 + 3)
    omega[cs] = -np.array(d['state_reward'])[cs]
    omega[81 + ca_idx] = -np.array(d['action_reward'])[ca_idx]
    omega[89 + cc] = -np.array(d['color_reward'])[cc]

    true_result = convert_to_icrl_result(omega, n.world, c.reward)
    p_true = get_probs(n.reward, true_result)

    N = len(r)
    mae_list = {k: np.zeros((N, len(lens))) for k in error_types}
    count_list = {k: np.zeros((N, len(lens))) for k in error_types}
    kl_list = np.zeros((N, len(lens)))

    # Generate demonstrations
    demo = generate_trajectories(
        c.world, c.reward, c.start, c.terminal, n_trajectories=100)

    for i in range(N):
        for j, result in enumerate(r[i]):
            result = convert_to_icrl_result(
                result['omega'], n.world, result['reward'])
            learned = MDP(n.world, result.reward, n.terminal, n.start)
            learned_demo = generate_trajectories(
                n.world, learned.reward, n.start, n.terminal, n_trajectories=100)

            p_learned = get_probs(n.reward, result)
            kl_list[i, j] = kl(demo.trajectories, learned_demo.trajectories)
            for (f, t) in mae_list:
                mae_list[f, t][i, j] = mae(
                    true_result.omega, result.omega, p_true, p_learned, f, t)
                count_list[f, t][i, j] = count(
                    true_result.omega, p_true, p_learned, f, t)

    return kl_list, mae_list, count_list


def convert_to_icrl_result(omega, world, reward):
    omega_action = {a: -omega[world.n_states + i]
                    for i, a in enumerate(world.actions)}
    omega_state = -omega[:world.n_states]
    omage_color = -omega[world.n_states + world.n_actions:]
    true_result = ICRL_Result(
        omega, reward, omega_state, omega_action, omage_color)
    return true_result


def draw_metric(x, y, sem, y_label, labels, filename):
    plt.figure()
    lwidth = 0.6
    for i in range(y.shape[1]):
        plt.plot(x, y[:, i], 'k', color='purple', marker='o', fillstyle='none',
                 linewidth=lwidth, markersize=5, markeredgewidth=lwidth, label=labels[i])
        plt.fill_between(lens, (y[:, i] - sem[:, i]).clip(0), y[:, i] + sem[:, i], alpha=0.2,
                         facecolor='purple', linewidth=lwidth, antialiased=True)

    plt.xlabel('Number of Demonstrations')
    if len(labels) > 1:
        plt.legend()
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(f'./reports/soft/{filename}.pdf')


def draw_result(kl_list, mae_list, count_list):
    draw_metric(lens, kl_list['mean'].reshape(-1, 1),
                kl_list['sem'].reshape(-1, 1), 'KL-Divergence', [''], 'soft_kl')

    for f, t in mae_list:
        draw_metric(lens, mae_list[f, t]['mean'].reshape(-1, 1),
                    mae_list[f, t]['sem'].reshape(-1, 1), 'Mean Absolute Error', [''], f'mae_{f}_{t}')

        draw_metric(lens, count_list[f, t]['mean'].reshape(-1, 1),
                    count_list[f, t]['sem'].reshape(-1, 1), 'Average #constraints', [''], f'count_{f}_{t}')


def main():
    with open('data/random_data_10.json') as f:
        data = json.load(f)

    with open('results/soft/random_data_10_non_deter.pkl', 'rb') as f:
        results = pickle.load(f)
    kl_list = {'mean': 0, 'sem': 0}
    mae_list, count_list = {}, {}
    n = len(results)

    for i, r in enumerate(results):
        kl_, mae_, count_ = get_results(data[i], r, 0.1)
        kl_list['mean'] += kl_.mean(0)
        # convert to variance for pooled st calculation
        kl_list['sem'] += np.std(kl_, 0) ** 2

        for f, t in mae_:
            if (f, t) not in mae_list:
                mae_list[f, t] = {'mean': 0, 'sem': 0}
                count_list[f, t] = {'mean': 0, 'sem': 0}

            mae_list[f, t]['mean'] += mae_[f, t].mean(0)
            mae_list[f, t]['sem'] += np.std(mae_[f, t], 0) ** 2
            count_list[f, t]['mean'] += count_[f, t].mean(0)
            count_list[f, t]['sem'] += np.std(count_[f, t], 0) ** 2

    kl_list['mean'] /= n
    # Pooled st. The number of samples(n1, n2, ...) are equal
    kl_list['sem'] = np.sqrt(kl_list['sem'] / n)
    for f, t in mae_:
        mae_list[f, t]['mean'] /= n
        count_list[f, t]['mean'] /= n

        mae_list[f, t]['sem'] += np.sqrt(mae_list[f, t]['sem'] / n)
        count_list[f, t]['sem'] += np.sqrt(count_list[f, t]['sem'] / n)

    draw_result(kl_list, mae_list, count_list)

    print('Done!')


if __name__ == '__main__':
    main()
