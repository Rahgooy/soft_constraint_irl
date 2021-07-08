from logging import debug
from max_ent.examples.grid_9_by_9 import config_world, plot_world
import matplotlib.pyplot as plt
import numpy as np
from max_ent.gridworld import Directions
from max_ent.algorithms.gridworld_icrl import ICRL_Result, learn_constraints, convert_constraints_to_probs, generate_trajectories, MDP

lens = list(range(1, 10)) + list(range(10, 101, 10))


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


def mae(true, learned):
    return np.abs(true - learned).mean()

def mean_error(true, learned):
    return learned[true == 0].mean()


def draw_line(x, y, std, y_label, labels, filename):
    plt.figure()
    lwidth = 0.6
    for i in range(y.shape[1]):
        plt.plot(x, y[:, i], 'k', color='purple', marker='o', fillstyle='none',
                 linewidth=lwidth, markersize=5, markeredgewidth=lwidth, label=labels[i])
        plt.fill_between(lens, (y[:, i]-std[:, i]).clip(0), y[:, i] + std[:, i], alpha=0.2,
                         facecolor='purple', linewidth=lwidth, antialiased=True)

    plt.xlabel('Number of Demonstrations')
    if len(labels) > 1:
        plt.legend()
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(f'./results/soft/{filename}.pdf')


def main():
    np.random.seed(123)
    N = 10
    goal = 8
    blue = [4, 13, 22]  # blue states
    green = [58, 67, 76]  # green states
    cs = [31, 39, 41, 47, 51]  # constrained states
    ca = [Directions.UP_LEFT, Directions.UP_RIGHT]  # constrained actions
    cc = [1, 2]  # constrained colors

    n_cfg = config_world(blue, green, [], [], [], goal)
    c_cfg = config_world(blue, green, cs, ca, cc, goal)
    n, c = n_cfg.mdp, c_cfg.mdp

    omega = np.zeros(81 + 8 + 3)
    omega[cs] = 50
    omega[81 + Directions.UP_LEFT.idx] = 50
    omega[81 + Directions.UP_RIGHT.idx] = 50
    omega[81 + 8 + 1] = 50
    omega[81 + 8 + 2] = 50
    omega_action = {a: -omega[c.world.n_states + i]
                    for i, a in enumerate(c.world.actions)}
    omega_state = -omega[:c.world.n_states]
    omage_color = -omega[c.world.n_states + c.world.n_actions:]
    true_result = ICRL_Result(
        omega, c.reward, omega_state, omega_action, omage_color)

    kl_list = np.zeros((N, len(lens)))
    mae_list = np.zeros((N, len(lens)))
    me_list = np.zeros((N, len(lens)))

    for i in range(N):
        for j, L in enumerate(lens):
            print(f'set #{i+1}, len={L}')
            # Generate demonstrations and plot the world
            demo = generate_trajectories(
                c.world, c.reward, c.start, c.terminal, n_trajectories=100)

            # Learn the constraints
            result = learn_constraints(
                n.reward, c.world, c.terminal, demo.trajectories[:L])

            learned = MDP(c.world, result.reward, c.terminal, c.start)
            learned_demo = generate_trajectories(
                learned.world, learned.reward, learned.start, learned.terminal, n_trajectories=100)

            kl_list[i, j] = kl(demo.trajectories, learned_demo.trajectories)
            mae_list[i, j] = mae(true_result.omega, result.omega)
            me_list[i, j] = mean_error(true_result.omega, result.omega)
            print(f'kl: {kl_list[i, j]:0.5f}, MAE: {mae_list[i, j]:0.5f}, ME: {me_list[i, j]:0.5f}')

    draw_line(lens, kl_list.mean(0).reshape(-1, 1),
              kl_list.std(0).reshape(-1, 1), 'KL-Divergence', [''], 'soft_kl')
    draw_line(lens, mae_list.mean(0).reshape(-1, 1),
              mae_list.std(0).reshape(-1, 1), 'Mean Absolute Error', [''], 'soft_mae')
    draw_line(lens, me_list.mean(0).reshape(-1, 1),
              me_list.std(0).reshape(-1, 1), 'Mean Error', [''], 'soft_me')
    debug('Done!')


if __name__ == '__main__':
    main()
