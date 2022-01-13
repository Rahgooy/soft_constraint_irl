from collections import defaultdict
from max_ent.examples.compare_hard_results import main
from max_ent.examples.compare_soft_results import convert_to_icrl_result
from max_ent.examples.grid_9_by_9 import config_world, generate_random_config, plot_world
import matplotlib.pyplot as plt
import numpy as np
from max_ent.gridworld import Directions
from max_ent.algorithms.gridworld_icrl import ICRL_Result, learn_constraints, convert_constraints_to_probs, generate_trajectories, MDP
import json
import pickle
from pathlib import Path
from max_ent.examples.transfer_results import draw_reports

lens = list(range(1, 10)) + list(range(10, 101, 10))


def run_game(d, p_slip, N):
    np.random.seed(123)
    epochs = 500
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
    n, c = n_cfg.mdp, c_cfg.mdp

    logs = []
    for i in range(N):
        print(f'game #{d["game"]}, set #{i+1}')

        # Generate demonstrations for the i-th set
        demo = generate_trajectories(
            c.world, c.reward, c.start, c.terminal, n_trajectories=50)

        # Learn the constraints
        original_learned = learn_constraints(
            n.reward, c.world, c.terminal, demo.trajectories)

        # Randomize the game
        _, r_cfg = generate_random_config(
            min_dist_start_goal=5, penalty=-50, dist_penalty=True)
        rand_cs = np.argwhere(r_cfg.state_penalties == -50)[:len(cs)].ravel()
        rand_n_cfg = config_world(r_cfg.blue.tolist(), r_cfg.green.tolist(), [], [], [],
                                  goal=r_cfg.mdp.terminal,
                                  start=r_cfg.mdp.start, p_slip=p_slip)
        rand_c_cfg = config_world(r_cfg.blue.tolist(), r_cfg.green.tolist(), rand_cs, ca, cc,
                                  goal=r_cfg.mdp.terminal,
                                  start=r_cfg.mdp.start, p_slip=p_slip)

        rc, rn = rand_c_cfg.mdp, rand_n_cfg.mdp

        # Generate demo for the randomized world
        r_demo = generate_trajectories(
            rc.world, rc.reward, rc.start, rc.terminal, n_trajectories=50)

        # Learn the random world from scratch
        learned_log = []
        rand_learned = learn_constraints(rn.reward, rn.world, rn.terminal,
                                         r_demo.trajectories, log=learned_log, max_iter=epochs)

        # plot_world('Learned Constrained', rc, r_cfg.state_penalties,
        #        r_cfg.action_penalties, r_cfg.color_penalties,
        #        r_demo, r_cfg.blue, r_cfg.green, vmin=-50, vmax=50)

        # Learn the random world with transfer
        transfer_log = []
        rand_learned_transfer = learn_constraints(rn.reward, rn.world, rn.terminal, r_demo.trajectories, log=transfer_log,
                                                  max_iter=epochs, initial_omega=original_learned.omega)
        logs.append((rc, learned_log, transfer_log))

    return logs


def run(name, n):
    with open('data/' + name + '.json') as f:
        data = json.load(f)
    results = []
    for d in data:
        results += run_game(d, 0.1, n)
    draw_reports(results, name)


def main():
    run('scobee_example_data', 50)
    print('Done!')


if __name__ == '__main__':
    main()
