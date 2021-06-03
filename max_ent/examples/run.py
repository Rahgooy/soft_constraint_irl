import json

from numpy.lib.arraysetops import intersect1d, setdiff1d
from max_ent.gridworld.gridworld import Directions
from max_ent.gridworld.trajectory import Trajectory
from max_ent.algorithms.gridworld_icrl import Demonstration, MDP, convert_constraints_to_probs, convert_constraints_to_probs2, generate_trajectories, learn_constraints
import numpy as np
from max_ent.examples.grid_9_by_9 import config_world, plot_world
import matplotlib.pyplot as plt


def get_kl(d, cs, ca, cc, trajs):
    ca = [Directions.ALL_DIRECTIONS[d] for d in ca]

    c_cfg = config_world(d['blue'], d['green'], cs,
                         ca, cc, d['goal'], start=d['start'], p_slip=0)
    c = c_cfg.mdp

    def kl(p, q):
        return (p * np.log2(p/q)).sum(1).mean()

    def dist(demo):
        p = np.zeros((c.world.n_states, c.world.n_actions)) + 1e-6
        for t in demo.trajectories:
            for s, a in t.state_actions():
                p[s, a] += 1
        p = p / p.sum(1).reshape(-1, 1)
        return p

    demo = generate_trajectories(c.world, c.reward, c.start, c.terminal)
    p1 = dist(demo)

    trj = [list(zip(s, a)) for s, a in trajs]
    trj = [Trajectory(
        [(s, a, c_cfg.mdp.world.state_index_transition(s, a)) for s, a in t]) for t in trj]
    demo = Demonstration(trj, None)
    p2 = dist(demo)

    return kl(p2, p1)


def get_stats(data, results, g_trajs):
    t = 0.0
    stats = np.zeros((2, len(data)))
    for i in range(len(data)):
        d = data[i]
        r = results[i]
        cs = np.argwhere(np.array(d['state_reward']) == -50).ravel()
        ca = np.argwhere(np.array(d['action_reward']) == -50).ravel()
        cc = np.argwhere(np.array(d['color_reward']) == -50).ravel()
        constrains = r['learned_constraints']
        const_s = [c['value']
                   for c in constrains if c['type'] == 'state' and c['p'] > t]
        const_a = [c['value']
                   for c in constrains if c['type'] == 'action' and c['p'] > t]
        const_c = [c['value']
                   for c in constrains if c['type'] == 'feature' and c['p'] > t]

        fp = len(setdiff1d(const_s, cs)) + \
            len(setdiff1d(const_a, ca)) + len(setdiff1d(const_c, cc))
        fp /= len(cs) + len(ca) + len(cc)
        stats[0, i] = fp
        stats[1, i] = get_kl(d, cs, ca, cc, g_trajs[d['game']])
    # print(stats.reshape(-1, 1))
    return stats.mean(1)


def main():
    threshold = 0.4
    with open('data/scobee_example_data.json') as f:
        data = json.load(f)

    with open('data/scobee_example_trajectories.json') as f:
        trajs = json.load(f)
        g_trajs = {t['game']: list(
            zip(t['state_seq'], t['action_seq'])) for t in trajs}

    results = []
    for d in data:
        print(f"Learning game #{d['game']}")

        cs = np.argwhere(np.array(d['state_reward']) == -50).ravel()
        ca = np.argwhere(np.array(d['action_reward']) == -50).ravel()
        ca = [Directions.ALL_DIRECTIONS[d] for d in ca]
        cc = np.argwhere(np.array(d['color_reward']) == -50).ravel()

        n_cfg = config_world(d['blue'], d['green'], [],
                             [], [], d['goal'], start=d['start'], p_slip=0)
        c_cfg = config_world(d['blue'], d['green'], cs,
                             ca, cc, d['goal'], start=d['start'], p_slip=0)
        n, c = n_cfg.mdp, c_cfg.mdp

        # Generate demonstrations and plot the world
        trj = [list(zip(s, a)) for s, a in g_trajs[d['game']]]
        trj = [Trajectory(
            [(s, a, c_cfg.mdp.world.state_index_transition(s, a)) for s, a in t]) for t in trj]
        demo = Demonstration(trj, None)

        # Learn the constraints
        result = learn_constraints(
            n.reward, c.world, c.terminal, demo.trajectories)             
        result.state_weights[c_cfg.mdp.start] = 0
        learned = MDP(c.world, result.reward, c.terminal, c.start)
        
        p_s, p_a, p_c = convert_constraints_to_probs(n.reward, result)
        # p_s, p_a, p_c = convert_constraints_to_probs2(n_cfg, result)

        p_a_v = np.array([p_a[a] for a in Directions.ALL_DIRECTIONS])

        cs = [{"type": "state", "value": s.tolist()[0], "p": p.tolist()}
              for s, p in zip(np.argwhere(p_s > threshold), p_s[p_s > threshold])]

        ca = [{"type": "action", "value": a.tolist()[0], "p": p.tolist()}
              for a, p in zip(np.argwhere(p_a_v > threshold), p_a_v[p_a_v > threshold])]

        cc = [{"type": "feature", "value": c.tolist()[0], "p": p.tolist()}
              for c, p in zip(np.argwhere(p_c > threshold), p_c[p_c > threshold])]

        cons = cs + ca + cc

        cons = sorted(cons, key=lambda x: -x['p'])
        results.append({
            "id": d['game'],
            "learned_constraints": cons
        })

        plot_world('Original Constrained', c, c_cfg.state_penalties,
                   c_cfg.action_penalties, c_cfg.color_penalties,
                   demo, c_cfg.blue, c_cfg.green, vmin=-50, vmax=10)

        demo = generate_trajectories(
            learned.world, learned.reward, learned.start, learned.terminal)

        plot_world('Learned Constrained', learned, p_s, p_a,
                   p_c, demo, c_cfg.blue, c_cfg.green, vmin=0, vmax=1)

        t = 0.6
        cs = [c['value'] for c in cs if c['p'] > t]
        ca = [Directions.ALL_DIRECTIONS[c['value']]
              for c in ca if c['p'] > t]
        cc = [c['value'] for c in cc if c['p'] > t]

        h_cfg = config_world(d['blue'], d['green'], cs,
                             ca, cc, d['goal'], start=d['start'], p_slip=0)
        h = h_cfg.mdp
        demo = generate_trajectories(
            h.world, h.reward, h.start, h.terminal)
        plot_world('Hard Constrained', h, h_cfg.state_penalties,
                   h_cfg.action_penalties, h_cfg.color_penalties,
                   demo, h_cfg.blue, h_cfg.green, vmin=-50, vmax=10)

        

        plt.show(block=False)
        plt.pause(10)
    with open('data/our_results_10.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
    input("Press any key ...")

