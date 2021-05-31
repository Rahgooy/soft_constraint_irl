from max_ent.gridworld.gridworld import Directions
from max_ent.examples.grid_9_by_9 import generate_random_config, generate_trajectories
import numpy as np
import json
from pathlib import Path

if __name__ == '__main__':
    N = 100
    games = []
    i = 0
    while i < N:
        n_cfg, c_cfg = generate_random_config(
            min_dist_start_goal=8, p_slip=0, penalty=-50, dist_penalty=True)
        trj = generate_trajectories(c_cfg.mdp.world, c_cfg.mdp.reward, c_cfg.mdp.start,
                                    c_cfg.mdp.terminal).trajectories
        state_seq = []
        action_seq = []
        violation = False
        for t in trj:
            states = []
            actions = []
            prevS = None
            for s, a in t.state_actions():
                if prevS == s:
                    continue
                prevS = s
                states.append(int(s))
                actions.append(int(a))
                action = Directions.ALL_DIRECTIONS[a]
                if c_cfg.state_penalties[s] == -50 or c_cfg.action_penalties[action] == -50:
                    violation = True
                    break
            state_seq.append(states)
            action_seq.append(actions)
            if violation:
                break

        if violation:
            print('.', end='')
            continue

        games.append({
            'game': i + 1,
            'start': int(c_cfg.mdp.start[0]),
            'goal': int(c_cfg.mdp.terminal[0]),
            'blue': c_cfg.blue.tolist(),
            'green': c_cfg.green.tolist(),
            'state_reward': c_cfg.state_penalties.tolist(),
            'action_reward': [c_cfg.action_penalties[a] for a in Directions.ALL_DIRECTIONS],
            'color_reward': c_cfg.color_penalties.tolist(),
            'state_seq': state_seq,
            'action_seq': action_seq,
        })
        print(i+1)
        i += 1

    out = Path("./data/deterministic_data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        json.dump(games, f, indent=4)
