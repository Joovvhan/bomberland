import numpy as np
from glob import glob
import json
from tqdm.auto import tqdm
import argparse
import os

# np.set_printoptions(precision=2)

TYPE2CODE = {
    'e': 0,
    'a': 1,
    'b': 2,
    'x': 3,
    'bp': 4,
    'm': 5,
    'o': 6,
    'w': 7,
    'unit': 8,
    'hp': 9,
    'p': 10,
}

ACTION2CODE = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "bomb": 4,
    "detonate": 5,
}

CODE2SYM = ['_', 'A', '@', '+', '^', 'B', 'R', 'P', '&']

TEAM2CODE = {
    'a': 1,
    'b': -1,
}

A2B = {'a': 'b', 'b': 'a'}

# a: ammunition (1)
# b: Bomb (2)
# x: Blast (3)
# bp: Blast Powerup (4)
# m: Metal Block (5)
# o: Ore Block (6)
# w: Wooden Block (7)

# e: Empty (0)
# unit: unit (8)
# hp: HP (9)
# p: Player (10)
# bo: Bomb Owner
# ex: Expires

# Entity	Properties
# Metal Block (indestructible)  | created, x, y, type=m
# Wooden Block (destructible)   | created, x, y, type=w, hp
# Ore Block (destructible)      | created, x, y, type=o, hp
# Ammunition pickup             | created, x, y, type=a, expires, hp
# Blast radius powerup	        | created, x, y, type=bp, expires, hp
# Bomb                          | created, x, y, type=b, owner_unit_id, expires, hp, blast_diameter
# Explosion	                    | created, x, y, type=x, owner_unit_id, expires
# End-game fire	                | created, x, y, type=x


def visualize_entities(entities):
    board = [['_'] * 15 for i in range(15)]
    
    for entity in entities:
        board[entity['x']][entity['y']] = entity['type']
    
    for line in board:
        print(''.join(line))
    print()

def visualize_board(board):
    max_arg = np.argmax(np.abs(board[:9, :, :]), axis=0)
    str_board = [['' for i in range(board.shape[1])] for j in range(board.shape[2])]
    
    for i in range(board.shape[1]):
        for j in range(board.shape[2]):
            str_board[i][j] = CODE2SYM[max_arg[i, j]]
            if CODE2SYM[max_arg[i, j]] == '&' and board[8, i, j] == -1:
                str_board[i][j] = '%'

    print()
    print('   | ' + '|'.join([f'{i%10:1d}' for i in range(board.shape[1])]))
    for i, line in enumerate(str_board):
        print(f'{i:02d} | ' + ' '.join(line))
    print()

def observe_entities(entities):
    # board = np.array([[0] * 15 for i in range(15)])
    # for entity in entities:
    #     board[entity['x']][entity['y']] = TYPE2CODE[entity['type']]

    board = np.zeros((11, 15, 15))
    for entity in entities:
        board[TYPE2CODE[entity['type']], entity['x']][entity['y']] = 1
        if 'hp' in entity:
            board[TYPE2CODE['hp'], entity['x']][entity['y']] = entity['hp']
        
        # else?
    
    return board

# "c": {
#     "coordinates": [
#         3,
#         10
#     ],
#     "hp": 3,
#     "inventory": {
#         "bombs": 3
#     },
#     "blast_diameter": 3,
#     "unit_id": "c",
#     "agent_id": "a",
#     "invulnerability": 0

def observe_units(board, status):

    live_units = dict()

    for uid in status:
        s = status[uid]

        if s['hp'] > 0:
            if s['agent_id'] == 'a':
                board[TYPE2CODE['unit'], s['coordinates'][0], s['coordinates'][1]] = 1
            else:
                board[TYPE2CODE['unit'], s['coordinates'][0], s['coordinates'][1]] = -1
            
            board[TYPE2CODE['hp'], s['coordinates'][0], s['coordinates'][1]] = s['hp']

            live_units[uid] = s

        else:
            pass
    
    return board, live_units

def observe_empty(board):
    board[0, :, :] = (np.argmax(board, axis=0) == 0)
    return board

def duplicate_observation(observation, q_vector, r_vector, logp_vector):

    obs = np.copy(observation)
    observation = np.concatenate([observation, np.flip(obs, axis=2)])

    for i in range(3):
        obs = np.rot90(obs, k=1, axes=(2, 3))
        observation = np.concatenate([observation, obs])
        observation = np.concatenate([observation, np.flip(obs, axis=2)])

    q_vector = duplicate_q_vector(q_vector)
    r_vector = np.tile(r_vector, 8)
    logp_vector = np.tile(logp_vector, 8)

    return observation, q_vector, r_vector, logp_vector

def duplicate_q_vector(q_vector):

    vector = np.copy(q_vector)
    q_vector = np.concatenate([q_vector, flip_q_vector(q_vector)])

    for i in range(3):
        vector = rotate_q_vector(vector)
        q_vector = np.concatenate([q_vector, vector])
        q_vector = np.concatenate([q_vector, flip_q_vector(vector)])

    return q_vector

def rotate_q_vector(q_vector):
    return [rotate_q(q) for q in q_vector]

def rotate_q(q):
    # actions = ["up", "down", "left", "right", "bomb", "detonate"]
    return [3, 2, 0, 1, 4, 5][q]

def flip_q_vector(q_vector):
    return [flip_q(q) for q in q_vector]

def flip_q(q):
    # actions = ["up", "down", "left", "right", "bomb", "detonate"]
    return [0, 1, 3, 2, 4, 5][q]

if __name__ == "__main__":

    GAMMA = 0.90

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()

    json_files = sorted(glob('./trajectory/*.json'))
    match_len = len(json_files) // 4
    status_list = [json_files[4*i:4*i+4] for i in range(match_len)]

    # for i, (status_file, next_status) in tqdm(enumerate(zip(status_list[:-1], status_list[1:]))):

    last_rewards = dict({
        'c': 0.0,
        'd': 0.0,
        'e': 0.0,
        'f': 0.0,
        'g': 0.0,
        'h': 0.0,
    })

    for i in tqdm(reversed(range(len(status_list) - 1))):

        # print('Before:', last_rewards)

        status_file, next_status = status_list[i], status_list[i+1]

        # print(i, status_file, next_status)

        action_a_json = status_file[0]
        action_b_json = status_file[1]
        entities_json = status_file[2]
        status_json = status_file[3]

        with open(entities_json, 'r') as f:
            entities = json.load(f)
            board = observe_entities(entities)
            # visualize_entities(entities)

        with open(status_json, 'r') as f:
            status = json.load(f)
            # print(status)
            board, live_units = observe_units(board, status)

        board = observe_empty(board)

        # id_list, q_list, coord_list, team_list = list(), list(), list(), list()
        id_list, q_list, coord_list, team_list = dict(), list(), list(), list()
        logp_list = list()
            
        with open(action_a_json, 'r') as f:
            actions = json.load(f)
            for uid in actions:
                if uid in live_units:
                    # id_list.append(uid)
                    id_list[uid] = len(id_list)
                    q_list.append(ACTION2CODE[actions[uid]['action']])
                    logp_list.append(actions[uid]['log_prob'])
                    coord_list.append(np.array(status[uid]["coordinates"]))
                    team_list.append(TEAM2CODE[status[uid]["agent_id"]])

        with open(action_b_json, 'r') as f:
            actions = json.load(f)
            for uid in actions:
                if uid in live_units:
                    # id_list.append(uid)
                    id_list[uid] = len(id_list)
                    q_list.append(ACTION2CODE[actions[uid]['action']])
                    logp_list.append(actions[uid]['log_prob'])
                    coord_list.append(np.array(status[uid]["coordinates"]))
                    team_list.append(TEAM2CODE[status[uid]["agent_id"]])

        q_vector = np.array(q_list)
        logp_vector = np.array(logp_list)

        observation = np.zeros((0, *board.shape))

        for coord, team in zip(coord_list, team_list):
            u_board = np.copy(board)
            u_board[TYPE2CODE['unit'], :, :] = team * u_board[TYPE2CODE['unit'], :, :]
            u_board[TYPE2CODE['p'], coord[0], coord[1]] = 1
            observation = np.concatenate((observation, np.expand_dims(u_board, axis=0)), axis=0)

        r_vector = np.zeros_like(q_vector, dtype=float)

        # if next_status is not None:
        next_status_json = next_status[3]
        with open(next_status_json, 'r') as f:
            next_status = json.load(f)

        team_life_reward = {'a': 0, 'b': 0}

        for uid in live_units:
            life_r = next_status[uid]['hp'] - live_units[uid]['hp'] 
            power_r = next_status[uid]['blast_diameter'] - live_units[uid]['blast_diameter'] 
            team_life_reward[live_units[uid]["agent_id"]] += life_r
            r_vector[id_list[uid]] += (life_r + power_r)

        for uid in live_units:
            r_vector[id_list[uid]] -= team_life_reward[A2B[live_units[uid]["agent_id"]]]

        for uid in last_rewards:
            last_rewards[uid] *= GAMMA

        for uid in live_units:
            # print(last_rewards, r_vector)
            r_vector[id_list[uid]] += last_rewards[uid]
            last_rewards[uid] = r_vector[id_list[uid]]

        # else:
        #     pass

        os.makedirs(f'./obs/{args.dir}', exist_ok=True)



        np.savez(f'./obs/{args.dir}/{i:03d}_obs.npz',
                 observation=observation, 
                 q_vector=q_vector, 
                 r_vector=r_vector,
                 logp_vector=logp_vector)

        observation, q_vector, r_vector, logp_vector = duplicate_observation(observation, q_vector, r_vector, logp_vector)

        print(observation.shape)
        # print(observation)
        print(q_vector.shape, q_vector)
        print(r_vector.shape, r_vector)
        print(logp_vector.shape, logp_vector)
        break

        # print('After:', last_rewards, r_vector)
        # print(('{:+0.2f} ' * len(r_vector)).format(*r_vector))

        # if any(r_vector):
        #     visualize_board(board)
        #     for i, (uid, q, coord, team, r) in enumerate(zip(id_list, q_list, coord_list, team_list, r_vector)):
        #         print(uid, f'{team:+2d}', live_units[uid]['hp'], q, f'{r:+2d}', coord)
        #         # print(observation[i, -1, :, :])

        #     print(observation.shape, q_vector.shape, r_vector.shape)
            # break

        # break


