import numpy as np
from glob import glob
import json
from tqdm.auto import tqdm
import argparse
import os

from tensorboardX import SummaryWriter

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
    'px': 11, # Potential blast
    'mb': 12, # My bomb
    'ma': 13, # My ammunition
    'mx': 14, # My flame
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

    # board = np.zeros((11, 15, 15))
    board = np.zeros((len(TYPE2CODE), 15, 15))
    for entity in entities:
        # board[TYPE2CODE[entity['type']], entity['x']][entity['y']] = 1
        board[TYPE2CODE[entity['type']], entity['x'], entity['y']] = 1
        if 'hp' in entity:
            # board[TYPE2CODE['hp'], entity['x']][entity['y']] = entity['hp']
            board[
                TYPE2CODE['hp'], entity['x'], entity['y']
            ] = entity['hp']
        # else?

    for entity in entities:
        if entity['type'] == 'b':
            board[TYPE2CODE['px'], entity['x']][entity['y']] = 1

            directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            for direction in directions:
                for i in range(1, entity["blast_diameter"] // 2 + 1):
                    
                    x = entity['x'] + i * direction[0]
                    y = entity['y'] + i * direction[1]

                    if x < 0 or y < 0 or x >= 15 or y >= 15: # Out of range
                        break
                    elif any(board[[TYPE2CODE['a'], TYPE2CODE['bp'], TYPE2CODE['m'], TYPE2CODE['o'], TYPE2CODE['w']], x, y]): # a[1], bp[4], m[5], o[6], w[7], blocked
                        break
                    else:
                        board[TYPE2CODE['px'], x, y] = 1
         
    return board

# {
#     "created": 43,
#     "x": 6,
#     "y": 4,
#     "type": "b",
#     "unit_id": "f",
#     "agent_id": "b",
#     "expires": 83,
#     "hp": 1,
#     "blast_diameter": 3
# },

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

def observe_my_bomb(board, entities, unit_id):
    
    for entity in entities:
        if entity['type'] == 'b' and entity['unit_id'] == unit_id:
            board[TYPE2CODE['mb'], entity['x']][entity['y']] = 1
            
    return board
    
def observe_my_flame(board, entities, unit_id):
    
    for entity in entities:
        if entity['type'] == 'x' and 'unit_id' in entity and entity['unit_id'] == unit_id:
            board[TYPE2CODE['mx'], entity['x']][entity['y']] = 1
            
    return board

def observe_my_ammunition(board, status, unit_id):
    
    if unit_id in status:
        board[TYPE2CODE['ma'], :, :] = status[unit_id]["inventory"]["bombs"];
            
    return board    

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
    return [2, 3, 1, 0, 4, 5][q]

def flip_q_vector(q_vector):
    return [flip_q(q) for q in q_vector]

def flip_q(q):
    # actions = ["up", "down", "left", "right", "bomb", "detonate"]
    return [0, 1, 3, 2, 4, 5][q]

if __name__ == "__main__":

    GAMMA = 0.90

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    # parser.add_argument('--code', type=str)
    parser.add_argument('--codes', type=str)
    parser.add_argument('--run_name', type=str, help='Run name', default=None)
    parser.add_argument('--ep', type=int, help='Episodes', default=0)
    parser.add_argument('--log', type=bool, help='Log', default=False)
    args = parser.parse_args()

    bomb_count = 0
    flame_count = 0

    # for i, (status_file, next_status) in tqdm(enumerate(zip(status_list[:-1], status_list[1:]))):

    print(args.codes)

    for code in args.codes.split():

        # json_files = sorted(glob(f'./trajectory/{args.code}/*.json'))
        json_files = sorted(glob(f'./trajectory/{code}/*.json'))
        match_len = len(json_files) // 4
        status_list = [json_files[4*i:4*i+4] for i in range(match_len)]

        last_rewards = dict({
            'c': 0.0,
            'd': 0.0,
            'e': 0.0,
            'f': 0.0,
            'g': 0.0,
            'h': 0.0,
        })

        print(code, len(json_files))

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
            uid_list = list()
            logp_list = list()
                
            with open(action_a_json, 'r') as f:
                actions = json.load(f)
                for uid in actions:
                    if uid in live_units:
                        # id_list.append(uid)
                        id_list[uid] = len(id_list)
                        uid_list.append(uid)
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
                        uid_list.append(uid)
                        q_list.append(ACTION2CODE[actions[uid]['action']])
                        logp_list.append(actions[uid]['log_prob'])
                        coord_list.append(np.array(status[uid]["coordinates"]))
                        team_list.append(TEAM2CODE[status[uid]["agent_id"]])

            q_vector = np.array(q_list)
            logp_vector = np.array(logp_list)

            observation = np.zeros((0, *board.shape))

            for uid, coord, team in zip(uid_list, coord_list, team_list):
                u_board = np.copy(board)
                u_board[TYPE2CODE['unit'], :, :] = team * u_board[TYPE2CODE['unit'], :, :]
                u_board[TYPE2CODE['p'], coord[0], coord[1]] = 1

                u_board = observe_my_bomb(u_board, entities, uid)
                u_board = observe_my_ammunition(u_board, status, uid)
                u_board = observe_my_flame(u_board, entities, uid)

                observation = np.concatenate((observation, np.expand_dims(u_board, axis=0)), axis=0)

            r_vector = np.zeros_like(q_vector, dtype=float)

            # if next_status is not None:
            next_status_json = next_status[3]
            with open(next_status_json, 'r') as f:
                next_status = json.load(f)

            team_life_reward = {'a': 0, 'b': 0}

            # for uid in live_units:
            for uid, coord, team in zip(uid_list, coord_list, team_list):

                u_board = np.copy(board)
                u_board[TYPE2CODE['unit'], :, :] = team * u_board[TYPE2CODE['unit'], :, :]
                u_board[TYPE2CODE['p'], coord[0], coord[1]] = 1
                u_board = observe_my_flame(u_board, entities, uid)

                on_fire_r = -1 * np.sum(u_board[TYPE2CODE['p'], :, :] * u_board[TYPE2CODE['x'], :, :])
                unit_on_my_fire_r = -1 * np.sum(u_board[TYPE2CODE['mx'], :, :] * u_board[TYPE2CODE['unit'], :, :])

                # if on_fire_r or unit_on_my_fire_r:
                #     print(i, on_fire_r, unit_on_my_fire_r)

                life_r = (next_status[uid]['hp'] - live_units[uid]['hp']) 
                power_r = (next_status[uid]['blast_diameter'] - live_units[uid]['blast_diameter'])
                bomb_r = max(next_status[uid]['inventory']['bombs'] - live_units[uid]['inventory']['bombs'], 0) 
                team_life_reward[live_units[uid]["agent_id"]] += 0.3 * life_r
                # r_vector[id_list[uid]] += (0.9 * life_r + 0.05 * power_r + 0.05 * bomb_r)
                r_vector[id_list[uid]] += (0.9 * life_r + 0.05 * power_r + 0.05 * bomb_r)
                bomb_count += bomb_r
                flame_count += power_r

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

            observation, q_vector, r_vector, logp_vector = duplicate_observation(observation, q_vector, r_vector, logp_vector)

            np.savez(f'./obs/{args.dir}/{code}_{i:03d}_obs.npz',
                observation=observation, 
                q_vector=q_vector, 
                r_vector=r_vector,
                logp_vector=logp_vector)

            # print(observation.shape)
            # # print(observation)
            # print(q_vector.shape, q_vector)
            # print(r_vector.shape, r_vector)
            # print(logp_vector.shape, logp_vector)

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

    if args.log:
        run_name = args.run_name
        writer = SummaryWriter(f'runs/{run_name}')
        writer.add_scalar('pick_up/bomb', bomb_count / (6 * len(args.codes)), args.ep)
        writer.add_scalar('pick_up/flame', flame_count / (6 * len(args.codes)), args.ep)
        writer.flush()
