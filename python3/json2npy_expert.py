import numpy as np
from glob import glob
import json
from tqdm.auto import tqdm
import argparse
import os

from tensorboardX import SummaryWriter

# np.set_printoptions(precision=2)
from json2npy import TEAM2CODE, ACTION2CODE, CODE2SYM, TEAM2CODE, A2B
from json2npy import observe_entities, observe_units, duplicate_observation, duplicate_q_vector
from json2npy import observe_my_bomb, observe_my_flame, observe_my_ammunition, observe_empty

if __name__ == "__main__":

    GAMMA = 0.90

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--ep', type=int, help='Episodes', default=0)
    parser.add_argument('--log_dir', type=str, help='Log directory', default='./expert')
    args = parser.parse_args()

    bomb_count = 0
    flame_count = 0

    # for i, (status_file, next_status) in tqdm(enumerate(zip(status_list[:-1], status_list[1:]))):


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
