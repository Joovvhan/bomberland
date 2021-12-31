from game_state import GameState
import asyncio
import random
import os
import argparse
from collections import defaultdict
import json
import numpy as np
import shutil

from glob import glob
from PPO import PPO
from tensorboardX import SummaryWriter

from json2npy import observe_entities, observe_units, observe_empty, TYPE2CODE

MODEL = None

SAVE = False

# a: ammunition
# b: Bomb
# x: Blast
# bp: Blast Powerup
# m: Metal Block
# o: Ore Block
# w: Wooden Block

# TYPE2CODE = {
#     'a': 1,
#     'b': 2,
#     'x': 3,
#     'bp': 4,
#     'm': 5,
#     'o': 6,
#     'w': 7,
# }

def visualize_entities(entities):
    board = [['_'] * 15 for i in range(15)]
    
    for entity in entities:
        board[entity['x']][entity['y']] = entity['type']
    
    for line in board:
        print(''.join(line))
    print()

# def observe_entities(entities):
#     board = [[0] * 15 for i in range(15)]
    
#     for entity in entities:
#         board[entity['x']][entity['y']] = TYPE2CODE[entity['type']]
    
#     return np.array(board)
    


uri = None
agent_id = None

# uri = os.environ.get(
#     'GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentA&name=defaultName"

actions = ["up", "down", "left", "right", "bomb", "detonate"]

class Agent():
    def __init__(self):

        self.step = 0
        self.team = 1 if agent_id == 'a' else -1

        self._client = GameState(uri)

        ### any initialization code can go here
        self._client.set_game_tick_callback(self._on_game_tick)

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

        

    # returns coordinates of the first bomb placed by a unit    
    def _get_bomb_to_detonate(self, unit) -> [int, int] or None:
        entities = self._client._state.get("entities")
        bombs = list(filter(lambda entity: entity.get(
            "owner_unit_id") == unit and entity.get("type") == "b", entities))
        bomb = next(iter(bombs or []), None)
        if bomb != None:
            return [bomb.get("x"), bomb.get("y")]
        else:
            return None

    async def _on_game_tick(self, tick_number, game_state):

        # get my units
        my_agent_id = game_state.get("connection").get("agent_id")
        my_units = game_state.get("agents").get(my_agent_id).get("unit_ids")

        decisions = defaultdict(dict)

        entities = game_state.get("entities")
        board = observe_entities(entities)
        unit_state = game_state.get("unit_state")
        board, live_units = observe_units(board, unit_state)
        board = observe_empty(board)

        # send each unit a random action
        for unit_id in my_units:

            s = unit_state[unit_id]

            if MODEL is not None and s['hp'] > 0:
                u_board = np.copy(board)
                u_board[TYPE2CODE['unit'], :, :] = self.team * u_board[TYPE2CODE['unit'], :, :]
                u_board[TYPE2CODE['p'], s['coordinates'][0], s['coordinates'][1]] = 1
                observation = np.expand_dims(u_board, axis=0)

                action, log_prob = MODEL.select_action(observation)
                # print(action, log_prob)
            else:
                prob = np.random.random(len(actions))
                prob = prob / np.sum(prob)

                action = np.random.choice(range(len(actions)), p=prob)
                log_prob = np.log(prob[action])

            decisions[unit_id].update({'action': actions[action]})
            decisions[unit_id].update({'log_prob': log_prob})

            action = actions[action]

            if action in ["up", "left", "right", "down"]:
                await self._client.send_move(action, unit_id)
            elif action == "bomb":
                await self._client.send_bomb(unit_id)
            elif action == "detonate":
                bomb_coordinates = self._get_bomb_to_detonate(unit_id)
                if bomb_coordinates != None:
                    x, y = bomb_coordinates
                    await self._client.send_detonate(x, y, unit_id)
            else:
                print(f"Unhandled action: {action} for unit {unit_id}")

        if agent_id == 'a' and SAVE:
            # entities = game_state.get("entities")
            # visualize_entities(entities)
            # print(decisions)

            # np.save(f'trajectory/{self.step:03d}_obs.npy', observe_entities(entities))

            with open(f'trajectory/{self.step:03d}_entities.json', 'w') as f:
                entities_json = json.dumps(entities, indent=4) 
                f.write(entities_json)

            with open(f'trajectory/{self.step:03d}_action_a.json', 'w') as f:
                decisions_json = json.dumps(decisions, indent=4) 
                f.write(decisions_json)

            # unit_state = game_state.get("unit_state")
            with open(f'trajectory/{self.step:03d}_status.json', 'w') as f:
                unit_state_json = json.dumps(unit_state, indent=4) 
                f.write(unit_state_json)

            # for unit in unit_state:
            #     print(unit, unit_state[unit])
        
        elif agent_id == 'b':
            with open(f'trajectory/{self.step:03d}_action_b.json', 'w') as f:
                decisions_json = json.dumps(decisions, indent=4) 
                f.write(decisions_json)

        self.step += 1


def main():
    agent = Agent()
    return agent.step

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, help='Agent ID', default=None)
    parser.add_argument('--ep', type=int, help='Episodes', default=0)
    parser.add_argument('--port', type=int, help='Port', default=3000)
    parser.add_argument('--eval', type=bool, help='Evaluation', default=False)
    parser.add_argument('--save', type=bool, help='Save trajectory', default=False)
    args = parser.parse_args()

    agent_id = args.id

    if args.id is None:
        uri = os.environ.get('GAME_CONNECTION_STRING')
    elif args.id == 'a':
        shutil.rmtree('./trajectory')
        os.mkdir('./trajectory')
        uri = f"ws://127.0.0.1:{args.port}/?role=agent&agentId=agentA&name=defaultName"
    elif args.id == 'b':
        uri = f"ws://127.0.0.1:{args.port}/?role=agent&agentId=agentB&name=defaultName"
    else:
        assert f'Invalid agent ID {args.id}'

    feature_dim = 16
    lr_actor = 0.0003
    lr_critic = 0.001
    K_epochs = 80
    eps_clip = 0.2         

    SAVE = args.save

    if args.id == 'b' and args.eval:

        MODEL = None
    
    else:
        run_name = 'exp'

        ppo_agent = PPO(feature_dim, lr_actor, lr_critic, K_epochs, eps_clip, run_name=run_name)

        ppo_agent.policy.eval()
        ppo_agent.policy_old.eval()

        directory = "PPO_preTrained"
        directory = directory + '/' + run_name + '/'

        past_checkpoints = sorted(glob(directory + "*.pth"))
        if len(past_checkpoints) > 0:
            last_checkpoint = past_checkpoints[-1]

            print("Loding Checkpoint: " + last_checkpoint)
            ppo_agent.load(last_checkpoint)
        else:
            print("No Checkpoint Found")
            
        MODEL = ppo_agent



    steps = main()

    if args.id == 'a':
        writer = SummaryWriter(f'runs/{run_name}')
        if not args.eval:
            writer.add_scalar('episode_len', steps, args.ep)
        else:
            writer.add_scalar('episode_len/eval', steps, args.ep)

    