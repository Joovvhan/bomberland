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
from tensorboardX import SummaryWriter

from json2npy import observe_entities, observe_units, observe_empty, TYPE2CODE

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

    def get_action_mask(self, unit_state, board):

        # "coordinates": [
        # 4,
        # 10
        # ],
        # "hp": 3,
        # "inventory": {
        #     "bombs": 3
        # },
        # "blast_diameter": 3,
        # "unit_id": "c",
        # "agent_id": "a",
        # "invulnerability": 0

        # mask = np.array([1, 1, 1, 1, 1, 1])
        mask = np.array([1, 1, 1, 1, 0, 0])

        if unit_state["hp"] <= 0:
            return mask

        unit_id = unit_state["unit_id"]
        unit_x, unit_y = unit_state["coordinates"]

        # No bomb to detonate
        if self._get_bomb_to_detonate(unit_id) is None:
            mask[5] = 0

        # No bomb to place
        if unit_state["inventory"]["bombs"] <= 0:
            mask[4] = 0

        def check_invalid_index(x, y, board):
            try:
                board[:, x, y]
                return True
            except IndexError:
                return False

        # x: Blast (3)
        # m: Metal Block (5)
        # o: Ore Block (6)
        # w: Wooden Block (7)
        # unit: unit (8)
        # hp: HP (9)

        def check_wall(x, y, board):
            if np.argmax(board[:, x, y]) in (5, 6, 7):
                return True
            if board[8, x, y] != 0 and board[9, x, y] <= 0:
                return True
            
            return False
        
        def check_flame(x, y, board):
            if np.argmax(board[:, x, y]) == 3:
                return True
            return False

        # actions = ["up", "down", "left", "right", "bomb", "detonate"]

        if unit_state["invulnerability"] - self.step <= 1:
            # up
            if not check_invalid_index(unit_x, unit_y+1, board) or check_flame(unit_x, unit_y+1, board):
                mask[0] = 0

            # down
            if not check_invalid_index(unit_x, unit_y-1, board) or check_flame(unit_x, unit_y-1, board):
                mask[1] = 0

            # left
            if not check_invalid_index(unit_x-1, unit_y, board) or check_flame(unit_x-1, unit_y, board):
                mask[2] = 0

            # right
            if not check_invalid_index(unit_x+1, unit_y, board) or check_flame(unit_x+1, unit_y, board):
                mask[3] = 0

        movement_mask = np.array([1, 1, 1, 1, 1, 1])

        # up
        if not check_invalid_index(unit_x, unit_y+1, board) or check_wall(unit_x, unit_y+1, board):
            movement_mask[0] = 0

        # down
        if not check_invalid_index(unit_x, unit_y-1, board) or check_wall(unit_x, unit_y-1, board):
            movement_mask[1] = 0

        # left
        if not check_invalid_index(unit_x-1, unit_y, board) or check_wall(unit_x-1, unit_y, board):
            movement_mask[2] = 0

        # right
        if not check_invalid_index(unit_x+1, unit_y, board) or check_wall(unit_x+1, unit_y, board):
            movement_mask[3] = 0

        return mask * movement_mask, mask

    async def _on_game_tick(self, tick_number, game_state):

        pass

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

            if s['hp'] > 0:
                prob = np.array([1, 1, 1, 1, 0, 0])
                prob = prob / np.sum(prob)
                mask, lethal_mask = self.get_action_mask(s, board)
                prob = mask * prob

                if np.sum(prob) == 0:
                    prob = np.array([1, 1, 1, 1, 0, 0])
                    prob = lethal_mask * prob

                if np.sum(prob) == 0:
                    prob = np.array([1, 1, 1, 1, 0, 0])

                prob = prob / np.sum(prob)
            else:
                prob = np.random.random(len(actions))
                prob = prob / np.sum(prob)

            # Do action probability masking here

            action = np.random.choice(range(len(actions)), p=prob)
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

        self.step += 1


def main():
    agent = Agent()
    return agent.step

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, help='Agent ID', default=None)
    parser.add_argument('--ep', type=int, help='Episodes', default=0)
    parser.add_argument('--port', type=int, help='Port', default=3000)
    parser.add_argument('--log', type=bool, help='Log', default=False)
    parser.add_argument('--save', type=bool, help='Save trajectory', default=False)
    parser.add_argument('--run_name', type=str, help='Run name', default='exp')
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

    steps = main()

    if args.log:
        run_name = args.run_name
        writer = SummaryWriter(f'runs/{run_name}')
        writer.add_scalar('episode_len/eval/smart_safe', steps, args.ep)
        writer.flush()