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

        # send each unit a random action
        for unit_id in my_units:

            prob = np.array([1, 1, 1, 1, 0, 0])
            prob = prob / np.sum(prob)

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
    parser.add_argument('--eval', type=bool, help='Evaluation', default=False)
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
    