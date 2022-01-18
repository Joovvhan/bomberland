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
        self.step += 1
        pass


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
    