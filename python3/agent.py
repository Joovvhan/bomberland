from game_state import GameState
import asyncio
import random
import os
import argparse
from collections import defaultdict
import json
import numpy as np
import shutil

# a: ammunition
# b: Bomb
# x: Blast
# bp: Blast Powerup
# m: Metal Block
# o: Ore Block
# w: Wooden Block

TYPE2CODE = {
    'a': 1,
    'b': 2,
    'x': 3,
    'bp': 4,
    'm': 5,
    'o': 6,
    'w': 7,
}

def visualize_entities(entities):
    board = [['_'] * 15 for i in range(15)]
    
    for entity in entities:
        board[entity['x']][entity['y']] = entity['type']
    
    for line in board:
        print(''.join(line))
    print()

def observe_entities(entities):
    board = [[0] * 15 for i in range(15)]
    
    for entity in entities:
        board[entity['x']][entity['y']] = TYPE2CODE[entity['type']]
    
    return np.array(board)
    


uri = None
agent_id = None
# uri = os.environ.get(
#     'GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentA&name=defaultName"

actions = ["up", "down", "left", "right", "bomb", "detonate"]

class Agent():
    def __init__(self):

        self.step = 0

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

        # send each unit a random action
        for unit_id in my_units:

            action = random.choice(actions)
            decisions[unit_id].update({'action': action})

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

        if agent_id == 'a':
            entities = game_state.get("entities")
            # visualize_entities(entities)
            # print(decisions)

            # np.save(f'trajectory/{self.step:03d}_obs.npy', observe_entities(entities))

            with open(f'trajectory/{self.step:03d}_entities.json', 'w') as f:
                entities_json = json.dumps(entities, indent=4) 
                f.write(entities_json)

            with open(f'trajectory/{self.step:03d}_action_a.json', 'w') as f:
                decisions_json = json.dumps(decisions, indent=4) 
                f.write(decisions_json)

            unit_state = game_state.get("unit_state")
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
    Agent()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, help='Agent ID')
    args = parser.parse_args()

    agent_id = args.id

    if args.id == 'a':
        shutil.rmtree('./trajectory')
        os.mkdir('./trajectory')
        uri = "ws://127.0.0.1:3000/?role=agent&agentId=agentA&name=defaultName"
    elif args.id == 'b':
        uri = "ws://127.0.0.1:3000/?role=agent&agentId=agentB&name=defaultName"
    else:
        assert f'Invalid agent ID {args.id}'

    main()
