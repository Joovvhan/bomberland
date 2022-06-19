import subprocess
from subprocess import DEVNULL
from tqdm.auto import tqdm
from time import sleep
from glob import glob
import os

import shutil
import argparse
import random

from enum import Enum

from datetime import datetime

# from tensorboardX import SummaryWriter

LOOP_NUM = 300
DOCKER_WAIT_TIME = 5
KEEP_OBS = 10
# KEEP_OBS = 5

# EVALUATION_TERM = 10
# EVALUATION_TERM = 1

class AgentType(Enum):
    STATIC = 1
    SAFE = 2
    NN = 3

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, help='run name', default='exp')
    # parser.add_argument('--k_epochs', type=int, help='k_epochs', default=10)
    # parser.add_argument('--num_matches', type=int, help='num_matches', default=4)
    parser.add_argument('--agent_type', type=int, help='agent_type', default=AgentType.STATIC)
    parser.add_argument('--opponent', type=str, help='opponent', default='b')
    parser.add_argument('--my_unit', type=str, help='my_unit', default='c')
    parser.add_argument('--rand', type=bool, help='Randomize my unit', default=False)
    parser.add_argument('--log_dir', type=str, help='log_dir', default='expert')
    args = parser.parse_args()

    # old_json_files = glob('./trajectory/*/*.json')
    # for file in old_json_files:
    #     os.remove(file)

    # old_npz_dirs = sorted(glob('./obs/*'))
    # for folder in old_npz_dirs:
    #     print(f"Removing Past Experiences: {folder}")
    #     shutil.rmtree(folder)

    for i in tqdm(range(0, LOOP_NUM)):

        if args.rand:
            args.opponent = random.choice(['a', 'b'])
            if args.opponent == 'a':
                args.my_unit = random.choice(['f', 'g', 'h'])
            elif args.opponent == 'b':
                args.my_unit = random.choice(['c', 'd', 'e'])
            else:
                assert False, 'Wrong opponent id {} detected in randomization'

        # old_json_files = glob('./trajectory/*/*.json')
        # for file in old_json_files:
        #     os.remove(file)

        time_str = datetime.now().strftime("%m-%d-%H-%M-%S")

        p_docker = subprocess.Popen(['bash', '../server-run.sh'])
        sleep(DOCKER_WAIT_TIME)

        if args.agent_type == AgentType.STATIC:
            p_agent = subprocess.Popen(['python', 'static_agent.py', f'--id={args.opponent}'], stdout=subprocess.PIPE, stderr=DEVNULL)
        elif args.agent_type == AgentType.SAFE:
            p_agent = subprocess.Popen(['python', 'smart_safe_agent.py', f'--id={args.opponent}'], stdout=subprocess.PIPE, stderr=DEVNULL)
        elif args.agent_type == AgentType.NN:
            p_agent = subprocess.Popen(['python', 'agent.py', f'--id={args.opponent}', f'--run_name={args.run_name}'])
        else:
            assert False, f'Wrong agent type [{args.agent_type}]'

        p_spectator = subprocess.Popen(['python', 'spectator.py', f'--log_dir={args.log_dir}/{args.my_unit}/{time_str}'])

        print(f'Select your agent [{}].')
              
        # [p.wait() for p in (p_docker, p_a, p_b, p_a_y, p_b_y)]
        # [p.wait() for p in (p_docker, p_a, p_b)]
        p_docker.wait()
        print("Episode Completed")
