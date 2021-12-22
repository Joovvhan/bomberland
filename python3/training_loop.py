import subprocess
from subprocess import DEVNULL
from tqdm.auto import tqdm
from time import sleep
from glob import glob
import os

LOOP_NUM = 10
DOCKER_WAIT_TIME = 5

if __name__ == "__main__":
    for i in tqdm(range(0, LOOP_NUM)):

        old_json_files = glob('./trajectory/*.json')
        for file in old_json_files:
            os.remove(file)

        old_npz_files = glob('./obs/*.npz')
        for file in old_npz_files:
            os.remove(file)


        p_docker = subprocess.Popen(['bash', '../server-run.sh']) 

        sleep(DOCKER_WAIT_TIME)

        p_a = subprocess.Popen(['python', 'agent.py', '--id=a', f'--ep={i}'], stdout=DEVNULL, stderr=DEVNULL)
        p_b = subprocess.Popen(['python', 'agent.py', '--id=b'])

        [p.wait() for p in (p_docker, p_a, p_b)]
        print("Episode Completed")

        p_batch = subprocess.Popen(['python', 'json2npy.py'])
        p_batch.wait()
        print("Building Observation Completed")

        p_train = subprocess.Popen(['python', 'train.py'])
        p_train.wait()
        print("Training Completed")

