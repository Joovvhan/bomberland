import subprocess
from subprocess import DEVNULL
from tqdm.auto import tqdm
from time import sleep
from glob import glob
import os

import shutil

LOOP_NUM = 300
DOCKER_WAIT_TIME = 5
KEEP_OBS = 10

EVALUATION_TERM = 1

if __name__ == "__main__":

    old_json_files = glob('./trajectory/*.json')
    for file in old_json_files:
        os.remove(file)

    old_npz_dirs = sorted(glob('./obs/*'))
    for folder in old_npz_dirs:
        print(f"Removing Past Experiences: {folder}")
        shutil.rmtree(folder)

    for i in tqdm(range(0, LOOP_NUM)):

        old_json_files = glob('./trajectory/*.json')
        for file in old_json_files:
            os.remove(file)

        p_docker = subprocess.Popen(['bash', '../server-run.sh'])

        sleep(DOCKER_WAIT_TIME)

        p_a = subprocess.Popen(['python', 'agent.py', '--id=a', f'--ep={i}', '--save=True'], stdout=DEVNULL, stderr=DEVNULL)
        p_b = subprocess.Popen(['python', 'agent.py', '--id=b'])

        [p.wait() for p in (p_docker, p_a, p_b)]
        print("Episode Completed")

        p_batch = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}'])
        p_batch.wait()
        print("Building Observation Completed")

        # old_npz_files = glob('./obs/*.npz')
        # for file in old_npz_files:
        #     os.remove(file)

        old_npz_dirs = sorted(glob('./obs/*'))
        if len(old_npz_dirs) > KEEP_OBS:
            for folder in old_npz_dirs[:-KEEP_OBS]:
                print(f"Removing Past Experiences: {folder}")
                shutil.rmtree(folder)

        p_train = subprocess.Popen(['python', 'train.py'])
        p_train.wait()
        print("Training Completed")


        if (i % EVALUATION_TERM) == 0:

            p_docker = subprocess.Popen(['bash', '../server-run.sh'])

            sleep(DOCKER_WAIT_TIME)

            p_a = subprocess.Popen(['python', 'agent.py', '--id=a', '--eval=True', f'--ep={i}'], stdout=DEVNULL, stderr=DEVNULL)
            p_b = subprocess.Popen(['python', 'agent.py', '--id=b', '--eval=True'])

            [p.wait() for p in (p_docker, p_a, p_b)]
            print("Evaluation Completed")

        # p_docker = subprocess.Popen(['bash', '../server-run-2.sh'])
        # sleep(DOCKER_WAIT_TIME)
        # p_a = subprocess.Popen(['python', 'agent.py', '--id=a', f'--ep={i}', '--port=3001'], stdout=DEVNULL, stderr=DEVNULL)
        # p_b = subprocess.Popen(['python', 'agent.py', '--id=b', '--port=3001'])
