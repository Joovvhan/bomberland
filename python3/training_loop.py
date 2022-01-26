import subprocess
from subprocess import DEVNULL
from tqdm.auto import tqdm
from time import sleep
from glob import glob
import os

import shutil
import argparse

LOOP_NUM = 300
DOCKER_WAIT_TIME = 5
KEEP_OBS = 10

EVALUATION_TERM = 10

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, help='run name', default='exp')
    parser.add_argument('--k_epochs', type=int, help='k_epochs', default=10)
    args = parser.parse_args()

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

        # p_docker = subprocess.Popen(['bash', '../server-run.sh'])
        p_docker = subprocess.Popen(['bash', '../server-run-2.sh'])
        sleep(DOCKER_WAIT_TIME)

        p_a = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=x'],
                                stdout=DEVNULL, stderr=DEVNULL)
        p_b = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', '--log=True', f'--ep={i}', '--code=x'])

        p_a_y = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=y', '--port=3001'], 
                                 stdout=DEVNULL, stderr=DEVNULL)
        p_b_y = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', f'--ep={i}', '--code=y', '--port=3001'])

        p_a_z = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=z', '--port=3002'], 
                                 stdout=DEVNULL, stderr=DEVNULL)
        p_b_z = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', f'--ep={i}', '--code=z', '--port=3002'])

        p_a_w = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=w', '--port=3003'], 
                                 stdout=DEVNULL, stderr=DEVNULL)
        p_b_w = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', f'--ep={i}', '--code=w', '--port=3003'])       


        # [p.wait() for p in (p_docker, p_a, p_b, p_a_y, p_b_y)]
        # [p.wait() for p in (p_docker, p_a, p_b)]
        p_docker.wait()
        print("Episode Completed")

        p_batch_a = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}', '--code=x'])
        p_batch_b = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}', '--code=y'])
        p_batch_c = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}', '--code=z'])
        p_batch_d = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}', '--code=w'])
        
        p_batch_a.wait()
        p_batch_b.wait()
        p_batch_c.wait()
        p_batch_d.wait()
        print("Building Observation Completed")

        # old_npz_files = glob('./obs/*.npz')
        # for file in old_npz_files:
        #     os.remove(file)

        old_npz_dirs = sorted(glob('./obs/*'))
        if len(old_npz_dirs) > KEEP_OBS:
            for folder in old_npz_dirs[:-KEEP_OBS]:
                print(f"Removing Past Experiences: {folder}")
                shutil.rmtree(folder)

        p_train = subprocess.Popen(['python', 'train.py', f'--run_name={args.run_name}', f'--k_epochs={args.k_epochs}'])
        p_train.wait()
        print("Training Completed")


        # if (i % EVALUATION_TERM) == 0 and i != 0:
        if (i % EVALUATION_TERM) == 0:
            print("Evaluation Started")
            p_docker = subprocess.Popen(['bash', '../server-run-3.sh'], stdout=DEVNULL, stderr=DEVNULL)
            # print("Static Evaluation Started")
            # p_docker = subprocess.Popen(['bash', '../server-run.sh'], stdout=DEVNULL, stderr=DEVNULL)
            sleep(DOCKER_WAIT_TIME)
            p_a = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}'], stdout=DEVNULL, stderr=DEVNULL)
            # p_b = subprocess.Popen(['python', 'static_agent.py', '--id=b', f'--run_name={args.run_name}', '--log=True', f'--ep={i}'], stdout=DEVNULL, stderr=DEVNULL)
            p_b = subprocess.Popen(['python', 'static_agent.py', '--id=b', f'--run_name={args.run_name}', '--log=True', f'--ep={i}'])
            # [p.wait() for p in (p_docker, p_a, p_b)]
            # print("Static Evaluation Completed")

            # print("Smart Safe Evaluation Started")
            # p_docker = subprocess.Popen(['bash', '../server-run.sh'], stdout=DEVNULL, stderr=DEVNULL)
            # sleep(DOCKER_WAIT_TIME)
            p_a_ = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', '--port=3001']) #, stdout=DEVNULL, stderr=DEVNULL)
            # p_b = subprocess.Popen(['python', 'smart_safe_agent.py', '--id=b', f'--run_name={args.run_name}', '--log=True', f'--ep={i}'], stdout=DEVNULL, stderr=DEVNULL)
            p_b_ = subprocess.Popen(['python', 'smart_safe_agent.py', '--id=b', f'--run_name={args.run_name}', '--log=True', f'--ep={i}', '--port=3001'])
            # [p.wait() for p in (p_docker, p_a, p_b)]
            # print("Smart Safe Evaluation Completed")

            p_docker.wait()
            print("Evaluation Completed")

        # p_docker = subprocess.Popen(['bash', '../server-run-2.sh'])
        # sleep(DOCKER_WAIT_TIME)
        # p_a = subprocess.Popen(['python', 'agent.py', '--id=a', f'--ep={i}', '--port=3001'], stdout=DEVNULL, stderr=DEVNULL)
        # p_b = subprocess.Popen(['python', 'agent.py', '--id=b', '--port=3001'])
