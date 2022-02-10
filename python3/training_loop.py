import subprocess
from subprocess import DEVNULL
from tqdm.auto import tqdm
from time import sleep
from glob import glob
import os

import shutil
import argparse

from tensorboardX import SummaryWriter

LOOP_NUM = 300
DOCKER_WAIT_TIME = 5
KEEP_OBS = 10
# KEEP_OBS = 5

EVALUATION_TERM = 10
# EVALUATION_TERM = 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, help='run name', default='exp')
    parser.add_argument('--k_epochs', type=int, help='k_epochs', default=10)
    parser.add_argument('--num_matches', type=int, help='num_matches', default=4)
    args = parser.parse_args()

    old_json_files = glob('./trajectory/*/*.json')
    for file in old_json_files:
        os.remove(file)

    old_npz_dirs = sorted(glob('./obs/*'))
    for folder in old_npz_dirs:
        print(f"Removing Past Experiences: {folder}")
        shutil.rmtree(folder)

    for i in tqdm(range(0, LOOP_NUM)):

        old_json_files = glob('./trajectory/*/*.json')
        for file in old_json_files:
            os.remove(file)

        # p_docker = subprocess.Popen(['bash', '../server-run.sh'])
        p_docker = subprocess.Popen(['bash', '../server-run-2.sh'])
        sleep(DOCKER_WAIT_TIME)

        codes = ['x', 'y', 'z', 'w', 'v', 'u', 'q', 'r', 's', 't'][:args.num_matches]

        for j in range(args.num_matches):
            p_a = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', f'--code={codes[j]}', f'--port={3000+j}'],
                                    stdout=DEVNULL, stderr=DEVNULL)
            if j == 0:
                p_b = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', f'--code={codes[j]}', f'--port={3000+j}', '--log=True'])
            else:
                p_b = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', f'--code={codes[j]}', f'--port={3000+j}'])

        # p_a = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=x'],
        #                         stdout=DEVNULL, stderr=DEVNULL)
        # p_b = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', '--log=True', '--save=True', f'--ep={i}', '--code=x'])

        # p_a_y = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=y', '--port=3001'], 
        #                         stdout=DEVNULL, stderr=DEVNULL)
        # p_b_y = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=y', '--port=3001'])

        # p_a_z = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--save=True', '--code=z', '--port=3002'], 
        #                         stdout=DEVNULL, stderr=DEVNULL)
        # p_b_z = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=z', '--port=3002'])

        # p_a_w = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=w', '--port=3003'], 
        #                         stdout=DEVNULL, stderr=DEVNULL)
        # p_b_w = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=w', '--port=3003'])
        
        # p_a_u = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=v', '--port=3003'], 
        #                         stdout=DEVNULL, stderr=DEVNULL)
        # p_b_u = subprocess.Popen(['python', 'agent.py', '--id=b', f'--run_name={args.run_name}', f'--ep={i}', '--save=True', '--code=v', '--port=3003'])              


        # [p.wait() for p in (p_docker, p_a, p_b, p_a_y, p_b_y)]
        # [p.wait() for p in (p_docker, p_a, p_b)]
        p_docker.wait()
        print("Episode Completed")

        # sleep(DOCKER_WAIT_TIME)


        p_batch = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}', f"--codes={' '.join(codes)}", f'--run_name={args.run_name}', '--log=True', f'--ep={i}'])
        # p_batch = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}', '--codes=v w x y z', f'--run_name={args.run_name}', '--log=True', f'--ep={i}'])
        # p_batch_a = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}', '--code=x'])
        # p_batch_b = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}', '--code=y'])
        # p_batch_c = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}', '--code=z'])
        # p_batch_d = subprocess.Popen(['python', 'json2npy.py',  f'--dir=ep{i:03d}', '--code=w'])
        
        # p_batch_a.wait()
        # p_batch_b.wait()
        # p_batch_c.wait()
        # p_batch_d.wait()

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

        p_train = subprocess.Popen(['python', 'train.py', f'--run_name={args.run_name}', f'--k_epochs={args.k_epochs}'])
        p_train.wait()
        print("Training Completed")


        if (i % EVALUATION_TERM) == 0 and i != 0:
        # if (i % EVALUATION_TERM) == 0:
            print("Evaluation Started")
            p_docker = subprocess.Popen(['bash', '../server-run-3.sh'], stdout=DEVNULL, stderr=DEVNULL)
            # print("Static Evaluation Started")
            # p_docker = subprocess.Popen(['bash', '../server-run.sh'], stdout=DEVNULL, stderr=DEVNULL)
            sleep(DOCKER_WAIT_TIME)
            p_a = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}'], stdout=DEVNULL, stderr=DEVNULL)
            # p_b = subprocess.Popen(['python', 'static_agent.py', '--id=b', f'--run_name={args.run_name}', '--log=True', f'--ep={i}'], stdout=DEVNULL, stderr=DEVNULL)
            p_b = subprocess.Popen(['python', 'static_agent.py', '--id=b', f'--run_name={args.run_name}', '--log=True', f'--ep={i}'], stdout=subprocess.PIPE, stderr=DEVNULL)
            # [p.wait() for p in (p_docker, p_a, p_b)]
            # print("Static Evaluation Completed")

            # print("Smart Safe Evaluation Started")
            # p_docker = subprocess.Popen(['bash', '../server-run.sh'], stdout=DEVNULL, stderr=DEVNULL)
            # sleep(DOCKER_WAIT_TIME)
            p_a_ = subprocess.Popen(['python', 'agent.py', '--id=a', f'--run_name={args.run_name}', '--port=3001']) #, stdout=DEVNULL, stderr=DEVNULL)
            # p_b = subprocess.Popen(['python', 'smart_safe_agent.py', '--id=b', f'--run_name={args.run_name}', '--log=True', f'--ep={i}'], stdout=DEVNULL, stderr=DEVNULL)
            p_b_ = subprocess.Popen(['python', 'smart_safe_agent.py', '--id=b', f'--run_name={args.run_name}', '--log=True', f'--ep={i}', '--port=3001'], stdout=subprocess.PIPE, stderr=DEVNULL)
            # [p.wait() for p in (p_docker, p_a, p_b)]
            # print("Smart Safe Evaluation Completed")

            p_docker.wait()
            print("Evaluation Completed")

            static_result = 1 if p_b.stdout.readlines()[0].strip().decode('utf-8')[-1] == 'a' else 0
            smart_result = 1 if p_b_.stdout.readlines()[0].strip().decode('utf-8')[-1] == 'a' else 0

            writer = SummaryWriter(f'runs/{args.run_name}')
            writer.add_scalar('win/eval/smart_safe', smart_result, i)
            writer.add_scalar('win/eval/static', static_result, i)
            writer.flush()
            writer.close()

        # p_docker = subprocess.Popen(['bash', '../server-run-2.sh'])
        # sleep(DOCKER_WAIT_TIME)
        # p_a = subprocess.Popen(['python', 'agent.py', '--id=a', f'--ep={i}', '--port=3001'], stdout=DEVNULL, stderr=DEVNULL)
        # p_b = subprocess.Popen(['python', 'agent.py', '--id=b', '--port=3001'])
