# https://github.com/nikhilbarhate99/PPO-PyTorch

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from glob import glob
import os
import numpy as np

from tqdm.auto import tqdm

from tensorboardX import SummaryWriter

ACTION_SPACE_SIZE = 6
OBS_DIM = 11
MAP_SIZE = 15


################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")




################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


    def fill(self, observation_path='./obs'):

        def unstack(array):
            return [torch.tensor(array[i], dtype=torch.float) for i in range(array.shape[0])]

        npz_files = sorted(glob(os.path.join(observation_path, '*.npz')))

        # print(npz_files)

        print('Filling Buffer')
        for npz_file in tqdm(npz_files):
            obs = np.load(npz_file)
            self.actions.extend(unstack(obs['q_vector']))
            self.states.extend(unstack(obs['observation']))
            self.logprobs.extend(unstack(obs['logp_vector']))
            self.rewards.extend(unstack(obs['r_vector']))
            # self.is_terminals.extend()


class ActorCritic(nn.Module):
    # def __init__(self, state_dim, action_dim, has_continuous_action_space=False, action_std_init=None):
    def __init__(self, feature_dim=16, p=0.3, has_continuous_action_space=False, action_std_init=None):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = ACTION_SPACE_SIZE
            self.action_var = torch.full((ACTION_SPACE_SIZE,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            # self.actor = nn.Sequential(
            #                 nn.Linear(state_dim, 64),
            #                 nn.Tanh(),
            #                 nn.Linear(64, 64),
            #                 nn.Tanh(),
            #                 nn.Linear(64, action_dim),
            #             )
            pass
        else:
            self.actor = nn.Sequential(
                            nn.Conv2d(OBS_DIM, feature_dim, 3), 
                            nn.BatchNorm2d(feature_dim),
                            nn.LeakyReLU(),
                            nn.Conv2d(feature_dim, feature_dim, 3), 
                            nn.BatchNorm2d(feature_dim),
                            nn.LeakyReLU(),
                            nn.Conv2d(feature_dim, feature_dim, 3), 
                            nn.BatchNorm2d(feature_dim),
                            nn.LeakyReLU(),
                            nn.Flatten(),
                            nn.Linear(9*9*feature_dim, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Dropout(p=p),
                            nn.Linear(512, 256),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(p=p),
                            nn.Linear(256, ACTION_SPACE_SIZE),
                            nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
                            nn.Conv2d(OBS_DIM, feature_dim, 3), 
                            nn.BatchNorm2d(feature_dim),
                            nn.LeakyReLU(),
                            nn.Conv2d(feature_dim, feature_dim, 3), 
                            nn.BatchNorm2d(feature_dim),
                            nn.LeakyReLU(),
                            nn.Conv2d(feature_dim, feature_dim, 3), 
                            nn.BatchNorm2d(feature_dim),
                            nn.LeakyReLU(),
                            nn.Flatten(),
                            nn.Linear(9*9*feature_dim, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Dropout(p=p),
                            nn.Linear(512, 256),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(p=p),
                            nn.Linear(256, 1)
                    )
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    # def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
    # def __init__(self, feature_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space=False, action_std_init=0.6):
    def __init__(self, feature_dim, lr_actor, lr_critic, K_epochs, eps_clip, run_name,
        has_continuous_action_space=False, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        # self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        # self.policy = ActorCritic(feature_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy = ActorCritic(feature_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        # self.policy_old = ActorCritic(feature_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old = ActorCritic(feature_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.writer = SummaryWriter(f'runs/{run_name}')


    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)

            return action.item(), action_logprob.item()


    def update(self, starting_step=0):

        # Monte Carlo estimate of returns
        # rewards = []
        discounted_reward = 0
        # for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)
            
        rewards = self.buffer.rewards

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        print("Policy Update")
        for step in tqdm(range(starting_step, starting_step + self.K_epochs)):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            v_loss = 0.5 * self.MseLoss(state_values, rewards)
            loss = -torch.min(surr1, surr2) + v_loss - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            self.writer.add_scalar('loss', loss.mean().item(), step)
            self.writer.add_scalar('value_loss', v_loss.mean().item(), step)
            self.writer.add_scalar('entropy', dist_entropy.mean().item(), step)
            
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        return starting_step + self.K_epochs 
        # clear buffer
        # self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
    def fill_rollout_buffer(self, observation_path='./obs'):
        self.buffer.clear()
        self.buffer.fill(observation_path)

        
if __name__ == "__main__":


    K_epochs = 80           # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    # gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    feature_dim = 16

    ppo = PPO(feature_dim, lr_actor, lr_critic, K_epochs, eps_clip)
    

    # tensor = torch.rand(4, 11, 15, 15)
    tensor = torch.rand(1, 11, 15, 15)

    ppo.policy.eval()
    ppo.policy_old.eval()

    action, log_prob = ppo.select_action(tensor)
    
    action_tensor = torch.tensor(action).reshape([1, 1])

    logprobs, state_values, dist_entropy = ppo.policy.evaluate(tensor, action_tensor)

    print(action_tensor, logprobs, state_values, dist_entropy)

    ppo.policy.train()
    ppo.policy_old.train()

    random_seed = 0

    env_name = 'Bomberland'

    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    ppo.load(checkpoint_path)

    ppo.fill_rollout_buffer()
    ppo.update()

    print("--------------------------------------------------------------------------------------------")
    print("saving model at : " + checkpoint_path)
    ppo.save(checkpoint_path)
    print("model saved")
    print("--------------------------------------------------------------------------------------------")


