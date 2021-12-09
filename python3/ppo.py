import torch
from torch import nn

ACTION_SPACE_SIZE = 6
OBS_DIM = 11
MAP_SIZE = 15

class PPO(nn.Module):

    def __init__(self, feature_dim=16, p=0.3):
        super(PPO, self).__init__()
        self.shared_layers = nn.Sequential(
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
                                            nn.Dropout(p=p)
        )
        self.actor_layers = nn.Sequential(
                                            nn.Linear(256, ACTION_SPACE_SIZE)
        )

        self.critic_layers = nn.Sequential(
                                            nn.Linear(256, 1)
        )

    def forward(self, tensor):
        features = self.shared_layers(tensor)
        pi = self.actor_layers(features)
        v = self.critic_layers(features)
        return pi, v

if __name__ == "__main__":

    tensor = torch.rand(4, 11, 15, 15)

    ppo = PPO()

    pi, v = ppo(tensor)

    print(pi, v)
