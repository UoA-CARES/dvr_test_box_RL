
import copy
import torch.optim as optim
from networks_architectures_v3 import Actor, Critic

class TD3:
    def __init__(self, device):

        # -------- Hyper-parameters --------------- #
        self.device = device

        self.critic_learning_rate = 3e-4 # 1e-3
        self.actor_learning_rate  = 3e-4 # 1e-4

        self.gamma = 0.99  # discount factor
        self.tau   = 0.005

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.obs_dim = 14
        self.act_dim = 4

        self.hidden_size = [1024, 1024]

        # ---------- Initialization and build the networks ----------- #
        # Main networks
        self.actor  = Actor(self.obs_dim,  self.hidden_size, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim, self.hidden_size, self.act_dim).to(self.device)

        # Target networks
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)


        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        self.actor.train(True)
        self.critic.train(True)



