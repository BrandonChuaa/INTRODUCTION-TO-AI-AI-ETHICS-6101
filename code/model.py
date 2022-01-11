import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_memory import ReplayMemory


class DQN(nn.Module):
    """
    Module of the DQN (Deep Q Network), network portion
    """

    def __init__(self, input_space_dim, action_num=6, lr=1e-4, batch_norm=True, save=True, save_name='', load_name=''):
        """
        Inputs: 
        input_space_dim: the built game enviornment's observation space's shape
        action_num: int, number of valid player actions (6 for space invaders)
        lr: learning rate
        batch_norm: bool, use batch normalization or not
        save: bool, to save the trained model or not
        """
        super(DQN, self).__init__()
        self.batch_norm = batch_norm
        self.save = save
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        if save is True:
            self.save_file = os.path.join(
                cur_dir, ".\\models\\", save_name, time.strftime('%Y%m%d-%H:%M', time.gmtime()))
        else:
            self.save_file = None
        self.load_file = os.path.join(
                cur_dir, ".\\models\\", save_name, load_name)
        # conv input
        self.conv1 = nn.Conv2d(input_space_dim[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # batch norm
        if batch_norm is True:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
        # fully connect
        conv_out_total_dim = self._conv_out(input_space_dim)
        self.fc1 = nn.Linear(conv_out_total_dim, 512)
        self.fc2 = nn.Linear(512, action_num)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _conv_out(self, input_space_dim):
        """
        Calculate convolution layer output size
        """
        zero = torch.zeros(1, *input_space_dim)
        dim = self.conv3(self.conv2(self.conv1(zero)))
        return int(np.prod(dim.size()))

    def forward(self, state):
        if self.batch_norm is True:
            conv1 = F.relu(self.bn1(self.conv1(state)))
            conv2 = F.relu(self.bn2(self.conv2(conv1)))
            conv3 = F.relu(self.bn3(self.conv3(conv2)))
        else:
            conv1 = F.relu(self.conv1(state))
            conv2 = F.relu(self.conv2(conv1))
            conv3 = F.relu(self.conv3(conv2))
        conv_out = conv3.view(conv3.size()[0], -1)
        fc1_out = F.relu(self.fc1(conv_out))
        fc2_out = self.fc2(fc1_out)

        return fc2_out

    def save_model(self):
        print("Saving Model to: {}".format(self.save_file))
        torch.save(self.state_dict(), self.save_file)

    def load_model(self):
        print("Loading Model from: {}".format(self.load_file))
        self.load_state_dict(torch.load(self.load_file))


class Agent(object):
    """
    Main class for the DQN agent
    """

    def __init__(self, input_space_dim, memory_size, batch_size, action_num=6, gamma=0.99,
                 epsilon_min=0.1, epsilon_decay=5e-6, replace_count=1000, lr=1e-4, batch_norm=True, save=True, algo='DQN'):
        """
        Inputs: 
        input_space_dim: the built game enviornment's observation space's shape
        action_num: int, number of valid player actions (6 for space invaders)
        memory_size: max number of stored steps
        batch_size: batch of steps in training
        gamma: discount factor
        epsilon_min: minimum epsilon to stop decaying
        epsilon_decay: rate of epsilon decay (per step)
        replace_count: threshold number of steps to replace "target" network with "online"
        lr: learning rate
        batch_norm: bool, use batch normalization or not
        save: bool, to save the trained model or not
        algo: to select between DQN and Double DQN
        """
        self.input_space_dim = input_space_dim
        self.action_num = action_num
        self.batch_size = batch_size
        self.action_num = action_num
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replace_count = replace_count
        self.batch_norm = batch_norm
        self.save = save
        self.lr = lr
        self.algo = algo
        self.action_space = [i for i in range(action_num)]
        self.learn_step_counter = 0
        self.memory = ReplayMemory(memory_size)

        self.online_net = DQN(input_space_dim=self.input_space_dim, action_num=self.action_num, lr=lr,
                              batch_norm=self.batch_norm, save=self.save, save_name='online')
        self.target_net = DQN(input_space_dim=self.input_space_dim, action_num=self.action_num, lr=lr,
                              batch_norm=self.batch_norm, save=self.save, save_name='target')
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
    def choose_action(self, observation):
        # pylint: disable=not-callable,no-member
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(self.device)
            actions = self.online_net.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, next_state, final):
        self.memory.store_transition(
            state=state, action=action, reward=reward, next_state=next_state, final=final)

    def sample_from_memory(self):
        return self.memory.sample_buffer(self.batch_size)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_count == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def save_models(self):
        self.online_net.save_model()
        self.target_net.save_model()

    def load_models(self):
        self.online_net.load_model()
        self.target_net.load_model()

    def learn(self):
        # record the first batch pure random actions
        if self.memory.memory_counter <= self.batch_size:
            return

        self.online_net.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, next_states, finals = self.sample_from_memory()
        indices = torch.LongTensor(np.arange(self.batch_size)).to(self.device)

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        finals = torch.tensor(finals).to(self.device)

        if self.algo == 'DQN':
            pred_q = self.online_net.forward(states)[indices, actions]
            next_q = self.target_net.forward(next_states).max(dim=1)[0]

            next_q[finals] = 0.0
            target_q = rewards + self.gamma*next_q
        elif self.algo == 'DDQN':
            pred_q = self.online_net.forward(states)[indices, actions]
            next_q_tar = self.target_net.forward(next_states)
            next_q_on = self.online_net.forward(next_states)

            max_actions = torch.argmax(next_q_on, dim=1)
            next_q_tar[finals] = 0.0

            target_q = rewards + self.gamma*next_q_tar[indices, max_actions]
        else:
            raise "Algorithm have to be eith DQN or DDQN"

        loss = self.online_net.loss(
            target_q, pred_q).to(self.device)
        loss.backward()
        self.online_net.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
