
import time
import gym
import random

from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Run DDPG on Pendulum')
parser.add_argument('--gpu', help='Use GPU', action='store_true')
args = parser.parse_args()


LOW_BOUND = -2
HIGH_BOUND = 2

STATE_SIZE = 3      # state vector size
ACTION_SIZE = 1     # action vector size (single-valued because actions are continuous in the interval (-2, 2))

MEMORY_CAPACITY = 1000000

BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE_CRITIC = 0.001
LEARNING_RATE_ACTOR = 0.001
TAU = 0.001

# Noise parameters
# Scale of the exploration noise process (1.0 is the range of each action dimension)
NOISE_SCALE_INIT = 0.1

# Decay rate (per episode) of the scale of the exploration noise process
NOISE_DECAY = 0.99

# Parameters for the exploration noise process:
# dXt = theta*(mu-Xt)*dt + sigma*dWt
EXPLO_MU = 0.0
EXPLO_THETA = 0.
EXPLO_SIGMA = 0.

MAX_STEPS = 200
MAX_EPISODES = 400
EPS = 0.001



# Choose device cpu or cuda if a gpu is available
if args.gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
device = torch.device(device)


env = gym.make("Pendulum-v0")

# Replay memory function
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_critic(nn.Module):

    def __init__(self, input_size):
        super(DQN_critic, self).__init__()

        self.hidden1 = nn.Linear(input_size, 8)
        self.hidden2 = nn.Linear(8, 8)
        self.hidden3 = nn.Linear(8, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        return self.output(x.view(x.size(0), -1))


class DQN_actor(nn.Module):

    def __init__(self, state_size):
        super(DQN_actor, self).__init__()

        self.hidden1 = nn.Linear(state_size, 8)
        self.hidden2 = nn.Linear(8, 8)
        self.hidden3 = nn.Linear(8, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.output(x)
        x = (torch.sigmoid(x)* (HIGH_BOUND - LOW_BOUND)) + LOW_BOUND
        return x.view(x.size(0), -1)


# Soft target update function
def update_targets(target, original):
        """Weighted average update of the target network and original network
            Inputs: target actor(critic) and original actor(critic)"""

        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + TAU*orgParam.data)
            
##### Etienne : compute the max of the action value function for a given state
NB_SAMPLES = 10

# Compute a "max" over NB_SAMPLES random actions (very poor approximation of the max)
def compute_max_poor(next_state_action):
    max_tot_value = - np.inf
    max_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        for i in range(NB_SAMPLES):
            action_sample = torch.rand(BATCH_SIZE) * (HIGH_BOUND - LOW_BOUND) + LOW_BOUND
            next_state_action[:,3] = action_sample
            temp_values = target_critic_nn(next_state_action)
            tot_temp_value = torch.sum(temp_values)
            if tot_temp_value > max_tot_value:
                max_tot_value = tot_temp_value
                max_values = temp_values
    return max_values

# Compute the max value for each state : well better approximation of the max
def compute_max(next_state_action):
    max_values = torch.ones(BATCH_SIZE)*(-np.inf)
    with torch.no_grad():
        for i in range(NB_SAMPLES):
            action_sample = torch.rand(BATCH_SIZE) * (HIGH_BOUND - LOW_BOUND) + LOW_BOUND
            next_state_action[:,3] = action_sample
            temp_values = target_critic_nn(next_state_action)
            for j in range(BATCH_SIZE):
                if temp_values[j] > max_values[j]:
                    max_values[j] = temp_values[j]
    return max_values            


def optimize_model():

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)

    # Divide memory into different tensors
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).view(BATCH_SIZE, 1)
    reward_batch = torch.cat(batch.reward)

    # Create state-action (s,a) tensor for input into the critic network with taken actions
    state_action = torch.cat([state_batch, action_batch], -1)

    # Compute Q(s,a) using critic network
    state_action_values = critic_nn(state_action)

    # Compute deterministic next state action using actor target network
    next_action = target_actor_nn(non_final_next_states).detach()

    # Compute next timestep state-action (s,a) tensor for non-final next states
    next_state_action = torch.zeros(BATCH_SIZE, 4, device=device)
    #next_state_action[non_final_mask, :] = torch.cat([non_final_next_states, next_action], -1)
    
    ################# Etienne #################
    next_state_action[non_final_mask,:3] = non_final_next_states
    max_state_values = compute_max(next_state_action)

    # Compute next state values at t+1 using target critic network
    #next_state_values = target_critic_nn(next_state_action).detach()

    # Compute expected state action values y[i]= r[i] + Q'(s[i+1], a[i+1])
    expected_state_action_values = reward_batch.view(BATCH_SIZE, 1) + GAMMA*max_state_values
    #expected_state_action_values = reward_batch.view(BATCH_SIZE, 1) + GAMMA*next_state_values

    # Critic loss by mean squared error
    loss_critic = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the critic network
    optimizer_critic.zero_grad()
    loss_critic.backward()

    for param in critic_nn.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer_critic.step()

    #optimize actor
    # Actor actions
    state_actor_action = actor_nn(state_batch)
    # State-actor-actions tensor
    state_actor_action_values = torch.cat([state_batch, state_actor_action], -1)
    # Loss
    loss_actor = -1 * torch.mean(critic_nn(state_actor_action_values))
    optimizer_actor.zero_grad()
    loss_actor.backward()
    for param in actor_nn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer_actor.step()

    # Soft parameter update
    update_targets(target_critic_nn, critic_nn)
    update_targets(target_actor_nn, actor_nn)



# Initialize neural nets
# Critic net with input (s,a) tensor and output a single q value for that state-action pair
critic_nn = DQN_critic(STATE_SIZE + ACTION_SIZE).to(device)
target_critic_nn = DQN_critic(STATE_SIZE + ACTION_SIZE).to(device)

# Actor net: state input -- action output bounded from lower bound to high bound
actor_nn = DQN_actor(STATE_SIZE).to(device)
target_actor_nn = DQN_actor(STATE_SIZE).to(device)

# Initialize replay memory
memory = ReplayMemory(MEMORY_CAPACITY)

target_critic_nn.load_state_dict(critic_nn.state_dict())
optimizer_critic = optim.Adam(critic_nn.parameters(), lr=LEARNING_RATE_CRITIC)
target_critic_nn.eval()

target_actor_nn.load_state_dict(actor_nn.state_dict())
optimizer_actor = optim.Adam(actor_nn.parameters(), lr=LEARNING_RATE_ACTOR)
target_actor_nn.eval()

## Benchmark parameter
MAX_TIME_SEC = 600
reward_time = np.zeros(MAX_TIME_SEC)


def main():

    episode_reward = [0]*MAX_EPISODES
    nb_total_steps = 0
    time_beginning = time.time()

    try:

        for i_episode in range(MAX_EPISODES):

            if i_episode % 10 == 0:
                print("Episode ", i_episode)
                
            print("Episode {} : reward  = ".format(i_episode),end='')

            state = env.reset()
            state = torch.tensor([state], dtype=torch.float, device=device)

            # Initialize exploration noise process, parameters in parameters file
            noise_process = np.zeros(ACTION_SIZE)
            noise_scale = (NOISE_SCALE_INIT * NOISE_DECAY**EPS) * (HIGH_BOUND - LOW_BOUND)
            done = False
            step = 0

            while not done and step < MAX_STEPS:

                action = actor_nn(state).detach() # deterministic choice of a using actor network

                # Add temporally-correlated exploration noise to action (using an Ornstein-Uhlenbeck process)
                noise_process = EXPLO_THETA * (EXPLO_MU - noise_process) + EXPLO_SIGMA * np.random.randn(ACTION_SIZE)
                noise = noise_scale*noise_process
                action += torch.tensor([noise[0]], dtype=torch.float, device=device)

                # Perform an action
                next_state, reward, done, _ = env.step(action)
                next_state = torch.tensor([next_state], dtype=torch.float, device=device)
                reward = torch.tensor([reward], dtype=torch.float, device=device)

                episode_reward[i_episode] += reward.item()

                # Save transition into memory
                memory.push(state, action, next_state, reward)
                state = next_state

                optimize_model()

                step += 1
                nb_total_steps += 1
                
            print(episode_reward[i_episode])
            time_execution = time.time() - time_beginning
            i_sec = int(time_execution)
            if i_sec < MAX_TIME_SEC:
                reward_time[i_sec] = episode_reward[i_episode]

                #env.render()

    except KeyboardInterrupt:
        pass

    time_execution = time.time() - time_beginning

    print('---------------------------------------------------')
    print('---------------------STATS-------------------------')
    print('---------------------------------------------------')
    print(nb_total_steps, ' steps and updates of the network done')
    print(MAX_EPISODES, ' episodes done')
    print('Execution time ', round(time_execution, 2), ' seconds')
    print('---------------------------------------------------')
    print('Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s')
    print('Average duration of one episode : ', round(time_execution/MAX_EPISODES, 3), 's')
    print('---------------------------------------------------')

    plt.plot(episode_reward[:i_episode])
    plt.show()
    
    return reward_time[:i_sec]


if __name__ == '__main__':
    main()
