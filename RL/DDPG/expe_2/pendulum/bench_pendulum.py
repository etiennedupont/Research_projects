import numpy as np
import matplotlib.pyplot as plt
import DDPG_pendulum_original
import DDPG_pendulum_modified

reward_time_updated, reward_time = DDPG_pendulum_modified.main(MAX_EPISODES=600)

def plot_rewards():
    max_time = min(len(reward_time), len(reward_time_updated))
    time = np.array([i for i in range(max_time)])
    fig, ax = plt.subplots()
    ax.plot(time,reward_time[:max_time],'b+',label='reward standard method')
    ax.plot(time,reward_time_updated[:max_time],'r+',label='reward new method')
    ax.set(xlabel='Execution time (s)',ylabel='Reward',title='DDPG benchmark on pendulum')
    plt.legend()
    fig.savefig("benchmark_time_reward1.jpg")
    plt.show()
    
def plot_rewards_averaged(n_sec, reward_time, reward_time_updated):
    max_time = min(len(reward_time), len(reward_time_updated))
    reward_time = reward_time[:max_time]
    reward_time_updated = reward_time_updated[:max_time]
    reward_time_averaged = list()
    reward_time_updated_averaged = list()
    j, k = 0, 0
    k0, k0_up = 0., 0. #nb occurences of 0 reward (0 => NC here)
    s, s_up = 0, 0
    while j < max_time:
        while k+j < max_time and k < n_sec:
            if reward_time[k+j] == 0:
                k0 += 1
            if reward_time_updated[k+j] == 0:
                k0_up += 1            
            s += reward_time[k+j]
            s_up += reward_time_updated[k+j]
            k += 1
        if k-k0 == 0:
            reward_time_averaged.append(0)
        else:
            reward_time_averaged.append(s/(k-k0))
        if k-k0_up == 0:
            reward_time_updated_averaged.append(0)
        else:
            reward_time_updated_averaged.append(s_up/(k-k0_up))
        k, k0, k0_up, s, s_up = 0, 0, 0, 0, 0
        j += n_sec   
    time = [(i+1)*n_sec for i in range(int(j / n_sec))]
    fig, ax = plt.subplots()
    ax.plot(time,reward_time_averaged,'b-',label='reward standard method')
    ax.plot(time,reward_time_updated_averaged,'r-',label='reward new method')
    ax.set(xlabel='Execution time (s)',ylabel='Average reward over {} s'.format(n_sec),title='DDPG benchmark on pendulum')
    plt.legend()
    fig.savefig("benchmark_time_reward_averaged.jpg")
    plt.show()
    
plot_rewards_averaged(10,reward_time,reward_time_updated)
