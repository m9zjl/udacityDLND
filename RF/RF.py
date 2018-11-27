import gym
from collections import defaultdict
import numpy as np
import sys
from tqdm import tqdm
from RF.rf_plot import plot_blackjack_values
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')
print(env.observation_space)
print(env.action_space)

for i_episode in range(3):
    state = env.reset()
    while True:
        print(state)
        action = env.action_space.sample()
        print('action', action)
        state, reward, done, info = env.step(action)
        print('new state', state)
        if done:
            print('End game! Reward: ', reward)
            print('You won :)\n') if reward > 0 else print('You lost :(\n')
            break


def generate_episode_from_limit(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        action = 0 if state[0] > 18 else 1
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


episode = generate_episode_from_limit(env)
state, action, reward = zip(*episode)
print('s', state, 'a', action, 'r', reward)
for i, state in enumerate(state):
    print(sum(reward[i:]))

for i in range(3):
    print(generate_episode_from_limit(env))

from collections import defaultdict
import numpy as np
import sys


def mc_prediction_v(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionary of lists
    returns = defaultdict(list)
    # loop over episodes
    for i_episode in tqdm(range(1, num_episodes + 1)):
        # monitor progress
        # if i_episode % 1000 == 0:
        #     print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        #     sys.stdout.flush()
        ## TODO: complete the function
        episode = generate_episode(env)
        state, actions, rewards = zip(*episode)
        discount = np.array([gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(state):
            returns[state].append(sum(rewards[i:] * discount[:-(1 + i)]))
    V = {k: np.mean(v) for k, v in returns.items()}
    return V


mc_prediction_v(env, 1, generate_episode_from_limit)



# obtain the value function
V = mc_prediction_v(env, 500000, generate_episode_from_limit)

# plot the value function

plot_blackjack_values(V)


def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionaries of arrays
    return_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in tqdm(range(1, num_episodes+1)):
        # monitor progress
#         if i_episode % 1000 == 0:
#             print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
#             sys.stdout.flush()
        ## TODO: complete the function
        episode=generate_episode(env)
        states,action,rewards = zip(*episode)
        discount = np.array([gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            return_sum[state][action[i]] += sum(rewards[i:]*discount[:-(1+i)])
            N[state][action[i]] += 1.0
            Q[state][action[i]] = return_sum[state][action[i]]/N[state][action[i]]
    return Q

# obtain the action-value function
Q = mc_prediction_q(env, 500000, generate_episode_from_limit_stochastic)

# obtain the state-value function
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
         for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V_to_plot)