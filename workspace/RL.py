import sys
import pandas as pd
import sys
sys.path.append('/Users/ben/Dropbox/git/udacity/workspace/agents')
sys.path.append('/Users/ben/Dropbox/git/udacity/workspace/tasks')
sys.path.append('/Users/ben/Dropbox/git/udacity/workspace')

from agent import DDPG
from takeoff import Task
import numpy as np


num_episodes = 500
target_pos = np.array([0., 0., 100.])
task = Task(target_pos=target_pos)
agent = DDPG(task)
worst_score = 1000000
best_score = -1000000.
reward_log = "reward.txt"

reward_labels = ['episode', 'reward']
reward_results = {x : [] for x in reward_labels}



for i_episode in range(1, num_episodes+1):

    state = agent.reset_episode() # start a new episode
    score = 0
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        score += reward
        best_score = max(best_score , score)
        worst_score = min(worst_score , score)
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f} , worst = {:7.3f})".format(
               i_episode, score, best_score, worst_score), end="")
            break
    reward_results['episode'].append(i_episode)
    reward_results['reward'].append(score)
    sys.stdout.flush()
