# -*- coding: utf-8 -*-

from frozenlake import FrozenLakeEnv


env = FrozenLakeEnv(is_slippery=True)

# print the state space and action space
print(env.observation_space)
print(env.action_space)

# print the total number of states and actions
print(env.nS)
print(env.nA)
P = env.P
print(P)
for i in env.P:
    print(env.P[0])



import numpy as np

def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)

    ## TODO: complete the function
    while True:
        delta = 0
        for s in range(env.nS):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V

random_policy = np.ones([env.nS, env.nA]) / env.nA


from plot_utils import plot_values

# evaluate the policy
V = policy_evaluation(env, random_policy)

plot_values(V)