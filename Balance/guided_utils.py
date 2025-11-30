import numpy as np
def compute_domain_reward(state,d):
    reward = 0
    if (state[9+0]-state[0])**2+(state[9+2]-state[2])**2 < d**2:
        reward = 1.
    return reward

def compute_domain_reward_2b(state,d):
    reward = 0
    if state[1]<state[4]:
        if (state[18+0]-state[0])**2+(state[18+2]-state[2])**2 < d**2:
            reward = 1.
    else:
        if (state[18+0]-state[3])**2+(state[18+2]-state[5])**2 < d**2:
            reward = 1.
    return reward

def compute_domain_dis(state):

    return np.sqrt((state[9+0]-state[0])**2+(state[9+2]-state[2])**2)

def compute_music_domain_reward(state,d):
    reward = 0
    if (state[9+0+6]-state[0+6])**2+(state[9+2+6]-state[2+6])**2 < d**2:
        reward = 1.
    return reward

def compute_music_domain_dis(state):
   
    return np.sqrt((state[9+0+6]-state[0+6])**2+(state[9+2+6]-state[2+6])**2)