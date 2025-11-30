import numpy as np
import torch
import copy


import numpy as np
import torch
import copy

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        ######
        self.state_2 = np.zeros((max_size, state_dim))
        self.action_2 = np.zeros((max_size, action_dim))
        self.next_state_2 = np.zeros((max_size, state_dim))
        self.reward_2 = np.zeros((max_size, 1))
        self.not_done_2 = np.zeros((max_size, 1))
        ######
        
        self.device = device

    def add(self, state, action, next_state, reward, done, state_2, action_2, next_state_2, reward_2, done_2):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        ######
        self.state_2[self.ptr] = state_2
        self.action_2[self.ptr] = action_2
        self.next_state_2[self.ptr] = next_state_2
        self.reward_2[self.ptr] = reward_2
        self.not_done_2[self.ptr] = 1. - done_2
        ######

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def store_eposide(self, state, action, next_state, reward, done,isIn, state_2, action_2, next_state_2, reward_2, done_2,isIn_2):  
            state1=copy.deepcopy(state)
            action1=copy.deepcopy(action)
            next_state1=copy.deepcopy(next_state)
            reward1=copy.deepcopy(reward)
            isIn1=copy.deepcopy(isIn)
            state_21=copy.deepcopy(state_2)
            action_21=copy.deepcopy(action_2)
            next_state_21=copy.deepcopy(next_state_2)
            reward_21=copy.deepcopy(reward_2)
            isIn_21=copy.deepcopy(isIn_2)
            length=len(state1)
            for i in range(length):
                bad=[]
                good=[]
                xb=0
                xg=0
                xb1=0
                xg1=0
                for j in range(8):
                    if isIn[i][j]==1 and j>1:# bad inside
                            bad.append(j)
                    if isIn[i][j]==0 and j<=1:# good outside
                            good.append(j)
                L1=min(len(bad),len(good))
                for j in range(L1):
                    next_state1[i][good[j]*2]=next_state[i][bad[j]*2]
                    next_state1[i][good[j]*2+1]=next_state[i][bad[j]*2+1]
                    next_state1[i][bad[j]*2]=next_state[i][good[j]*2]
                    next_state1[i][bad[j]*2+1]=next_state[i][good[j]*2+1]
                    next_state1[i][24+good[j]*2]=next_state[i][24+bad[j]*2]
                    next_state1[i][24+good[j]*2+1]=next_state[i][24+bad[j]*2+1]
                    next_state1[i][24+bad[j]*2]=next_state[i][24+good[j]*2]
                    next_state1[i][24+bad[j]*2+1]=next_state[i][24+good[j]*2+1]
                    
                    state1[i][good[j]*2]=state[i][bad[j]*2]
                    state1[i][good[j]*2+1]=state[i][bad[j]*2+1]
                    state1[i][bad[j]*2]=state[i][good[j]*2]
                    state1[i][bad[j]*2+1]=state[i][good[j]*2+1]
                    state1[i][24+good[j]*2]=state[i][24+bad[j]*2]
                    state1[i][24+good[j]*2+1]=state[i][24+bad[j]*2+1]
                    state1[i][24+bad[j]*2]=state[i][24+good[j]*2]
                    state1[i][24+bad[j]*2+1]=state[i][24+good[j]*2+1]
                    
                    
                    isIn1[i][good[j]]=1
                    isIn1[i][bad[j]]=0
                
                for j in range(8):
                    if isIn[i][j] and j>1:# bad inside
                            xb+=1
                    if isIn[i][j] and j<=1:# good inside
                            xg+=1
                    if isIn1[i][j] and j>1:# bad inside
                            xb1+=1
                    if isIn1[i][j] and j<=1:# good inside
                            xg1+=1
                if xg>0 or xb>0:
                    reward1[i]=reward1[i]*(xg1*xg1-0.4*xb1*xb1)/(xg*xg-0.4*xb*xb)
                
                
                bad=[]
                good=[]
                xb=0
                xg=0
                xb1=0
                xg1=0
                for j in range(8):
                    if isIn_2[i][j]==1 and j>1:# bad inside
                            bad.append(j)
                    if isIn_2[i][j]==0 and j<=1:# good outside
                            good.append(j)
                L=min(len(bad),len(good))
                for j in range(L):
                    next_state_21[i][good[j]*2]=next_state_2[i][bad[j]*2]
                    next_state_21[i][good[j]*2+1]=next_state_2[i][bad[j]*2+1]
                    next_state_21[i][bad[j]*2]=next_state_2[i][good[j]*2]
                    next_state_21[i][bad[j]*2+1]=next_state_2[i][good[j]*2+1]
                    next_state_21[i][24+good[j]*2]=next_state_2[i][24+bad[j]*2]
                    next_state_21[i][24+good[j]*2+1]=next_state_2[i][24+bad[j]*2+1]
                    next_state_21[i][24+bad[j]*2]=next_state_2[i][24+good[j]*2]
                    next_state_21[i][24+bad[j]*2+1]=next_state_2[i][24+good[j]*2+1]
                    
                    state_21[i][good[j]*2]=state_2[i][bad[j]*2]
                    state_21[i][good[j]*2+1]=state_2[i][bad[j]*2+1]
                    state_21[i][bad[j]*2]=state_2[i][good[j]*2]
                    state_21[i][bad[j]*2+1]=state_2[i][good[j]*2+1]
                    state_21[i][24+good[j]*2]=state_2[i][24+bad[j]*2]
                    state_21[i][24+good[j]*2+1]=state_2[i][24+bad[j]*2+1]
                    state_21[i][24+bad[j]*2]=state_2[i][24+good[j]*2]
                    state_21[i][24+bad[j]*2+1]=state_2[i][24+good[j]*2+1]
                    
                    
                    isIn_21[i][good[j]]=1
                    isIn_21[i][bad[j]]=0
                
                for j in range(8):
                    if isIn_2[i][j] and j>1:# bad inside
                            xb+=1
                    if isIn_2[i][j] and j<=1:# good inside
                            xg+=1
                    if isIn_21[i][j] and j>1:# bad inside
                            xb1+=1
                    if isIn_21[i][j] and j<=1:# good inside
                            xg1+=1
                if xg>0 or xb>0:
                    reward_21[i]=reward_21[i]*(xg1*xg1-0.4*xb1*xb1)/(xg*xg-0.4*xb*xb)
                # store
                if L1>0:
                    self.state[self.ptr] = state1[i]
                    self.action[self.ptr] = action1[i]
                    self.next_state[self.ptr] = next_state1[i]
                    self.reward[self.ptr] = reward1[i]
                    self.not_done[self.ptr] = 1. - done[i]
                    
                    self.state_2[self.ptr] = state_21[i]
                    self.action_2[self.ptr] = action_21[i]
                    self.next_state_2[self.ptr] = next_state_21[i]
                    self.reward_2[self.ptr] = reward_21[i]
                    self.not_done_2[self.ptr] = 1. - done_2[i]

                    self.ptr = (self.ptr + 1) % self.max_size
                    self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.state_2[ind]).to(self.device),
            torch.FloatTensor(self.action_2[ind]).to(self.device),
            torch.FloatTensor(self.next_state_2[ind]).to(self.device),
            torch.FloatTensor(self.reward_2[ind]).to(self.device),
            torch.FloatTensor(self.not_done_2[ind]).to(self.device)
        )
    