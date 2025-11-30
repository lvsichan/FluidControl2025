import numpy as np
import torch
import copy


import numpy as np
import torch
import copy

# class ReplayBuffer(object):
# 	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
# 		self.max_size = max_size
# 		self.ptr = 0
# 		self.size = 0

# 		self.state = np.zeros((max_size, state_dim))
# 		self.action = np.zeros((max_size, action_dim))
# 		self.next_state = np.zeros((max_size, state_dim))
# 		self.reward = np.zeros((max_size, 1))
# 		self.not_done = np.zeros((max_size, 1))

# 		self.eposide_to_end_len = np.zeros((max_size, 1))
# 		self.eposide_start = 0
# 		self.eposide_end = 0
        
# 		self.device = device

# 	def add(self, state, action, next_state, reward, done):
# 		self.state[self.ptr] = state
# 		self.action[self.ptr] = action
# 		self.next_state[self.ptr] = next_state
# 		self.reward[self.ptr] = reward
# 		self.not_done[self.ptr] = 1. - done

# 		self.eposide_end += 1nt(her_goal[i,0])
        # print(tmpidx)
# 	def sample(self, batch_size):
# 		ind = np.random.randint(0, self.size, size=int(batch_size/5))
# 		# K-feature
# 		tmp_rest_len = self.eposide_to_end_len[ind]
# 		# print(tmp_rest_len.shape)
# 		k = 4
# 		her_state =  copy.deepcopy(self.state[ind])
# 		her_action = copy.deepcopy(self.action[int(her_goal[i,0])
        # print(tmpidx)), 2))
# 			achieve_goal = np.zeros((int(batch_size/5), 2))
# 			her_goal[:,1] = self.next_state[feature%self.max_size][:,4]
# 			# print(her_goal.shape)
# 			achieve_goal[:,1] = self.next_state[ind][:,4]
# 			# print(achieve_goal)
# 			tmp_state = copy.deepcopy(self.state[ind])
# 			tmp_next_state = copy.deepcopy(self.next_state[ind])
# 			tmp_state[:,0:2] = her_goal
# 			tmp_state[:,3] = 0
# 			tmp_next_state[:,0:2] = her_goal
# 			tmp_next_state[:,3] = 0
# 			#compute reward
# 			tmp_reward = compute_reward(achieve_goal,her_goal)
# 			# print(tmp_reward.shape)
# 			her_state = np.append(her_state,tmp_state,axis=0)
# 			her_action = np.append(her_action,self.action[ind],axis=0)
# 			her_next_state = np.append(her_next_state,tmp_next_state,axis=0)
# 			her_reward = np.append(her_reward,tmp_reward,axis=0)
# 			her_not_done = np.append(her_not_done,self.not_done[ind],axis=0)


# 		return (
# 			torch.FloatTensor(her_state).to(self.device),
# 			torch.FloatTensor(her_action).to(self.device),
# 			torch.FloatTensor(her_next_state).to(self.device),
# 			torch.FloatTensor(her_reward).to(self.device),
# 			torch.FloatTensor(her_not_done).to(self.device)
# 		)

# def compute_reward(achieve_goal,desire_goal):
# 	dist = (achieve_goal[:,0] - desire_goal[:,0]) **2 + (achieve_goal[:,1] - desire_goal[:,1]) **2 
            
# 	reward = 2.5 * np.exp(-30 * dist)

# 	return reward.reshape(-1,1)




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

        # self.eposide_to_end_len = np.zeros((max_size, 1))
        # self.eposide_start = 0
        # self.eposide_end = 0
        
        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def store_eposide(self, state, action, next_state, reward, done,isIn):  
            state1=copy.deepcopy(state)
            action1=copy.deepcopy(action)
            next_state1=copy.deepcopy(next_state)
            reward1=copy.deepcopy(reward)
            isIn1=copy.deepcopy(isIn)
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
                L=min(len(bad),len(good))
                for j in range(L):
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
                
                # store
                if L>0:
                    self.state[self.ptr] = state1[i]
                    self.action[self.ptr] = action1[i]
                    self.next_state[self.ptr] = next_state1[i]
                    self.reward[self.ptr] = reward1[i]
                    self.not_done[self.ptr] = 1. - done[i]

                    self.ptr = (self.ptr + 1) % self.max_size
                    self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
