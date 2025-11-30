import taichi as ti
import numpy as np
import math
import random
from scoop_good_balls.MPM_solver import MPMSolver

import torch
from scoop_good_balls.AE_CNN import AutoEncoder
from torch.autograd import Variable
import gym
torch.set_num_threads(4)

ti.init(arch=ti.gpu) # Try to run on GPU
#CUDA_VISIBLE_DEVICES=['0','1']
class ScoopJellyEnvs(MPMSolver):
    def __init__(self,render=False):
        self.render_flag = render
        self.n_tasks=1
        path = "./scoop_good_balls/Encoder/encoder.pkl"
        self.model=torch.load(path)
        self.model.eval()
        self.count=0
        self.cnt = 0
        self._max_episode_steps = 150
        self.qualitys=np.random.randint(20,21,self.n_tasks)*0.1 #for meta train
        # self.qualitys=np.random.randint(160,161,self.n_tasks)*0.025

        self.Es=np.random.randint(9,10,self.n_tasks)*100.0
        self.quality=float(self.qualitys[0])
        self.E=self.Es[0]
        
        srand=np.random.randint(1,2,size=(self.n_tasks,20))
        self.projections=np.random.randint(1,2,size=self.n_tasks)
        self.projection=self.projections[0]
        self.jelly_phos=srand*0.8
        self.jelly_pho=self.jelly_phos[0]
        #self.jelly_pho=np.array([23, 17,  3, 16,  8, 20, 10,  4, 22, 19,  4, 18])*0.1
        
        srand=np.random.randint(7,8,self.n_tasks)*20.0
        self.fluid_phos=srand*0.01
        self.fluid_pho=self.fluid_phos[0]
        
        srand=np.random.randint(7,8,self.n_tasks)*10.0
        self.gravitys=srand
        self.gravity=self.gravitys[0]
        
        srand=np.zeros([self.n_tasks,6])
        for i in range(self.n_tasks):
            srand[i]=random.sample(range(0, 8), 6)
            #srand[i]=np.array([0,1,2,3])
        #srand=np.random.randint(0,2,size=(self.n_tasks,12))
        self.r_types=srand
        self.r_type=self.r_types[0]
        if self.render_flag:
            self.gui = ti.GUI("MPM SCOOP THE JELLY", res=512, background_color=0x112F41)
        super().__init__(self.quality,self.E,self.jelly_pho,self.fluid_pho,self.gravity,self.projection,self.r_type)
        self.observation_space=self.get_observation_space()
        # self.action_space=np.zeros(3)
    @property
    def action_space(self):
        return gym.spaces.Box(np.array([-1,-1,-1]),np.array([1,1,1]))
    def get_all_task_idx(self):
        return range(self.n_tasks)
    def reset_task(self,idx):

        self.reset_env()
    def normalization(self,data):
        #mu = np.mean(data)
        #sigma = np.std(data)state+1e-7
        
        mu=0
        sigma=1.1
        '''
        mu=2.8998700831383357e-05 
        sigma=0.00015228158427660628
        '''
        return (data - mu) / sigma,mu,sigma
    def get_observation_space(self):
        
        state=self.get_state().reshape([-1,2,128,128])
        state,mu,sigma=self.normalization(state)
        #state[np.isnan(state)] = 0
        mask=np.zeros(state.shape)
        mask[:,:,4:124,4:124]=1
        state[mask==0]=0
        
        sta=torch.from_numpy(state)
        sta = Variable(sta.float().cuda())
        encode,_=self.model(sta)
        encode=encode.cpu().detach().numpy()
        
                
        #decode=decode.cpu().detach().numpy()
        #print(np.mean(np.square(decode-state)))
        
        #print(encode)
        
        a = np.append(self.get_jelly_state(), self.get_rigid_state())

        obs = np.append(a, encode)

        #return obs
        #print(a)
        return obs#a
        
        
    def step(self,action):
        for s in range(int(12.0e-3 / self.dt+0.1)):
            #print(int(12.0e-3 / self.dt+0.1))
            
            self.solve_windmill(action[0]*40,action[1]*40,action[2]*80)
            self.substep()

        obs=self.get_observation_space()
        reward=self.get_reward()
        done=0
        In=self.In

        self.cnt+=1
        if self.cnt==self._max_episode_steps:
           done = 1
        if self.render_flag:
            self.render()
        return obs,reward,done,In
        
    def reset(self):
        #self.o=random.sample(range(0, 5), 5)
        self.cnt = 0
        srand=random.sample(range(0, 8), 6)
        # srand = [0,1,3,4,6,7]
        for i in range(12):
            self.R[i]=0
        
        for i in range(6):
            self.R[int(srand[i])]=1
            
        self.count=0
        self.maxx=0
        self.initialize()

        return self.get_observation_space()
    def render(self):

        color_jelly=np.array([0xff7fff,
                              0xff99ff,
                              0xffb2ff,
                              0xffd1ff,
                              0xfff4ff,0x84bf96,0x000000],dtype=np.uint32)
        self.gui.circles(self.x.to_numpy(), radius=1.5, color=color_jelly[self.color.to_numpy()])
        self.gui.circles(self.windmill_x.to_numpy(), radius=1.5,color=0x845538)
        self.gui.show()






