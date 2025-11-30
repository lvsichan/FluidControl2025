import taichi as ti
import numpy as np
import math
import random
from scoop_balls.MPM_solver import MPMSolver

import torch
from scoop_balls.AE_CNN import AutoEncoder
from torch.autograd import Variable
import gym
ti.init(arch=ti.gpu,ad_stack_size=512)  # Try to run on GPU
#CUDA_VISIBLE_DEVICES=['0','1']
torch.set_num_threads(4)

seed = 1024
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)   
random.seed(seed)    
np.random.seed(seed)  

class ScoopJellyEnvs(MPMSolver):
    def __init__(self,render=False):
        self.render_flag = render
        self.n_tasks=1
        self.cnt=0
        # self.model = torch.load(path)
        path = "./scoop_balls/Encoder/encoder.pkl"
        self.model=torch.load(path,map_location=torch.device('cuda'))
        self.model.eval()
        scale=0.8
        self.summ=0
        self.epo=0

        self._max_episode_steps = 150

        self.qualitys=np.random.randint(20,21,self.n_tasks)*0.1 #for meta train
        # self.qualitys=np.random.randint(80,81,self.n_tasks)*0.1 #for meta eval
        self.Es=np.random.randint(14,15,self.n_tasks)*100.0
        self.aas=np.random.randint(10,11,self.n_tasks)*0.1
        self.quality=float(self.qualitys[0])
        self.E=self.Es[0]
        
        srand=np.random.randint(1,4,size=(self.n_tasks,12))*0.4+1.0
        self.projections=np.random.randint(1,2,size=self.n_tasks)
        self.projection=self.projections[0]
        self.jelly_phos=srand*1.0
        self.jelly_pho=self.jelly_phos[0]
        
        srand=np.random.randint(8,9,self.n_tasks)*40.0
        self.fluid_phos=srand*0.01
        self.fluid_pho=self.fluid_phos[0]
        
        srand=np.zeros([self.n_tasks,3])
        for i in range(self.n_tasks):
            srand[i][0]=scale*(np.random.randint(0,3)*0.04+0.08)#length
            srand[i][1]=scale*(np.random.randint(0,1)*0.06+0.00)#height
            srand[i][2]=scale*(np.random.randint(0,3)*0.04+0.12)#bottom

        self.shapes=srand
        self.shape=self.shapes[0]
        
        srand=np.random.randint(3,4,self.n_tasks)*20.0#-20.0
        self.gravitys=srand
        self.gravity=self.gravitys[0]
        
        srand=np.zeros([self.n_tasks,10])
        for i in range(self.n_tasks):
            srand[i]=random.sample(range(0, 10), 10)
        #srand=np.random.randint(0,2,size=(self.n_tasks,12))
        self.r_types=srand
        self.r_type=self.r_types[0]
        if self.render_flag:
            self.gui = ti.GUI("MPM SCOOP THE JELLY", res=512, background_color=0x112F41)
        super().__init__(self.quality,self.E,self.jelly_pho,self.fluid_pho,self.gravity,self.projection,self.r_type,self.shape)
        self.observation_space=self.get_observation_space()
        # self.action_space=np.zeros(3)
    @property
    def action_space(self):
        return gym.spaces.Box(np.array([-1,-1,-1]),np.array([1,1,1]))
        
    def get_all_task_idx(self):
        return range(self.n_tasks)
    def reset_task(self,idx):
        
        ti.reset()
        ti.init(arch=ti.gpu) # Try to run on GPU
        self.quality=float(self.qualitys[idx])
        self.E=self.Es[idx]
        self.jelly_pho=self.jelly_phos[idx]
        self.fluid_pho=self.fluid_phos[idx]
        self.gravity=self.gravitys[idx]
        self.projection=self.projections[idx]
        self.r_type=self.r_types[idx]
        self.shape=self.shapes[idx]
        self.reset_mpm(self.quality,self.E,self.jelly_pho,self.fluid_pho,self.gravity,self.projection,self.r_type,self.shape)
        
        self.reset()

    def get_observation_space(self):
        alpha=0.1
        state=self.get_state().reshape([-1,2,128,128])

        #state[np.isnan(state)] = 0
        mask=np.zeros(state.shape)
        mask[:,:,4:124,4:124]=1
        state[mask==0]=0
        
        sta=torch.from_numpy(state)
        sta = Variable(sta.float().cuda())
        #print(torch.std(sta))
        #print(torch.std(sta))
        sta=sta/(torch.std(sta)+1e-7)
        
        encode=self.model.encode(sta)
        encode=encode.cpu().detach().numpy()
        
        #decode=decode.cpu().detach().numpy()
        #print(np.mean(np.square(decode-state)))
        
        #print(encode)
        
        a = np.append(self.get_jelly_state(), self.get_rigid_state())
        # print(a.shape)
        obs = np.append(a, alpha*encode)

        #return obs
        #print(a)
        return obs#a
        
        
    def step(self,action):
        
        # print(int(80.0e-4 / self.dt+0.1))
        # print(self.dt)
        for s in range(int(80.0e-4 / self.dt+0.1)):
            self.solve_windmill(action[0]*40,action[1]*40,action[2]*80)
            self.substep()
        
        obs=self.get_observation_space()
        reward=self.get_reward()
        done=0
        self.cnt+=1
        
        if self.cnt==self._max_episode_steps:
           done = 1
        if self.render_flag:
            self.render()
        #print(reward)
        # if self.bounded[None]==1:
        #    done=1
        return obs,reward,done,dict(reward=reward)
        
    def reset(self):
        self.o=random.sample(range(0, 5), 5)
        #print(self.maxx[None])
        self.maxx=0
        self.initialize()
        self.cnt=0
        return self.get_observation_space()
    def render(self):
        
        #colors = np.array([0x008B8B, 0xFF6347, 0xEEEEF0], dtype=np.uint32)
        color_jelly=np.array([0xff7fff,
                              0xff99ff,
                              0xffb2ff,
                              0xffd1ff,
                              0xfff4ff,0x84bf96,0x000000],dtype=np.uint32)
        self.gui.circles(self.x.to_numpy(), radius=1.5, color=color_jelly[self.color.to_numpy()])
        self.gui.circles(self.windmill_x.to_numpy(), radius=1.5,color=0x845538)
        self.gui.lines(self.begin.reshape([-1,2]),self.end.reshape([-1,2]), radius=1,color=0xd71345)
        self.gui.show()
        
        

'''
gui = ti.GUI("MPM SCOOP THE JELLY", res=512, background_color=0x112F41)

#model='./music_model/autoencoder16.pkl'
model=1
mpm=ScoopJellyEnvs({'n_tasks':200},200,0,model,gui)
velocity_256=np.zeros([1000,128,128,2])
flag=0
for epoch in range(200):
    print(epoch)
    itt=0
    mpm.reset_task(epoch)
    for frame in range(1000):    
        if frame%100==0:
            mpm.reset()
            flag=0
        x=mpm.jelly_pos[6]-0.176
        x_=mpm.jelly_pos[6]-0.226
        y=mpm.jelly_pos[7]-0.1
        x0=mpm.center[None][0]
        y0=mpm.center[None][1]
        dis=math.sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))
        dis_=math.sqrt((x_-x0)*(x_-x0)+(y-y0)*(y-y0))
        vx=(x-x0)/dis*2
        vy=(y-y0)/dis*2
        vx_=(x_-x0)/dis_*2
        vy_=(y-y0)/dis_*2
        #print(dis_)
        if dis_<0.02:
            flag==1
        if dis<0.02:
            flag=2
        if flag==0:
            action = np.array([vx_,vy_,0])
        elif flag==1:
            action = np.array([vx,vy,0])
        else:
            action=np.array([vx,2.0,0])
        mpm.step(action)
        velocity_256[itt]=mpm.get_state()
        itt=itt+1
        mpm.render()
    #np.save("./scoop_data/%d_velocity.npy"%epoch,velocity_256)
'''