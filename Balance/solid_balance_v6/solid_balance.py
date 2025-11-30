import time
import taichi as ti
import numpy as np
import math
import random
from solid_balance_v6.MPM_solver import MPMSolver

# import torch
# from music_envs.AE_CNN import AutoEncoder
# from torch.autograd import Variable
import gym
import torch
torch.set_num_threads(4)
# ti.init(arch=ti.gpu,debug=True)  # Try to run on GPU
ti.init(arch=ti.gpu,ad_stack_size=512)  # Try to run on GPU

seed = 1024
# torch.manual_seed(seed)            
# torch.cuda.manual_seed(seed)       
# torch.cuda.manual_seed_all(seed)   
random.seed(seed)    
np.random.seed(seed)  
# import tina


# CUDA_VISIBLE_DEVICES=['0','1']
class MusicPipeEnvs(MPMSolver):
    def __init__(self, n_balls,n_pipes,show):

        self.show = show
        self.n_tasks = 1

        self._max_episode_steps = 1000
        self.cnt = 0

        self.qualitys = np.random.randint(41, 42, self.n_tasks) * 0.025  # for meta train 40 60
        self.Es = np.random.randint(800, 801, self.n_tasks)#600 900
        self.quality = float(self.qualitys[0])
        self.E = self.Es[0]

        srand = np.random.randint(1, 2, size=(self.n_tasks,n_balls))#9 12
        self.projections = np.random.randint(1, 2, size=self.n_tasks)
        self.projection = self.projections[0]
        self.jelly_phos = srand * 4.0
        self.jelly_pho = self.jelly_phos[0]

        srand = np.random.randint(100, 101, self.n_tasks)
        self.fluid_phos = srand * 0.01
        self.fluid_pho = self.fluid_phos[0]

        srand = np.random.randint(60, 61, self.n_tasks)
        self.gravitys = srand
        self.gravity = self.gravitys[0]

        
        super().__init__(self.quality, self.E, self.jelly_pho, self.fluid_pho, self.gravity, self.projection,n_balls,n_pipes)
        
        # GUI init
        if self.show:
            self.scene = ti.ui.Scene()

            self.camera = ti.ui.Camera()
            self.camera.position(1.,1.3,2)  # x, y, z
            self.camera.lookat(0.8, 0., 0)

            self.window = ti.ui.Window(name='Solid Balance', res = (640, 360), fps_limit=200, pos = (150, 150))
            self.canvas = self.window.get_canvas()
            self.canvas.set_background_color((1, 1, 1))

            self.indices = ti.field(ti.i32, 2*3)
            self.vertices = ti.Vector.field(3, ti.f32, 2*3)
            self.colours = ti.Vector.field(3, ti.f32, 2*3)

            self.indices.from_numpy(np.array([0,1,2,3,4,5]))
            self.vertices.from_numpy(np.array([
                [0,0,0],
                [0,0,1],
                [1.6,0,0],
                [1.6,0,1],
                [1.6,0,0],
                [0,0,1]
            ]).astype(np.float32))
            self.colours.from_numpy((1/256)*np.array([
                [201, 201, 201]
            ]*6).astype(np.float32))


    def render(self,close =True):
        # self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.2, 0.2, 0.2))
        self.scene.point_light(pos=(1.5, 1.5, 1.5), color=(0.7, 0.7, 0.7))
        
        # print(self.tmp_color[0])
        self.scene.mesh(self.vertices,
            indices=self.indices,
            per_vertex_color=self.colours,
            two_sided=True)
        self.scene.particles(centers=self.x, per_vertex_color=self.color, radius = 0.005)
        # print(1)
        self.canvas.scene(self.scene)
        self.window.show()
        
        # return self.scene.img
    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf,(self.n_pipes*8+self.n_balls*9,))
        
    @property
    def action_space(self):
        return gym.spaces.Box(-np.ones(self.n_pipes*4),np.ones(self.n_pipes*4))

    def get_all_task_idx(self):
        return range(self.n_tasks)

    def reset(self):
        self.reset_env()
        return self.get_observation_space()

    def set_seed(self,seed):
        np.random.seed(seed)
        self.projections = np.random.randint(1, 2, size=self.n_tasks)

    def get_observation_space(self):
        a=np.append(self.get_jelly_state(),self.get_rigid_state())
        return  a

    def step(self, action):
        
        for i in range(self.n_balls):
            self.out[i] = 0
        old = np.zeros(self.n_pipes)*1.0
        for i in range(self.n_pipes):
            old[i] = self.outflux[i]
            self.outflux[i] =  (action[4*i+3]+1)*0.5
            self.outflux[i] = max(0.0, self.outflux[i])
            self.outflux[i] = min(1.0, self.outflux[i])

        for s in range(int(4e-3 // self.dt)):
            for i in range(self.n_pipes):
                self.solve_pipe(i, action[4*i] * 1600, action[4*i+1] * 400, action[4*i+2] * 2000)

            if s * self.dt < 1e-3:
                for i in range(self.n_pipes):
                    self.add_cube(i, old[i] - (s * self.dt / 1e-3) * (old[i] - self.outflux[i]))
            else:
                for i in range(self.n_pipes):
                    self.add_cube(i, self.outflux[i])
            # print(self.grid_m.shape)
            
            # print(self.sample_m.shape)
            self.substep()
            # print(5)
        
        self.rest_time1 = self.rest_time1 - 0.1
        self.rest_time1 = max(self.rest_time1, -2.0)
        
        obs = self.get_observation_space()
        reward = self.get_reward()/50  # + at * 0.15
        # reward = 0 
        done = 0
        self.cnt += 1
        if self.cnt==self._max_episode_steps:
        #    print("MAX timesteps!")
           self.flag=2
        if self.flag == 1 or self.flag == 2 or self.flag == 3:
            done = 1
            self.cnt = 0
        if self.show:
            self.render()
        # torch.set_num_threads(4)
        return obs, reward, done, dict(reward=reward)

    def reset_env(self):

        self.initialize()
        # return self.get_observation_space()
    def cost_np_vec(self, obs, acts, next_obs):
        reward = np.zeros(len(obs))
        # tmp_j[i][0]
        for i in range(self.n_balls):
            dist = 2 * (obs[:,i*3] - obs[:,self.n_balls*6+i*3]) **2 + (obs[:,i*3+1] - 0.55) **2 + (obs[:,i*3+2] -0.32)**2
            v = np.sqrt(
            obs[:,self.n_balls*3+i*3] **2+ obs[:,self.n_balls*3+i*3+1]**2 + obs[:,self.n_balls*3+i*3+2]**2)
            reward +=2.5 * np.exp(-30 * dist)+ 1.5 * np.exp(-2 * v)

        for i in range(len(reward)):
            for j in range(self.n_balls):
                if obs[i,j*3]<=0.08+0.05 or obs[i,j*3]>=1.56-0.05:
                    reward[i] -= 25
                if obs[i,j*3+1]<=0.25+0.05 or obs[i,j*3+1]>=0.85-0.05:
                    reward[i] -= 25
                if obs[i,j*3+2]<=0.12+0.05 or obs[i,j*3+2]>=0.52-0.05:
                    reward[i] -= 25

        reward =  reward * 0.1
        return -reward

