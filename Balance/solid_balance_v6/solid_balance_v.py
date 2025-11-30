import time
import taichi as ti
import numpy as np
import math
import random
from solid_balance_v6.MPM_solver import MPMSolver

import torch
torch.set_num_threads(4)

# from vae_model import BetaVAE_B,reconstruction_loss,kl_divergence
from solid_balance_v6.AE_CNN import AutoEncoder
from torch.autograd import Variable
from torchvision import transforms


import gym
# ti.init(arch=ti.gpu,debug=True)  # Try to run on GPU
ti.init(arch=ti.cuda,ad_stack_size=512)  # Try to run on GPU

import time

seed = 1024
# torch.manual_seed(seed)            
# torch.cuda.manual_seed(seed)       
# torch.cuda.manual_seed_all(seed)   
random.seed(seed)    
np.random.seed(seed)  
# import tina

class MusicPipeEnvs(MPMSolver):
    def __init__(self, n_balls,n_pipes,show):

        self.show = show
        self.n_tasks = 1

        path = "./solid_balance_v6/autoencoder.pkl"
        # path = "./FV_models/vae1.pkl"
        self.model=torch.load(path,map_location=torch.device('cuda'))
        self.model.eval()

        self._max_episode_steps = 1000
        self.cnt = 0

        self.qualitys = np.random.randint(40, 41, self.n_tasks) * 0.025  # for meta train 40 60
        # self.qualitys = np.random.randint(60, 61, self.n_tasks) * 0.025  # for meta train 40 60

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
            # self.camera.up(0, 1, 0)
            # self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
            self.window = ti.ui.Window(name='Solid balance with Encoder', res = (640, 360), fps_limit=200, pos = (150, 150))
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
        # self.scene.input(self.gui)
        # self.pars.set_particles(self.x.to_numpy())
        # self.pars.set_particle_colors(self.color.to_numpy())
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.2, 0.2, 0.2))
        self.scene.point_light(pos=(1.5, 1.5, 1.5), color=(0.7, 0.7, 0.7))
        
        # print(self.tmp_color[0])
        self.scene.mesh(self.vertices,
            indices=self.indices,
            per_vertex_color=self.colours,
            two_sided=True)
        self.scene.particles(centers=self.x, per_vertex_color=self.color, radius = 0.001)
        # print(1)
        self.canvas.scene(self.scene)
        self.window.show()
        
        # return self.scene.img
    @property
    def observation_space(self):
        # low = np.zeros([512, 512, 3], dtype=np.uint8)
        # high = 255 * np.ones([512, 512, 3], dtype=np.uint8)
        # low = np.zeros([64, 64, 3], dtype=np.uint8)
        # high = 255 * np.ones([64, 64, 3], dtype=np.uint8)
        # spaces = {'image': gym.spaces.Box(low, high, dtype=np.uint8)}
        # spaces['state'] =  gym.spaces.Box(-np.inf, np.inf,(26,), dtype=np.float)
        # return gym.spaces.Dict(spaces)
        return gym.spaces.Box(-np.inf, np.inf,(self.n_pipes*8+self.n_balls*9+64,))
        
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

    def normalization(self, data):
        # mu = np.mean(data)
        # sigma = np.std(data)+1e-7
        mask = np.zeros(data.shape)
        mask[:, :, 2:78, 2:43, 2:30] = 1
        data[mask == 0] = 0
        mu = np.mean(data)
        sigma = np.std(data) + 1e-7
        '''
        mu=2.8998700831383357e-05 
        sigma=0.00015228158427660628
        '''
        return (data - mu) / sigma, mu, sigma

    def get_observation_space(self):
        state=self.get_state().reshape([-1,3,80,48,32])

        # state,mu,sigma=self.normalization(state)
        # sta=torch.from_numpy(state)
        # sta = Variable(sta.float().cuda())
        # encode=self.model(sta)
        # encode=encode.cpu().detach().numpy()
        # a=np.append(self.get_jelly_state(),self.get_rigid_state())
        # obs=np.append(a,0.5*encode)
        # tt = time.time()
        # norm = transforms.Normalize((0.5 ), (0.5)) 
        norm = transforms.Normalize((0.07 ), (1.3)) 
        sta=torch.from_numpy(state)
        sta = Variable(sta.float().cuda())
        with torch.no_grad():
            encode=self.model(norm(sta))
            # encode = self.model.get_f(norm(sta))
            # print(encode.size())
        encode=encode.cpu().detach().numpy()

        # print(time.time()-tt)
        a=np.append(self.get_jelly_state(),self.get_rigid_state())
        obs=np.append(a,0.5*encode)

        return  obs
    def save_ply(self,dir):
        if self.cnt ==1:
            self.pipe_state_list =[]
        np_pos_all = np.reshape(self.x.to_numpy(), (self.n_particles, 3))
        np_pos_fluid = np_pos_all[:self.fluid_size]
        
        series_prefix_1 = "./"+dir+"/fluid.ply"
        writer = ti.tools.PLYWriter(num_vertices=self.fluid_size)
        writer.add_vertex_pos(np_pos_fluid[:, 0], np_pos_fluid[:, 1], np_pos_fluid[:, 2])
        writer.export_frame_ascii(self.cnt, series_prefix_1)
        
        series_prefix_2 = "./"+dir+"/ball_1.ply"
        np_pos_ball1 = np_pos_all[self.fluid_size:self.fluid_size+self.jelly_size]
        writer1 = ti.tools.PLYWriter(num_vertices=self.jelly_size)
        writer1.add_vertex_pos(np_pos_ball1[:, 0], np_pos_ball1[:, 1], np_pos_ball1[:, 2])
        writer1.export_frame_ascii(self.cnt, series_prefix_2)
        
        series_prefix_3 = "./"+dir+"/ball_2.ply"
        np_pos_ball2 = np_pos_all[self.fluid_size+self.jelly_size:self.fluid_size+2*self.jelly_size]
        writer2 = ti.tools.PLYWriter(num_vertices=self.jelly_size)
        writer2.add_vertex_pos(np_pos_ball2[:, 0], np_pos_ball2[:, 1], np_pos_ball2[:, 2])
        writer2.export_frame_ascii(self.cnt, series_prefix_3)

        i = 0
        pipe_state = np.array(
                    [self.center[i][0], self.center[i][1], self.center[i][2], self.angle[i]])
        self.pipe_state_list.append(pipe_state)
        # print(pipe_state)
        
        np.savetxt("./"+dir+"/output.csv", self.pipe_state_list, delimiter=",", fmt="%64f")


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