from solid_balance_v6.solid_balance_v import MusicPipeEnvs
from solid_balance_v6.AE_CNN import AutoEncoder
import taichi as ti

n_ball = 2
n_pipe = 1

env = MusicPipeEnvs(n_ball,n_pipe,False)
env._max_episode_steps=1000