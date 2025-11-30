import taichi as ti
import numpy as np
import math
import random

block_num= 128

@ti.data_oriented
class MPMSolver:
    def __init__(self, quality, E_, mass, rho, G, P,n_ball,n_pipe):
        self.reset_mpm(quality, E_, mass, rho, G, P,n_ball,n_pipe)

    def reset_mpm(self, quality, E_, mass, rho, G, P,n_ball,n_pipe):
        ball_p_num=700
        fluid_p_num=15000+5000

        self.n_balls = n_ball
        self.n_pipes = n_pipe
        
        self.P = P
        self.rho = rho
        self.G = G
        self.E_ = E_
        self.quality = quality  # Use a larger value for higher-res simulations
        self.n_particles, self.x_grid, self.y_grid, self.z_grid = int((self.n_balls*ball_p_num+fluid_p_num*self.n_pipes) * self.quality ** 3), int(
            80 * self.quality), int(48 * self.quality), int(32 * self.quality)
        self.dx, self.inv_dx = 1.6 / self.x_grid, float(self.x_grid / 1.6)
        self.sample_x, self.sample_y, self.sample_z = 80, 48, 32
        self.sample_dx, self.sample_inv_dx = 1.6 / self.sample_x, float(self.sample_x / 1.6)
        self.dt = 2.0e-4 / self.quality
        self.p_vol = (self.dx * 0.5) ** 3
        self.p_rho = ti.field(dtype=ti.f32, shape=self.n_particles)  # material id
        self.p_mass = ti.field(dtype=ti.f32, shape=self.n_particles)  # material id
        self.tmp_mass = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.id = ti.field(dtype=ti.i32, shape=self.n_particles)
        self.active = ti.field(dtype=ti.i32, shape=self.n_particles)
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)  # position

        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)  # velocity
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_particles)  # affine velocity field
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_particles)  # deformation gradient
        self.material = ti.field(dtype=ti.i32, shape=self.n_particles)  # material id
        self.Jp = ti.field(dtype=ti.f32, shape=self.n_particles)  # plastic deformation
        self.sample_m = ti.field(dtype=ti.f32, shape=(self.sample_x, self.sample_y, self.sample_z))
        self.grid_v = ti.Vector.field(3, dtype=ti.f32,
                                shape=(self.x_grid, self.y_grid, self.z_grid))  # grid node momentum/velocity
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.x_grid, self.y_grid, self.z_grid))  # grid node mass
        
        self.jelly_vel = ti.field(dtype=ti.f32, shape=self.n_balls*3)
        self.jelly_pos = ti.field(dtype=ti.f32, shape=self.n_balls*3)

        # initial jelly mass for each task
        self.jelly_mass = ti.field(dtype=ti.f32, shape=self.n_balls)
        self.out = ti.field(dtype=ti.i32, shape=self.n_balls)
        self.hit = ti.field(dtype=ti.i32, shape=())
        self.num_now = ti.field(dtype=ti.i32, shape=())
        self.goal_f = ti.field(dtype=ti.f32, shape=())

        self.sample_v = ti.Vector.field(3, dtype=ti.f64, shape=(self.sample_x, self.sample_y, self.sample_z))


        # color
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)

        # scoop pipe
        self.bottom = 0.9  # axis-y pad position
        self.pad_length = 0.18
        self.num_pipe = 720 * self.quality ** 3
        pi = math.pi
        self.length = 0.008
        self.width = 0.12
        self.center = ti.Vector.field(3, dtype=ti.f32, shape=self.n_pipes)
        self.outflux = ti.field(dtype=ti.f32, shape=self.n_pipes)
        self.angle = ti.field(dtype=ti.f32, shape=self.n_pipes)
        self.omega = ti.field(dtype=ti.f32, shape=self.n_pipes)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=self.n_pipes)
        I = 1e-2
        self.grid_assit = ti.Vector.field(3, dtype=ti.f32,
                                    shape=(self.x_grid, self.y_grid, self.z_grid))  # grid node momentum/velocity

        self.fluid_size = int(fluid_p_num*self.n_pipes * self.quality ** 3)
        self.jelly_size = (self.n_particles - self.fluid_size) // self.n_balls
        # print(self.jelly_size)
        self.tmp_x = ti.Vector.field(2, dtype=ti.f32, shape=self.jelly_size)
        self.tmp_j = ti.Vector.field(3, dtype=ti.f32, shape=self.n_balls)
        self.rest_time1 = 3.0
        self.goal_list = np.random.randint(0, 7, 10000)
        self.task_id = 0
        self.goal_idx = 0
        self.goal_jelly = 0
        # Standard Sphere 
        self.stand_ball = ti.Vector.field(3, dtype=ti.f32, shape=self.jelly_size)

        # num_pts = self.jelly_size
        # self.bx = [0.5, 1.2]
        self.bx = [0.6, 1.1]
        self.by = [0.25, 0.85] # down to up
        self.bz = [0.12, 0.52]

        for j in range(1):
            if j==0:
                num_pts = self.jelly_size
                indices = np.arange(0, num_pts, dtype=float) + 0.5
                phi = np.arccos(1 - 2*indices/num_pts)
                theta = np.pi * (1 + 5**0.5) * indices
                x =  np.cos(theta) * np.sin(phi) 
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(phi)
                for i in range(num_pts):
                    x[i] =  x[i]*0.0125*(4-j)
                    y[i] = y[i]*0.0125*(4-j)
                    z[i] = z[i]*0.0125*(4-j)
            else:
                num_pts = self.jelly_size//4

                indices = np.arange(0, num_pts, dtype=float) + 0.5
                phi = np.arccos(1 - 2*indices/num_pts)
                theta = np.pi * (1 + 5**0.5) * indices
                xx =  np.cos(theta) * np.sin(phi) 
                yy = np.sin(theta) * np.sin(phi)
                zz = np.cos(phi)
                for i in range(num_pts):
                    xx[i] =  xx[i]*0.0125*(4-j)
                    yy[i] = yy[i]*0.0125*(4-j)
                    zz[i] = zz[i]*0.0125*(4-j)
                x = np.append(x,xx,axis=0)
                y = np.append(y,yy,axis=0)
                z = np.append(z,zz,axis=0)


        t = np.concatenate(([x], [y],[z] ),axis=0,dtype=np.float32)
        self.stand_ball.from_numpy(t.transpose())

        # matching shape
        self.radius_vector = ti.Vector.field(3, float, self.jelly_size*self.n_balls)
        self.paused = ti.field(ti.i32, shape=())
        self.q_inv = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.n_balls))

        self.flag = 0

        for i in range(self.n_balls):
            self.jelly_mass[i] = mass[i]
        # print(type(self.G))
        print("mass: ",  mass)
        print("rho: ",self.rho)
        print("Timestep size: ", self.dt)
        print("Gravity: ",self.G)
        print("Youg's Module: ",self.E_)
        print("Particles: ",self.n_particles)
        print("Grid resolution: ",self.x_grid,self.y_grid,self.z_grid)
        self.initialize()
        
    @ti.kernel
    def solve_pipe(self, idx: ti.i32, a: ti.f32, b: ti.f32, c: ti.f32):  # cation[ax,aw,ad]
        self.vel[idx][0] += self.dt * a
        self.vel[idx][1] += self.dt * b
        self.omega[idx] += self.dt * c
        self.vel[idx][0] = ti.min(self.vel[idx][0], 20.0)
        self.vel[idx][0] = ti.max(self.vel[idx][0], -20.0)
        self.vel[idx][1] = ti.min(self.vel[idx][1], 5.0)
        self.vel[idx][1] = ti.max(self.vel[idx][1], -5.0)
        self.omega[idx] = ti.min(self.omega[idx], 25.0)
        self.omega[idx] = ti.max(self.omega[idx], -25.0)
        self.angle[idx] += self.dt * self.omega[idx]
        self.center[idx][0] += self.dt * self.vel[idx][0]
        self.center[idx][2] += self.dt * self.vel[idx][1]

        # if self.center[idx][0] < 0.1+1.4/self.n_pipes*idx:
        #     self.center[idx][0] = 0.1+1.4/self.n_pipes*idx
        #     self.vel[idx][0] = 0
        # elif self.center[idx][0] > 0.1+1.4/self.n_pipes*(idx+1):
        #     self.center[idx][0] = 0.1+1.4/self.n_pipes*(idx+1)
        #     self.vel[idx][0] = 0
        if self.center[idx][0] > self.bx[1]+0.05:
            self.center[idx][0] = self.bx[1]+0.05
            self.vel[idx][0] = 0
        elif self.center[idx][0] <self.bx[0]-0.05:
            self.center[idx][0] = self.bx[0]-0.05
            self.vel[idx][0] = 0

        if self.angle[idx] < 1.0:
            self.angle[idx] = 1.0
            self.omega[idx] = 0
        elif self.angle[idx] > 2.2:
            self.angle[idx] = 2.2
            self.omega[idx] = 0
        # self.angle[idx] = 1.5
        if self.center[idx][2] < 0.14:
            self.center[idx][2] = 0.14
            self.vel[idx][1] = 0
        elif self.center[idx][2] > 0.50:
            self.center[idx][2] = 0.50
            self.vel[idx][1] = 0

    @ti.kernel
    def add_cube(self, idx: ti.i32, alpha: ti.f32):
        corner = ti.Vector([self.center[idx][0], self.center[idx][1], self.center[idx][2]])
        num = int(self.num_pipe * alpha * alpha * self.dt / 4e-3)
        ct = self.num_now[None]
        cnt = 0

        while (cnt < num):
            if self.active[ct] == 1:
                ct = (ct + 1) % self.fluid_size
            else:
                i = ct
                r = self.random_point_in_unit_sphere()
                a = ti.random()
                width = self.width * alpha
                self.x[i] = corner + ti.Vector(
                    [a * self.length * self.dt / 4e-3, r[0] * 0.5 * width, r[1] * 0.5 * width])
                self.p_rho[i] = self.rho
                self.p_mass[i] = self.p_rho[i] * self.p_vol
                self.material[i] = 0  # 0: fluid 1: jelly
                # self.color[i] = ti.hex_to_rgb(0xabc88b)
                self.active[i] = 1
                self.v[i] = 20 * ti.Vector([ti.cos(self.angle[idx]), ti.sin(self.angle[idx]), 0])
                self.F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                self.C[i] = ti.Matrix.zero(ti.f32, 3, 3)
                self.Jp[i] = 1
                relative_pos = self.x[i] - self.center[idx]
                rot1 = ti.Vector([ti.cos(self.angle[idx]), -ti.sin(self.angle[idx]), 0])
                rot2 = ti.Vector([ti.sin(self.angle[idx]), ti.cos(self.angle[idx]), 0])
                new_relativepos = ti.Vector([rot1.dot(relative_pos), rot2.dot(relative_pos), relative_pos[2]])
                self.x[i] = new_relativepos + self.center[idx]
                cnt += 1
        self.num_now[None] = ct
        
    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3,) * 3))

    @ti.kernel
    def substep(self):
        E, nu = self.E_, 0.2  # Young's modulus and Poisson's ratio
        mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
        for i, j, k in self.grid_m:
            if i < self.sample_x and j < self.sample_y and k < self.sample_z:
                self.sample_v[i, j, k] = [0, 0, 0]
                self.sample_m[i, j, k] = 0
            self.grid_v[i, j, k] = [0, 0, 0]
            self.grid_m[i, j, k] = 0
            self.grid_assit[i, j, k] = [0, 0, 0]

        # print(2)
        ti.loop_config(block_dim=block_num)

        for p in self.x:  # Particle state update and scatter to grid (P2G)
            if self.active[p] == 0:
                continue
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            base_sample = (self.x[p] * self.sample_inv_dx - 0.5).cast(int)
            fx_sample = self.x[p] * self.sample_inv_dx - base_sample.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            sample_w = [0.5 * (1.5 - fx_sample) ** 2, 0.75 - (fx_sample - 1) ** 2, 0.5 * (fx_sample - 0.5) ** 2]

            self.F[p] = (ti.Matrix.identity(ti.f32, 3) + self.dt * self.C[p]) @ self.F[p]  # deformation gradient update
            h = ti.exp(10 * (1.0 - self.Jp[p]))  # Hardening coefficient: snow gets harder when compressed
            if self.material[p] == 1:  # jelly, make it softer
                h = 5.0
            mu, la = mu_0 * h, lambda_0 * h
            if self.material[p] == 0:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(self.F[p])
            J = 1.0

            for d in ti.static(range(3)):
                new_sig = sig[d, d]
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig

            if self.material[p] == 0:  # Reset deformation gradient to avoid numerical instability
                new_F = ti.Matrix.identity(ti.f32, 3)
                new_F[0, 0] = J
                self.F[p] = new_F

            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose() + ti.Matrix.identity(ti.f32, 3) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.p_mass[p] * self.C[p]

            if self.material[p] == 0:  # and self.P:
                self.Jp[p] = (1 + self.dt * self.C[p].trace()) * self.Jp[p]
                st = -self.dt * 4 * E * self.p_vol * (self.Jp[p] - 1) / self.dx ** 2
                affine = ti.Matrix([[st, 0, 0], [0, st, 0], [0, 0, st]]) + self.p_mass[p] * self.C[p]

            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(float) - fx) * self.dx
                dpos_sample = (offset.cast(float) - fx_sample) * self.sample_dx
                weight = 1.0
                weight_sample = 1.0
                for d in ti.static(range(3)):
                    weight *= w[offset[d]][d]
                    weight_sample = sample_w[offset[d]][d]
                # print((base_sample + offset)[0])
                if (base_sample + offset)[0] < self.sample_x and (base_sample + offset)[1]  < self.sample_y and (base_sample + offset)[2]  < self.sample_z:
                    self.sample_v[base_sample + offset] += weight_sample * (
                            self.p_mass[p] * self.v[p] + affine @ dpos_sample)
                    self.sample_m[base_sample + offset] += weight_sample * self.p_mass[p]
                if (base + offset)[0] < self.x_grid and (base + offset)[1]  < self.y_grid and (base + offset)[2]  < self.z_grid:
                    self.grid_assit[base + offset] += weight * (affine @ dpos)
                    self.grid_v[base + offset] += weight * (self.p_mass[p] * self.v[p] + affine @ dpos)
                    self.grid_m[base + offset] += weight * self.p_mass[p]

        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 0:  # No need for epsilon here
                self.grid_v[i, j, k] = (1 / self.grid_m[i, j, k]) * self.grid_v[i, j, k]  # Momentum to velocity
                self.grid_v[i, j, k][1] -= self.dt * self.G  # gravity
            self.grid_v[i, j, k][2] = ti.min(self.grid_v[i, j, k][2], 10)
            self.grid_v[i, j, k][1] = ti.min(self.grid_v[i, j, k][1], 20)
            self.grid_v[i, j, k][0] = ti.min(self.grid_v[i, j, k][0], 20)
            self.grid_v[i, j, k][2] = ti.max(self.grid_v[i, j, k][2], -10)
            self.grid_v[i, j, k][1] = ti.max(self.grid_v[i, j, k][1], -20)
            self.grid_v[i, j, k][0] = ti.max(self.grid_v[i, j, k][0], -20)

            if i < self.sample_x and j < self.sample_y and k < self.sample_z:
                if self.sample_m[i, j, k] > 1e-8:  # No need for epsilon here
                    self.sample_v[i, j, k] = (1 / self.sample_m[i, j, k]) * self.sample_v[
                        i, j, k]  # Momentum to velocity
                    self.sample_v[i, j, k][1] -= self.dt * self.G
                self.sample_v[i, j, k][2] = ti.min(self.sample_v[i, j, k][2], 10)
                self.sample_v[i, j, k][1] = ti.min(self.sample_v[i, j, k][1], 20)
                self.sample_v[i, j, k][0] = ti.min(self.sample_v[i, j, k][0], 20)
                self.sample_v[i, j, k][2] = ti.max(self.sample_v[i, j, k][2], -10)
                self.sample_v[i, j, k][1] = ti.max(self.sample_v[i, j, k][1], -20)
                self.sample_v[i, j, k][0] = ti.max(self.sample_v[i, j, k][0], -20)

            if j < self.quality and self.grid_v[i, j, k][1] < 0:
                n = ti.Vector([0, 1, 0])
                nv = self.grid_v[i, j, k].dot(n)
                self.grid_v[i, j, k] = self.grid_v[i, j, k] - n * nv
            if j > int(self.y_grid / 0.96 * 0.95) and self.grid_v[i, j, k][1] > 0:
                n = ti.Vector([0, -1, 0])
                nv = self.grid_v[i, j, k].dot(n)
                self.grid_v[i, j, k] = self.grid_v[i, j, k] - n * nv
            if k < int(self.quality * 0.25) and self.grid_v[i, j, k][2] < 0:
                n = ti.Vector([0, 0, 1])
                nv = self.grid_v[i, j, k].dot(n)
                self.grid_v[i, j, k] = self.grid_v[i, j, k] - n * nv
            if k > int(self.z_grid * 0.75) and self.grid_v[i, j, k][2] > 0:
                n = ti.Vector([0, 0, -1])
                nv = self.grid_v[i, j, k].dot(n)
                self.grid_v[i, j, k] = self.grid_v[i, j, k] - n * nv

        ti.loop_config(block_dim=block_num)
        for p in self.x:  # grid to particle (G2P)
            if self.active[p] == 0:
                continue
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(ti.f32, 3)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)
            for offset in ti.static(ti.grouped(self.stencil_range())):
                if (base + offset)[0] < self.x_grid and (base + offset)[1]  < self.y_grid and (base + offset)[2]  < self.z_grid:
                    dpos = offset.cast(float) - fx
                    g_v = self.grid_v[base + offset]
                    weight = 1.0
                    for d in ti.static(range(3)):
                        weight *= w[offset[d]][d]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += self.dt * self.v[p]  # advection

            if self.material[p] == 0:
                if self.x[p][0] > 1.56 or self.x[p][0] < 0.04 or self.x[p][1] > 0.92 or self.x[p][1] < 0.04 or \
                        self.x[p][
                            2] < 0.17 or self.x[p][2] > 0.47:
                    self.active[p] = 0
                    self.x[p] = ti.Vector([0, 0, 0])
        # matching shape
        self.shape_matching()
        self.cal_jelly_state()
    @ti.func
    def compute_radius_vector(self):
        #compute the mass center and radius vector
        for jj in range(self.n_balls):
            center_mass = ti.Vector([0.0, 0.0, 0.0])
            for i in range(self.fluid_size+jj*self.jelly_size,self.fluid_size+(jj+1)*self.jelly_size):
                center_mass += self.x[i]
            center_mass /= self.jelly_size
            for i in range(self.fluid_size+jj*self.jelly_size,self.fluid_size+(jj+1)*self.jelly_size):
                self.radius_vector[i-self.fluid_size] = self.x[i] - center_mass


    @ti.func
    def precompute_q_inv(self):
        for jj in range(self.n_balls):
            res = ti.Matrix([[0.0] * 3 for _ in range(3)], ti.f32)
            # idx = (i - self.fluid_size) // self.jelly_size
            for i in range(self.fluid_size+jj*self.jelly_size,self.fluid_size+(jj+1)*self.jelly_size):
                res += self.radius_vector[(i - self.fluid_size)].outer_product(self.radius_vector[(i - self.fluid_size)])
            
            self.q_inv[jj] = res.inverse()

    @ti.func
    def shape_matching(self): 
        for jj in range(self.n_balls):
            #compute the new(matched shape) mass center
            c = ti.Vector([0.0, 0.0, 0.0])
            for i in range(self.fluid_size+jj*self.jelly_size,self.fluid_size+(jj+1)*self.jelly_size):
                c += self.x[i]
            c /= self.jelly_size

            #compute transformation matrix and extract rotation
            A = sum1 = ti.Matrix([[0.0] * 3 for _ in range(3)], ti.f32)
            for i in range(self.fluid_size+jj*self.jelly_size,self.fluid_size+(jj+1)*self.jelly_size):
                sum1 += (self.x[i] - c).outer_product(self.radius_vector[i-self.fluid_size])
            A = sum1 @ self.q_inv[jj]

            R, _ = ti.polar_decompose(A)

            #update velocities and positions
            for i in range(self.fluid_size+jj*self.jelly_size,self.fluid_size+(jj+1)*self.jelly_size):
                self.x[i] = c + R @ self.radius_vector[i-self.fluid_size]
    
    @ti.func
    def random_point_in_unit_sphere(self):
        ret = ti.Vector.zero(dt=ti.f32, n=2)
        while True:
            for i in ti.static(range(2)):
                ret[i] = ti.random(ti.f32) * 2 - 1
            if ret.norm_sqr() <= 1:
                break
        return ret

    @ti.func
    def random_point_in_unit_sphere3(self):
        ret = ti.Vector.zero(dt=ti.f32, n=3)
        while True:
            for i in ti.static(range(3)):
                ret[i] = ti.random(ti.f32) * 2 - 1
            if ret.norm_sqr() <= 1:
                break
        return ret

    @ti.kernel
    def initialize(self):
        for i in range(self.n_balls):
            # self.tmp_j[i] = ti.Vector([0.07 + 1.4 /(self.n_balls+1)*(i+1)+0.02, 0.55, 0.32])
            self.tmp_j[i] = ti.Vector([0.75+i*0.2, 0.75-i*0.02, 0.32])

        for i, j, k in self.grid_m:
            if i < self.sample_x and j < self.sample_y and k < self.sample_z:
                self.sample_v[i, j, k] = [0, 0, 0]
                self.sample_m[i, j, k] = 0
            self.grid_v[i, j, k] = [0, 0, 0]
            self.grid_m[i, j, k] = 0
            self.grid_assit[i, j, k] = [0, 0, 0]

        for i in range(self.n_particles):
            if i < self.fluid_size:
                self.x[i] = [0.0, 0.0, 0.0]
                self.p_rho[i] = self.rho
                self.p_mass[i] = self.p_rho[i] * self.p_vol
                self.material[i] = 0  # 0: fluid 1: jelly
                self.color[i] = ti.hex_to_rgb(0x3399ff)
                self.active[i] = 0
            else:
                idx = int((i - self.fluid_size) // self.jelly_size)
                if idx >= self.n_balls-1:
                    idx = self.n_balls-1
                self.id[i] = idx
                # if int(i - self.fluid_size) %self.jelly_size==0:
                #     self.x[i] = self.tmp_j[idx]
                #     # print(self.x[i])
                    
                # else:
                self.x[i] = self.tmp_j[idx] + self.stand_ball[(i - self.fluid_size)%self.jelly_size]
                self.material[i] = 1
                self.active[i] = 1
                self.p_rho[i] = self.jelly_mass[idx]
                self.p_mass[i] = self.p_rho[i] * self.p_vol
                self.color[i] = ti.hex_to_rgb(0xee1d24)
            self.v[i] = ti.Matrix([0.0, 0.0, 0.0])
            self.F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.Jp[i] = 1

        # initialize pipe
        for i in range(self.n_pipes):
            self.center[i] = ti.Vector([0.85, 0.06, 0.32])
            self.angle[i] = 1.5
            self.omega[i] = 0.
            self.vel[i] = ti.Vector([0, 0])

        self.num_now[None] = 0

        self.cal_jelly_state()
        self.compute_radius_vector() #store the shape of rigid body
        self.precompute_q_inv()

    def get_state(self):
        '''
        get V state
        '''
        state = self.sample_v.to_numpy()
        return state

    def get_rigid_state(self):
        '''
        get pipes' state
        '''
        for i in range(self.n_pipes):
            if i==0:
                II = np.array(
                    [self.center[i][0], self.center[i][1], self.center[i][2], self.angle[i] ,
                    self.vel[i][0]  * 0.1, self.vel[i][1] * 0.1,
                    self.omega[i] * 0.1, self.outflux[i]])
            else:
                II = np.append(II,np.array(
                    [self.center[i][0], self.center[i][1], self.center[i][2], self.angle[i] ,
                    self.vel[i][0]  * 0.1, self.vel[i][1] * 0.1,
                    self.omega[i] * 0.1, self.outflux[i]]))
        return II

    @ti.func
    def cal_jelly_state(self):
        # for i in range(self.n_balls*3):
        for i in ti.grouped(self.jelly_vel):
            self.jelly_pos[i] = 0.0
            self.jelly_vel[i] = 0.0
        for i in range(self.fluid_size,self.n_particles):
            idx = (i - self.fluid_size) // self.jelly_size
            if idx >= self.n_balls:
                continue

            # if int(i - self.fluid_size) % self.jelly_size==0:
            self.jelly_pos[idx * 3] += self.x[i][0]/self.jelly_size
            self.jelly_pos[idx * 3 + 1] += self.x[i][1] /self.jelly_size
            self.jelly_pos[idx * 3 + 2] += self.x[i][2]/self.jelly_size
            self.jelly_vel[idx * 3] += self.v[i][0]/self.jelly_size
            self.jelly_vel[idx * 3 + 1] += self.v[i][1]/self.jelly_size
            self.jelly_vel[idx * 3 + 2] += self.v[i][2]/self.jelly_size
        


    def get_jelly_state(self):
        '''
        get balls' state
        '''
        poss = np.zeros(self.n_balls*3)
        vels = np.zeros(self.n_balls*3)
        ori = np.zeros(self.n_balls*3)
        for i in range(self.n_balls*3):
            poss[i] = self.jelly_pos[i]
            vels[i] = self.jelly_vel[i]
            ori[i] = self.tmp_j[i//3][i%3]
        a = np.append(poss, vels)
        a = np.append(a,ori)
        return a

    def get_reward(self):
        self.flag = 0
        reward = 0
        # tmp_j[i][0]
        for i in range(self.n_balls):
            # dist = 2 * (self.jelly_pos[i*3+0] - self.tmp_j[i][0]) **2 + (self.jelly_pos[i*3+1] - 0.45) **2 + (self.jelly_pos[i*3+2] - 0.32)**2
            dist = 2 * (self.jelly_pos[i*3+0] - self.tmp_j[i][0]) **2 + (self.jelly_pos[i*3+1] - 0.55) **2 + (self.jelly_pos[i*3+2] - 0.32)**2
           
            v = math.sqrt(
            self.jelly_vel[i*3+0] **2+ self.jelly_vel[i*3+1]**2 + self.jelly_vel[i*3+2]**2)
            reward +=2.5 * math.exp(-30 * dist)+ 1.5 * math.exp(-2 * v)
            self.update_jelly(i)
            if self.out[i] >= 5:
                # if self.out[i] ==7:
                #     print("left")
                # if self.out[i] ==9:
                #     print("right")
                reward -= 25
                self.flag = 2
        return reward * 0.1

    @ti.kernel
    def update_jelly(self, idx: ti.i32):
        # detection crush by out[idx]'s value
        '''boundry ellsipe detetion and musci key crush detection'''

        for i in range(self.fluid_size,self.n_particles):
            if idx != self.id[i]:
                continue
            # left
            if self.jelly_pos[3 * idx] < self.bx[0]:
                # self.out[idx] = 7
                self.v[i][0] = 0.5 * abs(self.v[i][0])
                # if i==self.fluid_size:
                #     print("touch left!")

            # right
            if self.jelly_pos[3 * idx] > self.bx[1]:
                # self.out[idx] = 9
                self.v[i][0] = -0.5 * abs(self.v[i][0])
                # if i==self.fluid_size:
                #     print("touch right!")

            # down
            if self.jelly_pos[3 * idx + 1] <= self.by[0]:
                self.out[idx] = 8
                self.v[i][1] = 0.5 * abs(self.v[i][1])
                self.v[i][0] = 0.95 * self.v[i][0] # 摩擦力
                # if i==self.fluid_size:
                #     print("touch down!")

            # up
            if self.jelly_pos[3 * idx + 1] > self.by[1]:
                # self.out[idx] = 8
                self.v[i][1] = -abs(self.v[i][1])
                # if i==self.fluid_size:
                #     print("touch up!")

            # forward
            if self.jelly_pos[3 * idx + 2] < self.bz[0]:
                # self.out[idx] = 8
                self.v[i][2] = 0.08 * abs(self.v[i][2])
                # if i==self.fluid_size:
                #     print("touch forward!")

            # back
            if self.jelly_pos[3 * idx + 2] > self.bz[1]:
                # self.out[idx] = 8
                self.v[i][2] = -0.08 * abs(self.v[i][2])
                # if i==self.fluid_size:
                #     print("touch back!")