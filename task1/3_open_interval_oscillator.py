#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:17:20 2017

@author: yk
"""
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math

def handler(func, *args):
    return func(*args)

#%%
class Linear:
    def __init__(self, N):
        self.N = N
    def gradient(self, x):
        d = np.zeros(self.N)
        d[0] = x[1]
        d[1] = -x[0]
        return d
    
    def gradient_adjoint(self, la, x):
        mt = np.zeros((self.N,self.N))
        mt[0][1] = -1
        mt[1][0] = 1
        gr = mt @ la
        return gr        
    
class RK4:
    def __init__(self, N, dt):
        self.N = N
        self.dt = dt

    def nextstep(self, gradient, x):
        k1 = handler(gradient, x)
        k2 = handler(gradient, x + k1*self.dt/2)
        k3 = handler(gradient, x + k2*self.dt/2)
        k4 = handler(gradient, x + k3*self.dt)
        return x + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
    
    def orbit(self, gradient, t0, x0, T):
        t = np.arange(0., T, dt)
        steps = int(T/self.dt)
        o = np.zeros((steps,self.N))
        o[0] = np.copy(x0)
        for i in range(1,steps):
            o[i] = self.nextstep(gradient, o[i-1])
        return o

    def observed(self, gradient, t0, x0, T, stddev, rng):
        steps = int(T/self.dt)
        o = self.orbit(gradient, t0, x0, T)
        for i in range(steps):
            for j in range(self.N):                
                o[i][j] += stddev * rng.randn()
        return o

    def true_observed(self, gradient, t0, x0, T, stddev, rng):
        steps = int(T/self.dt)
        o = self.orbit(gradient, t0, x0, T)
        obs = np.copy(o)
        for i in range(steps):
            for j in range(self.N):
                obs[i][j] += stddev * rng.randn()
        return o, obs

def plot_orbit(dat):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(dat[:,0],dat[:,1],dat[:,2])
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    plt.show()
    
    
#%%
N = 2
F = 8
stddev = 0.1
year = 2
day = 365 * year
dt = 0.01
T = day * 0.2
steps = int(T/dt)
interval = 5

oscillator = Linear(N)
rk4 = RK4(N, dt)

t = np.arange(0., T, dt)
t_day = np.arange(0.,T/0.2, dt/0.2)

seeds = np.array([1,2])

for seed in seeds:
    rng = np.random.RandomState(seed)
    
    x0 = rng.uniform(-10, 10, 2)
    
    true_orbit, observed = rk4.true_observed(oscillator.gradient, 0., x0, T, stddev, rng)
    
    print ("observation RMSE: ", np.mean([np.linalg.norm(observed[i] - true_orbit[i])/math.sqrt(N) for i in range(steps)]))
    
    assimilation_xzero = observed[rng.randint(len(observed))]
    with open('oscillator_data/assimilation_xzero.' + str(seed) + '.dat', 'w') as ff:
        ff.write(("%f\t"*N + "\n") % tuple(assimilation_xzero))
        ff.close()
        
    with open('oscillator_data/year.' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps):
            f.write(("%f\t"*N + "\n") % tuple(true_orbit[i,:]))
        f.close()

    sparse_orbit = []
    with open('oscillator_data/year.' + str(interval) + '.' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps, interval):
            f.write(("%f\t"*N + "\n") % tuple(true_orbit[i,:]))
            sparse_orbit.append(true_orbit[i,:])
        f.close()
    
    with open('oscillator_data/observed.' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps):
            f.write(("%f\t"*N + "\n") % tuple(observed[i,:]))
        f.close()

    sparse_obs = []
    with open('oscillator_data/observed.' + str(interval) + '.' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps, interval):
            f.write(("%f\t"*N + "\n") % tuple(observed[i,:]))
            sparse_obs.append(observed[i,:])
        f.close()
        
    np.savetxt("oscillator_data/cov." + str(interval) + "." + str(seed) + ".dat", np.cov(np.transpose(np.asarray(sparse_obs))))