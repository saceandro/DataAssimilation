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
class Lorenz96:
    def __init__(self, N, F):
        self.N = N
        self.F = F
        
    def gradient(self,t,x):
        d = np.zeros(self.N)
        d[0] = (x[1] - x[self.N-2]) * x[self.N-1] - x[0]
        d[1] = (x[2] - x[self.N-1]) * x[0]- x[1]
        d[self.N-1] = (x[0] - x[self.N-3]) * x[self.N-2] - x[self.N-1]
        for i in range(2, self.N-1):
            d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
        return d + self.F    

    def jacobian(self, x):
        m = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if (((i-1) % self.N) == (j % self.N)):
                    m[i][j] += x[(i+1) % self.N] - x[(i-2) % self.N]
                if (((i+1) % self.N) == (j % self.N)):
                    m[i][j] += x[(i-1) % self.N]
                if (((i-2) % self.N) == (j % self.N)):
                    m[i][j] -= x[(i-1) % self.N]
                if ((i     % self.N) == (j % self.N)):
                    m[i][j] -= 1
        return m
        
    
class RK4:
    def __init__(self, N, dt):
        self.N = N
        self.dt = dt

    def nextstep(self, gradient, t, x):
        k1 = handler(gradient, t, x)
        k2 = handler(gradient, t + self.dt/2, x + k1*self.dt/2)
        k3 = handler(gradient, t + self.dt/2, x + k2*self.dt/2)
        k4 = handler(gradient, t + self.dt  , x + k3*self.dt)
        return x + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
    
    def orbit(self, gradient, t0, x0, T):
        t = np.arange(0., T, dt)
        steps = int(T/self.dt)
        o = np.zeros((steps,self.N))
        o[0] = np.copy(x0)
        for i in range(1,steps):
            o[i] = self.nextstep(gradient, t[i], o[i-1])
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
N = 40
F = 8
stddev = 1.
year = 2
day = 365 * year
dt = 0.01
T = day * 0.2
steps = int(T/dt)
interval = 5

lorenz = Lorenz96(N, F)
rk4 = RK4(N, dt)

t = np.arange(0., T, dt)
t_day = np.arange(0.,T/0.2, dt/0.2)

seeds = np.array([1,2])

for seed in seeds:
    rng = np.random.RandomState(seed)
    
    x0 = F * np.ones(N)
    x0[rng.randint(N)] += 0.1*rng.randn()
    
    true_orbit, observed = rk4.true_observed(lorenz.gradient, 0., x0, T, stddev, rng)
    
    print ("observation RMSE: ", np.mean([np.linalg.norm(observed[i] - true_orbit[i])/math.sqrt(N) for i in range(steps)]))
    
    assimilation_xzero = observed[rng.randint(len(observed))]
    with open('data/assimilation_xzero.' + str(seed) + '.dat', 'w') as ff:
        ff.write(("%f\t"*N + "\n") % tuple(assimilation_xzero))
        ff.close()
        
    with open('data/year.' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps):
            f.write(("%f\t"*N + "\n") % tuple(true_orbit[i,:]))
        f.close()

    sparse_orbit = []
    with open('data/year.' + str(interval) + '.' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps, interval):
            f.write(("%f\t"*N + "\n") % tuple(true_orbit[i,:]))
            sparse_orbit.append(true_orbit[i,:])
        f.close()
    
#    true_orbit_every6h = []
#    with open("data/year6h."+ str(seed) +".dat", "w") as h:
#        for i in range(int(steps/2),steps,interval):
#            h.write(("%f\t"*N + "\n") % tuple(true_orbit[i,:]))
#            true_orbit_every6h.append(true_orbit[i,:])
#        h.close()

    with open('data/observed.' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps):
            f.write(("%f\t"*N + "\n") % tuple(observed[i,:]))
        f.close()

    sparse_obs = []
    with open('data/observed.' + str(interval) + '.' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps, interval):
            f.write(("%f\t"*N + "\n") % tuple(observed[i,:]))
            sparse_obs.append(observed[i,:])
        f.close()
    
#    observed_every6h = []
#    with open("data/observed6h." + str(seed) + ".dat", "w") as g:    
#        for i in range(int(steps/2),steps,interval):
#            g.write(("%f\t"*N + "\n") % tuple(observed[i,:]))
#            observed_every6h.append(observed[i,:])
#        g.close()
    
#    np.savetxt("data/cov." + str(seed) + ".dat", np.cov(np.transpose(np.asarray(observed_every6h))))
    
    np.savetxt("data/cov." + str(interval) + "." + str(seed) + ".dat", np.cov(np.transpose(np.asarray(sparse_obs))))
    
#    plot_orbit(np.asarray(true_orbit_every6h))
#    plot_orbit(np.asarray(observed_every6h))


#t_day_every6h = []
#for i in range(int(steps/2),steps,5):
#    t_day_every6h.append(t_day[i])
#
##%%
#plt.xlabel("day")
#plt.plot(t_day_every6h[0:100],[item[0] for item in observed_every6h[0:100]], label='observed')
#plt.plot(t_day_every6h[0:100],[item[0] for item in true_orbit_every6h[0:100]], label='true_orbit')
#plt.legend()
#plt.show()