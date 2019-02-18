#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:17:20 2017

@author: yk
"""
import sys
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math

def handler(func, *args):
    return func(*args)

#%%
#class Lorenz96:
#    def __init__(self, N, F, dt):
#        self.N = N
#        self.F = F
#        self.dt = dt
#        
##    def gradient(self,t,x):
##        d = np.zeros(self.N)
##        d[0] = (x[1] - x[self.N-2]) * x[self.N-1] - x[0]
##        d[1] = (x[2] - x[self.N-1]) * x[0]- x[1]
##        d[self.N-1] = (x[0] - x[self.N-3]) * x[self.N-2] - x[self.N-1]
##        for i in range(2, self.N-1):
##            d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
##        return d + self.F    
#
#    def nextstep(self, x, dt):
#        z = np.zeros(self.N)
#        z[0] =        ((x[1]   - x[self.N-2]) * x[self.N-1] + self.F) * self.dt + x[0]        * (1. - self.dt)
#        z[1] =        ((x[2]   - x[self.N-1]) * x[0]        + self.F) * self.dt + x[1]        * (1. - self.dt)
#        for i in range(2, self.N-1):
#            z[i] =    ((x[i+1] - x[i-2])      * x[i-1]      + self.F) * self.dt + x[i]        * (1. - self.dt)
#        z[self.N-1] = ((x[0]   - x[self.N-3]) * x[self.N-2] + self.F) * self.dt + x[self.N-1] * (1. - self.dt)
#        return z        
    
class Lorenz:
    def __init__(self, N, F, dt):
        self.N = N
        self.F = F
        self.dt = dt
        self.x = np.zeros(self.N)
        self.x_bak = np.zeros(self.N)

    def nextstep(self):
        self.x_bak[0] =        ((self.x[1]   - self.x[self.N-2]) * self.x[self.N-1] + self.F) * self.dt + self.x[0]        * (1. - self.dt)
        self.x_bak[1] =        ((self.x[2]   - self.x[self.N-1]) * self.x[0]        + self.F) * self.dt + self.x[1]        * (1. - self.dt)
        for i in range(2, self.N-1):
            self.x_bak[i] =    ((self.x[i+1] - self.x[i-2])      * self.x[i-1]      + self.F) * self.dt + self.x[i]        * (1. - self.dt)
        self.x_bak[self.N-1] = ((self.x[0]   - self.x[self.N-3]) * self.x[self.N-2] + self.F) * self.dt + self.x[self.N-1] * (1. - self.dt)
        self.x = self.x_bak
        return self.x

    def orbit(self, x0, T):
        self.x = np.copy(x0)
        steps = int(T/self.dt)
        o = np.zeros((steps,self.N))
        o[0] = np.copy(x0)
        for i in range(1,steps):
            o[i] = np.copy(self.nextstep())
        return o

    def observed(self, x0, T, stddev, rng):
        steps = int(T/self.dt)
        return self.orbit(x0, T) + stddev * rng.randn(steps, self.N)

    def true_observed(self, x0, T, stddev, rng):
        steps = int(T/self.dt)
        o = self.orbit(x0, T)
        return o, o + stddev * rng.randn(steps, self.N)

def plot_orbit(dat):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(dat[:,0],dat[:,1],dat[:,2])
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    plt.show()
    
    
#%%
N = int(sys.argv[1])
F = 8
stddev = 1.
year = 2
day = 365 * year
dt = 0.01
T = day * 0.2
steps = int(T/dt)
interval = 5

scheme = Lorenz(N, F, dt)

t = np.arange(0., T, dt)
t_day = np.arange(0.,T/0.2, dt/0.2)

seeds = np.array([1,2])

for seed in seeds:
    rng = np.random.RandomState(seed)
    
    x0 = F * np.ones(N)
    x0[rng.randint(N)] += 0.1*rng.randn()
    
    true_orbit, observed = scheme.true_observed(x0, T, stddev, rng)
    
    print ("observation RMSE: ", np.mean([np.linalg.norm(observed[i] - true_orbit[i])/math.sqrt(N) for i in range(steps)]))
    
    assimilation_xzero = observed[rng.randint(len(observed))]
    
    pref = '/Users/konta/bitbucket/androsace/dacamp/task1/lorenz_discrete_data/' + str(N) + '/'
    if not os.path.exists(pref):
        os.mkdir(pref)
    
    with open(pref + 'assimilation_xzero.' + str(seed) + '.dat', 'w') as ff:
        ff.write(("%f\t"*N + "\n") % tuple(assimilation_xzero))
        ff.close()
        
    with open(pref + 'year.' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps):
            f.write(("%f\t"*N + "\n") % tuple(true_orbit[i,:]))
        f.close()

    sparse_orbit = []
    with open(pref + 'year.' + str(interval) + '.' + str(seed) +'.dat', 'w') as f:
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

    with open(pref + 'observed.' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps):
            f.write(("%f\t"*N + "\n") % tuple(observed[i,:]))
        f.close()

    sparse_obs = []
    with open(pref + 'observed.' + str(interval) + '.' + str(seed) +'.dat', 'w') as f:
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
    
    np.savetxt(pref + "cov." + str(interval) + "." + str(seed) + ".dat", np.cov(np.transpose(np.asarray(sparse_obs))))
    
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