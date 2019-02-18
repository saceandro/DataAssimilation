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

import scipy.stats

def handler(func, *args):
    return func(*args)

#%%
class Hill:
    def __init__(self, ctrl, a, b, K, n):
        self.ctrl = ctrl
        self.a = a
        self.b = b
        self.K = K
        self.n = n
        
    def gradient(self,t,x):
        u = self.ctrl(t)
        u_n = math.pow(u, self.n)
        K_n = math.pow(self.K, self.n)

        return self.a * x + u_n / (K_n + u_n) * self.b

    def jacobian(self, x):
        return self.a        
    
class RK4:
    def __init__(self, dx, dt):
        self.dt = dt
        self.dx = dx

    def nextstep(self, t, x):
        k1 = handler(self.dx, t, x)
        k2 = handler(self.dx, t + self.dt/2, x + k1*self.dt/2)
        k3 = handler(self.dx, t + self.dt/2, x + k2*self.dt/2)
        k4 = handler(self.dx, t + self.dt  , x + k3*self.dt)
        return x + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
    
    def orbit(self, t0, x0, T):
        t = np.arange(0., T, dt)
        steps = int(T/self.dt)
        o = np.zeros(steps)
        o[0] = np.copy(x0)
        for i in range(1,steps):
#            o[i] = self.nextstep(t[i], o[i-1])
            o[i] = self.nextstep(t[i-1], o[i-1])
        return o

    def observed(self, t0, x0, T, stddev, rng):
        steps = int(T/self.dt)
        o = self.orbit(t0, x0, T)
        for i in range(steps):
            o[i] += stddev * rng.randn()
        return o

    def true_observed(self, t0, x0, T, stddev, rng):
        steps = int(T/self.dt)
        o = self.orbit(t0, x0, T)
        obs = np.copy(o)
        for i in range(steps):
            obs[i] += stddev * rng.randn()
        return o, obs

def plot_orbit(t, dat):
    fig = plt.figure()
    plt.plot(t,dat)
    plt.show()
    
def ramp(t):
    return t

def impulse(t):
    return scipy.stats.norm.pdf(t, 0, 0.1)

def rect(t):
    if (t <= 5):
        return 1.
    else:
        return 0.
    
def zero(t):
    return 0.

def sin(t):
    return 1.0 + math.sin(t)

def cos(t):
    return 1.0 + math.cos(t)

def cos2t(t):
    return 1.0 + math.cos(2.*t)

#%%
N = 1
stddev = 0.1
T = 20.
dt = 0.01
steps = int(T/dt)
interval = 5

a = -3.
b = 12.
K = 0.3132
n = 1.276

ctrls = [cos2t]
#ctrls = [ramp, impulse, rect, zero, sin, cos, cos2t]

for ctrl in ctrls:
    hill = Hill(ctrl, a, b, K, n)
    rk4 = RK4(hill.gradient, dt)
    
    t = np.arange(0., T, dt)
    
    seeds = np.array([1,2])
    
    for seed in seeds:
        rng = np.random.RandomState(seed)
        
        x0 = seed - 1
        true_orbit, observed = rk4.true_observed(0., x0, T, stddev, rng)
        print ("observation RMSE: ", np.mean(observed - true_orbit))
        
        assimilation_xzero = observed[rng.randint(len(observed))]
        
        pref = '/Users/konta/bitbucket/androsace/dacamp/hill/data/' + ctrl.__name__ + '/' +  str(N) + '/'
        if not os.path.exists(pref):
            os.mkdir(pref)
        
        with open(pref + 'assimilation_xzero.' + str(seed) + '.dat', 'w') as ff:
            ff.write(("%f\n") % assimilation_xzero)
            ff.close()
            
        with open(pref + 'true.' + str(seed) +'.dat', 'w') as f:
            for i in range(steps):
                f.write(("%f\n") % true_orbit[i])
            f.close()
    
        sparse_orbit = []
        with open(pref + 'true.' + str(interval) + '.' + str(seed) +'.dat', 'w') as f:
            for i in range(0, steps, interval):
                f.write(("%f\n") % true_orbit[i])
                sparse_orbit.append(true_orbit[i])
            f.close()
        
        with open(pref + 'observed.' + str(seed) +'.dat', 'w') as f:
            for i in range(steps):
                f.write(("%f\n") % observed[i])
            f.close()
    
        sparse_obs = []
        with open(pref + 'observed.' + str(interval) + '.' + str(seed) +'.dat', 'w') as f:
            for i in range(0, steps, interval):
                f.write(("%f\n") % observed[i])
                sparse_obs.append(observed[i])
            f.close()
        
        #np.savetxt(pref + "cov." + str(interval) + "." + str(seed) + ".dat", np.cov(np.transpose(np.asarray(sparse_obs))))
    
        plot_orbit(t, true_orbit)
        plot_orbit(t, observed)