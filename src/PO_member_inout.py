#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:01:40 2017

@author: yk
"""

import numpy as np
import math

import sys

def handler(func, *args):
    return func(*args)

#%%
class Lorenz96:
    def __init__(self, N, F, m):
        self.N = N
        self.F = F
        self.m = m
        
    def gradient(self,t,x):
        d = np.zeros((self.N, self.m))
        d[0] = (x[1] - x[self.N-2]) * x[self.N-1] - x[0]
        d[1] = (x[2] - x[self.N-1]) * x[0]- x[1]
        d[self.N-1] = (x[0] - x[self.N-3]) * x[self.N-2] - x[self.N-1]
        for i in range(2, self.N-1):
            d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
        return d + self.F        
    
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

    def observed(self, gradient, t0, x0, T, stddev):
        steps = int(T/self.dt)
        o = self.orbit(gradient, t0, x0, T)
        for i in range(steps):
            o[i] += stddev * np.random.randn()
        return o

# x = (N, m) where N is #vars and m is #ensemble_members
# R means R'
class POstep:
    def __init__(self, model, intmodel, H, R_original, m, rng):
        self.model = model
        self.intmodel = intmodel
        self.H = H
        self.R = (m-1) * R_original
        self.RT = np.linalg.inv(R_original) / (m-1)
        self.Im = np.identity(m)
        self.rng = rng

    def predict(self, xa):
        xf = self.intmodel.nextstep(self.model.gradient, 0., xa)
        xf_mean = np.mean(xf, axis=1)
        return xf, xf_mean
        
    def update(self, xf, xf_mean, y):
        dxf = xf - np.transpose(np.array([xf_mean for k in range(m)]))
        dyf = self.H @ dxf
        dyfT = np.transpose(dyf)
        Gain = dxf @ np.linalg.inv(self.Im + dyfT @ self.RT @ dyf) @ dyfT @ self.RT
#        Gain = dxf @ dyfT @ np.linalg.inv(dyf @ dyfT + self.R)
#        print (Gain)
#        print (np.linalg.norm(np.trace(Gain)/np.sqrt(40)))
        
        xa = xf + Gain @ self.H @ (y + self.rng.randn(N, m) - xf)
        xa_mean = np.mean(xa, axis=1)
        return xa, xa_mean

class PO:
    def __init__(self, postep, T, dt, xf, xa, xf_mean, xa_mean, y, it, m):
        self.it = it
        self.minute_steps = int(T/dt)
        self.steps = int(minute_steps/it)
        self.m = m
        self.postep = postep
        self.xf = xf
        self.xa = xa
        self.xf_mean = xf_mean
        self.xa_mean = xa_mean
        self.y = y
    
    def filtering(self):
        k = 0
        self.xa[0], self.xa_mean[0] = self.postep.update(self.xf[0], self.xf_mean[0], self.y[0])
        for i in range(1, self.steps):
            self.xf[k+1], self.xf_mean[k+1] = self.postep.predict(self.xa[i-1])
            k += 1
            for j in range(1, self.it):
                self.xf[k+1], self.xf_mean[k+1] = self.postep.predict(self.xf[k])
                k += 1
            self.xa[i], self.xa_mean[i] = self.postep.update(self.xf[k], self.xf_mean[k], self.y[i])
    
    def pred_cov(self):
        Pa = np.zeros((self.steps, N, N))
        for i in range(self.steps):
            dxa = self.xa[i] - np.transpose(np.array([self.xf_mean[i] for k in range(m)]))
            Pa[i] = dxa @ np.transpose(dxa) / (self.m-1)
        return Pa

#%%
N = 40
F = 8
year = 1
day = 365 * year
dt = 0.01
T = day * 0.2
it = 5
minute_steps = int(T/dt)
steps = int(minute_steps/it)
stddev = 1
M = 40
m = int(sys.argv[1])
seed = 1

rng = np.random.RandomState(seed)

obs_index = np.random.choice(N, M, replace=False)


true_orbit = np.loadtxt("data/year.1.dat")

y1 = np.loadtxt("data/observed." + str(it) + ".1.dat")
y = np.array([np.transpose(np.array([np.copy(y1[i]) for j in range(m)])) for i in range(steps)])
y2 = np.loadtxt("data/observed." + str(it) + ".2.dat")

R_original = np.zeros((M, M))
np.fill_diagonal(R_original, 1)

H = np.zeros((M, N))
count = 0
for i in obs_index:
    H[count][i] = 1
    count += 1
    
#%%
lorenz = Lorenz96(N, F, m)
rk4 = RK4(N, dt)

t = np.arange(0., T, dt)
t_day = np.copy(t)/0.2
t_day_every6h = []
for i in range(0,minute_steps,it):
    t_day_every6h.append(t_day[i])

xf = np.zeros((minute_steps, N, m))
xf[0] = np.transpose(np.array([y2[i] for i in np.random.choice(steps, m, replace=False)]))
xf_mean = np.zeros((minute_steps, N))

xa = np.zeros((steps, N, m))
xa_mean = np.zeros((steps, N))

postep = POstep(lorenz, rk4, H, R_original, m, rng)
po = PO(postep, T, dt, xf, xa, xf_mean, xa_mean, y, it, m)
po.filtering()
Pa = po.pred_cov()

print (m, np.mean([np.linalg.norm(po.xa_mean[i] - true_orbit[i*it])/math.sqrt(N) for i in range(1000,(int(len(t)/it)))]))
