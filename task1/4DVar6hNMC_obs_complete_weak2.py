#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:01:40 2017

@author: yk
"""

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
    def gradient(self,x):
        d = np.zeros(self.N)
        d[0] = (x[1] - x[self.N-2]) * x[self.N-1] - x[0]
        d[1] = (x[2] - x[self.N-1]) * x[0]- x[1]
        d[self.N-1] = (x[0] - x[self.N-3]) * x[self.N-2] - x[self.N-1]
        for i in range(2, self.N-1):
            d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
        return d + self.F

    def gradient_sysnoise(self, x, la):
        d = np.zeros(self.N)
        d[0] = (x[1] - x[self.N-2]) * x[self.N-1] - x[0]
        d[1] = (x[2] - x[self.N-1]) * x[0]- x[1]
        d[self.N-1] = (x[0] - x[self.N-3]) * x[self.N-2] - x[self.N-1]
        for i in range(2, self.N-1):
            d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
        return d + self.F - la
        
    def gradient_adjoint(self, la, x):
        mt = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if (((i-1) % self.N) == (j % self.N)):
                    mt[j][i] += x[(i+1) % self.N] - x[(i-2) % self.N]
                if (((i+1) % self.N) == (j % self.N)):
                    mt[j][i] += x[(i-1) % self.N]
                if (((i-2) % self.N) == (j % self.N)):
                    mt[j][i] -= x[(i-1) % self.N]
                if ((i     % self.N) == (j % self.N)):
                    mt[j][i] -= 1
        gr = mt @ la
        return gr

class Adjoint:
    def __init__(self, dx, dla, dx_sysnoise, N, T, dt, it, x, y):
        self.dx = dx
        self.dla = dla
        self.dx_sysnoise = dx_sysnoise
        self.N = N
        self.T = T
        self.dt = dt
        self.x = x
        self.y = y
        self.it = it
        self.minute_steps = int(T/self.dt)
        self.steps = int(self.minute_steps/it)
        self.la = np.zeros((self.minute_steps, N))
        self.dla_bak = np.zeros((self.minute_steps, N))
        
    def orbit(self):
        for i in range(self.minute_steps-1):
            k1 = handler(self.dx, self.x[i])
            k2 = handler(self.dx, self.x[i] + k1*self.dt/2)
            k3 = handler(self.dx, self.x[i] + k2*self.dt/2)
            k4 = handler(self.dx, self.x[i] + k3*self.dt)
            self.x[i+1] = self.x[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
        return self.x
    
    def orbit_sysnoise_smoothed(self, x_sysnoise_smoothed):
        x_sysnoise_smoothed[0] = np.copy(self.x[0])
        for i in range(self.minute_steps-1):
            k1 = handler(self.dx_sysnoise, x_sysnoise_smoothed[i]               , self.la[i])
            k2 = handler(self.dx_sysnoise, x_sysnoise_smoothed[i] + k1*self.dt/2, self.la[i] - self.dla_bak[i]*self.dt/2.)
            k3 = handler(self.dx_sysnoise, x_sysnoise_smoothed[i] + k2*self.dt/2, self.la[i] - self.dla_bak[i]*self.dt/2.)
            k4 = handler(self.dx_sysnoise, x_sysnoise_smoothed[i] + k3*self.dt  , self.la[i] - self.dla_bak[i]*self.dt)
            x_sysnoise_smoothed[i+1] = x_sysnoise_smoothed[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
        return self.x
        
    
    def observed(self, stddev):
        self.orbit()
        for i in range(self.steps):
            for j in range(self.N):
                self.x[i,j] += stddev * np.random.randn() # fixed
        return self.x

    def true_observed(self, stddev):
        tob = np.copy(self.orbit())
        for i in range(self.steps):
            for j in range(self.N):
                self.x[i,j] += stddev * np.random.randn() # fixed
        return tob, self.x
    
    def gradient(self):
        for i in range(self.steps-1, -1, -1):
            for j in range(it-1, -1, -1):
                n = self.it*i + j
                if (n < self.it*self.steps - 1):
                    p1 = handler(self.dx, self.x[n])
                    p2 = handler(self.dx, self.x[n] + p1*self.dt/2)
                    p3 = handler(self.dx, self.x[n] + p2*self.dt/2)
                    p4 = handler(self.dx, self.x[n] + p3*self.dt)
                    gr = (p1 + 2*p2 + 2*p3 + p4)/6
    
                    k1 = handler(self.dla, self.la[n+1], self.x[n+1])
                    k2 = handler(self.dla, self.la[n+1] - k1*self.dt/2, self.x[n+1] - gr*self.dt/2)
                    k3 = handler(self.dla, self.la[n+1] - k2*self.dt/2, self.x[n+1] - gr*self.dt/2)
                    k4 = handler(self.dla, self.la[n+1] - k3*self.dt, self.x[n])
                    
                    self.dla_bak[n] = (k1 + 2*k2 + 2*k3 + k4)/6.
                    self.la[n] = self.la[n+1] + self.dla_bak[n] * self.dt
            self.la[self.it*i] += self.x[self.it*i] - self.y[i]
        return self.la[0]

    def gradient_from_x0(self, x0):
        print('gradient_called')
        self.x[0] = x0
        self.orbit()
        for i in range(self.steps-1, -1, -1):
            for j in range(it-1, -1, -1):
                n = self.it*i + j
                if (n < self.it*self.steps - 1):
                    p1 = handler(self.dx, self.x[n])
                    p2 = handler(self.dx, self.x[n] + p1*self.dt/2)
                    p3 = handler(self.dx, self.x[n] + p2*self.dt/2)
                    p4 = handler(self.dx, self.x[n] + p3*self.dt)
                    gr = (p1 + 2*p2 + 2*p3 + p4)/6
    
                    k1 = handler(self.dla, self.la[n+1], self.x[n+1])
                    k2 = handler(self.dla, self.la[n+1] - k1*self.dt/2, self.x[n+1] - gr*self.dt/2)
                    k3 = handler(self.dla, self.la[n+1] - k2*self.dt/2, self.x[n+1] - gr*self.dt/2)
                    k4 = handler(self.dla, self.la[n+1] - k3*self.dt, self.x[n])
                    
                    self.dla_bak[n] = (k1 + 2*k2 + 2*k3 + k4)/6.
                    self.la[n] = self.la[n+1] + self.dla_bak[n] * self.dt
            self.la[self.it*i] += self.x[self.it*i] - self.y[i]
        return self.la[0]
    
    def cost(self, x0):
        print('cost_called')
        self.x[0] = x0
        self.orbit()
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps):
#            print ((self.x[self.it*i] - self.y[i]) @ (self.x[self.it*i] - self.y[i]))
            cost += (self.x[self.it*i] - self.y[i]) @ (self.x[self.it*i] - self.y[i])
        return cost/2.0 # fixed
    
    def true_cost(self):
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps):
            cost += (self.x[self.it*i][0:self.N] - self.y[i]) @ (self.x[self.it*i][0:self.N] - self.y[i])
        return cost/2.0 # fixed
    
    def numerical_gradient_from_x0(self,x0,h):
        gr = np.zeros(N)
        c1 = self.cost(x0)
        for j in range(N):
            xx = np.copy(x0)
            xx[j] += h
            c = self.cost(xx)
            gr[j] = (c - c1)/h
        return gr

    def cbf(self, x0):
        print('callback called')
        global count, axL, axR
        count += 1
        plt.scatter(count, self.cost(x0), c='b')
    
#%%
def plot_orbit(dat):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(dat[:,0],dat[:,1],dat[:,2])
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    plt.show()

def compare_orbit(dat1, dat2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(dat1[:,0],dat1[:,1],dat1[:,2],label='true orbit')
    ax.plot(dat2[:,0],dat2[:,1],dat2[:,2],label='assimilated')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    plt.legend()
    plt.show()

def compare_orbit3(dat1, dat2, dat3, label1, label2, label3):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(dat1[:,0],dat1[:,1],dat1[:,2],label=label1)
    ax.plot(dat2[:,0],dat2[:,1],dat2[:,2],label=label2)
    ax.plot(dat3[:,0],dat3[:,1],dat3[:,2],label=label3)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    plt.legend()
    plt.show()

    
#%%
from scipy.optimize import minimize

N = 7
pref = "/Users/konta/bitbucket/androsace/dacamp/task1/lorenz_sysnoise_data/" + str(N) + "/"

F = 8
year = 0.01


day = 365 * year
dt = 0.01

# T = day * 0.2
T = 1.0
print("T", T)
print("day", T/0.2)
it = 5
minute_steps = int(T/dt)
steps = int(minute_steps/it)

stddev = 1

lorenz = Lorenz96(N, F)

tob = np.loadtxt(pref + "year.1.dat")

obs = np.loadtxt(pref + "observed." + str(it) + ".1.dat")

compare_orbit(tob[0:minute_steps], obs[0:steps])

t = np.arange(0., T, dt)

x_opt = np.loadtxt(pref + "assimilation_xzero.2.dat")

x = np.zeros((minute_steps,N))
scheme = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, lorenz.gradient_sysnoise, N, T, dt, it, x, obs)


print("Before assimilation")
print("cost", scheme.cost(x_opt))
compare_orbit3(tob[0:minute_steps], obs[0:steps], scheme.x, 'true_orbit', 'observed', 'initial value')
compare_orbit(tob[0:minute_steps], scheme.x)

print("Analytical and numerical gradient comparison")
gr_anal = scheme.gradient_from_x0(x_opt)
print ("gr_anal", gr_anal)
gr_num = scheme.numerical_gradient_from_x0(x_opt, 0.001)
print ("gr_num", gr_num)
print ("relative error", (gr_anal - gr_num)/gr_num)

#%%
fig = plt.figure()
print ('grad desc start')
res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', callback=scheme.cbf)
print ('grad desc end')
print (res)
print ("true x0", tob[0])

x_sysnoise_smoothed = np.zeros((minute_steps, N))
scheme.orbit_sysnoise_smoothed(x_sysnoise_smoothed)

for j in range(3):
#for j in range(N):
    fig = plt.figure()
    plt.plot(t, tob[0:minute_steps,j], label='true orbit')
    plt.plot(t, x_sysnoise_smoothed[0:minute_steps,j], label='assimilated')
    plt.legend()
    plt.show()

compare_orbit(tob[0:minute_steps], x_sysnoise_smoothed)

#%%
fig = plt.figure()
plt.plot(t, [np.linalg.norm(x_sysnoise_smoothed[i] - tob[i])/math.sqrt(N) for i in range(len(t))], label='x norm')
plt.xlabel('t')
plt.ylabel('RMSE')
plt.yscale('symlog')
plt.legend()
plt.show()

print ("RMSE: ", np.mean([np.linalg.norm(x_sysnoise_smoothed[i] - tob[i])/math.sqrt(N) for i in range(int(len(t)*0.4),int(len(t)*0.6))]))

print('4DVar optimal cost: ', res.fun)

scheme_weak = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, lorenz.gradient_sysnoise, N, T, dt, it, x_sysnoise_smoothed, obs)
print('4DVarWeak optimal cost: ', scheme_weak.true_cost())

scheme_true = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, lorenz.gradient_sysnoise, N, T, dt, it, tob, obs)
print('true cost: ', scheme_true.true_cost())
