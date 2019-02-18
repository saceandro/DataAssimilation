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
    def __init__(self, N, dt):
        self.N = N
        self.M = 1
        self.dt = dt
        
#    def gradient(self, x, x_next):
#        x_next[0] =        ((x[1]   - x[self.N-2]) * x[self.N-1] + x[self.N]) * self.dt + x[0]        * (1. - self.dt)
#        x_next[1] =        ((x[2]   - x[self.N-1]) * x[0]        + x[self.N]) * self.dt + x[1]        * (1. - self.dt)
#        for i in range(2, self.N-1):
#            x_next[i] =    ((x[i+1] - x[i-2])      * x[i-1]      + x[self.N]) * self.dt + x[i]        * (1. - self.dt)
#        x_next[self.N-1] = ((x[0]   - x[self.N-3]) * x[self.N-2] + x[self.N]) * self.dt + x[self.N-1] * (1. - self.dt)
#        
#        x_next[self.N] = x[self.N]
#        return x_next

    def gradient(self, x):
        d = np.zeros(self.N + self.M)
        d[0] =        ((x[1]   - x[self.N-2]) * x[self.N-1] + x[self.N]) * self.dt + x[0]        * (1. - self.dt)
        d[1] =        ((x[2]   - x[self.N-1]) * x[0]        + x[self.N]) * self.dt + x[1]        * (1. - self.dt)
        for i in range(2, self.N-1):
            d[i] =    ((x[i+1] - x[i-2])      * x[i-1]      + x[self.N]) * self.dt + x[i]        * (1. - self.dt)
        d[self.N-1] = ((x[0]   - x[self.N-3]) * x[self.N-2] + x[self.N]) * self.dt + x[self.N-1] * (1. - self.dt)
        
        d[self.N] = x[self.N]
        return d
    
    def gradient_adjoint(self, la, x):
        # fastest code
        d = np.zeros(self.N + self.M)
        for j in range(self.N):
            d[j] = self.dt * (x[(j-2) % self.N] * la[(j-1) % self.N] - x[(j+1) % self.N] * la[(j+2) % self.N] + (x[(j+2) % self.N] - x[(j-1) % self.N]) * la[(j+1) % self.N])\
                 + (1. - self.dt) * la[j]
        d[self.N] = self.dt * sum(la[:self.N]) + la[self.N]
        return d
        
#    def tl(self, x):
#        m = np.zeros((self.N + self.M, self.N + self.M))
#        for i in range(self.N):
#            for j in range(self.N):
#                m[i,j] = self.dt * (((((i+1) % self.N) == j) - (((i-2) % self.N) == j)) * x[(i-1) % self.N] + (x[(i+1) % self.N] - x[(i-2) % self.N]) * (((i-1) % self.N) == j))\
#                       + (1. - self.dt) * (i == j)
#            m[i, self.N] = self.dt
#        m[self.N, self.N] = 1. # fixed
#        return m
#            
#    def gradient_adjoint(self, la, x):
#        return self.tl(x).transpose() @ la

    
class Adjoint:
    def __init__(self, dx, dla, N, T, dt, it, obs_variance, x, y):
        self.dx = dx
        self.dla = dla
        self.N = N
        self.M = 1
        self.T = T
        self.dt = dt
        self.x = x
        self.y = y
        self.it = it
        self.minute_steps = int(T/self.dt)
        self.steps = int(self.minute_steps/it)
        self.obs_variance = obs_variance
        
    def orbit(self):
        for i in range(self.minute_steps):
            self.x[i+1] = handler(self.dx, self.x[i])
#            handler(self.dx, self.x[i], self.x[i+1])
        return self.x
    
    def observed(self, stddev):
        self.orbit()
        self.x[:self.N] += stddev * np.random.randn(self.steps, self.N)
        return self.x

    def true_observed(self, stddev):
        tob = np.copy(self.orbit())
        self.x[:self.N] += stddev * np.random.randn(self.steps, self.N)
        return tob, self.x
    
    def gradient(self):
        la = np.zeros((self.minute_steps + 1, self.N + self.M))
        la[self.minute_steps, :self.N] = (self.x[self.minute_steps, :self.N] - self.y[self.steps])/self.obs_variance
        for i in range(self.steps-1, -1, -1):
            for j in range(it-1, -1, -1):
                n = self.it*i + j
                la[n] = handler(self.dla, la[n+1], self.x[n]) # x should be current one.
            la[self.it*i, :self.N] += (self.x[self.it*i, :self.N] - self.y[i])/self.obs_variance
        return la[0]

    def gradient_from_x0(self, x0):
        self.x[0] = np.copy(x0)
        self.orbit()
        return self.gradient()
    
    def cost(self, x0):
        self.x[0] = np.copy(x0)
        self.orbit()
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps+1):
#            print ((self.x[self.it*i] - self.y[i]) @ (self.x[self.it*i] - self.y[i]))
            cost += (self.x[self.it*i, 0:self.N] - self.y[i]) @ (self.x[self.it*i, 0:self.N] - self.y[i])
        return cost/self.obs_variance/2.0 # fixed
    
    def true_cost(self):
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps+1):
            cost += (self.x[self.it*i][0:self.N] - self.y[i]) @ (self.x[self.it*i][0:self.N] - self.y[i])
        return cost/self.obs_variance/2.0 # fixed
    
    def numerical_gradient_from_x0(self,x0,h):
        gr = np.zeros(self.N + self.M)
        c1 = self.cost(x0)
        for j in range(self.N + self.M):
            xx = np.copy(x0)
            xx[j] += h
#            print(xx)
            c = self.cost(xx)
            gr[j] = (c - c1)/h
        return gr

    def cbf(self, x0):
        global count
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
M = 1
#pref = "data/" + str(N) + "/"
#pref = "/Users/konta/bitbucket/androsace/dacamp/task1/lorenz_sysnoise_data/" + str(N) + "/"
pref = "/Users/konta/bitbucket/androsace/dacamp/task1/lorenz_discrete_data/" + str(N) + "/"

F = 8
year = 0.01


day = 365 * year
dt = 0.01

# T = day * 0.2
T = 1.
print("T", T)
print("day", T/0.2)
it = 5
minute_steps = int(T/dt)
steps = int(minute_steps/it)

stddev = 1.

lorenz = Lorenz96(N, dt)

tob = np.loadtxt(pref + "year.1.dat")

obs = np.loadtxt(pref + "observed." + str(it) + ".1.dat")

compare_orbit(tob[0:minute_steps], obs[0:steps])

t = np.arange(0., T, dt)
t_plus_1 = np.arange(0., T+dt, dt)
t_it = np.arange(0., T, dt*it)
t_it_plus_1 = np.arange(0., T+dt, dt*it)

x_opt = np.zeros(N + M)
x_opt[0:N] = np.loadtxt(pref + "assimilation_xzero.2.dat")
#x_opt[0:N] = np.copy(tob[0])
x_opt[N] = 4.

x = np.zeros((minute_steps+1, N + M))
scheme = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, it, stddev, x, obs)


print("Before assimilation")
print("cost", scheme.cost(x_opt))
compare_orbit3(tob[0:minute_steps+1], obs[0:steps+1], scheme.x, 'true_orbit', 'observed', 'initial value')
compare_orbit(tob[0:minute_steps+1], scheme.x)

print("Analytical and numerical gradient comparison")
gr_anal = scheme.gradient_from_x0(x_opt)
print ("gr_anal", gr_anal)
gr_num = scheme.numerical_gradient_from_x0(x_opt, 0.001)
print ("gr_num", gr_num)
print ("relative error", (gr_anal - gr_num)/gr_num)

#%%
fig = plt.figure()
count = 0
#bnds = tuple([(-8., 24.) for i in range(N)])
res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', callback=scheme.cbf, options={'disp': None, 'maxls': 40, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
#res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', callback=scheme.cbf)
plt.show()
print (res)
print ("true x0", tob[0])

for j in range(N):
    fig = plt.figure()
    plt.plot(t_plus_1, tob[0:minute_steps+1,j], label='true orbit')
    plt.plot(t_plus_1, scheme.x[0:minute_steps+1,j], label='assimilated')
    plt.plot(t_it_plus_1, obs[0:steps+1,j], label='observed')
    plt.legend()
    plt.show()

compare_orbit(tob[0:minute_steps+1], scheme.x[:,0:N])

#%%
fig = plt.figure()
plt.plot(t_plus_1, [np.linalg.norm(scheme.x[i,0:N] - tob[i])/math.sqrt(N) for i in range(len(t_plus_1))], label='x norm')
plt.xlabel('t')
plt.ylabel('RMSE')
#plt.yscale('symlog')
plt.legend()
plt.show()

print ("RMSE: ", np.mean([np.linalg.norm(scheme.x[i,0:N] - tob[i])/math.sqrt(N) for i in range(int(len(t_plus_1)*0.4),int(len(t_plus_1)*0.6))]))

#print('4DVar optimal cost: ', res.fun)
#scheme_true = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, it, stddev, tob, obs)
#print('true cost: ', scheme_true.true_cost())
