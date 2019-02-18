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

count = 0

def handler(func, *args):
    return func(*args)

#%%
class Lorenz96:
    def __init__(self, N):
        self.N = N # number of variables
        self.M = 2 # number of parameters
    def gradient(self,x):
        d = np.zeros(self.N + self.M)
        d[0]        = x[self.N+1] * (x[1]   - x[self.N-2]) * x[self.N-1] - x[0]        + x[self.N]
        d[1]        = x[self.N+1] * (x[2]   - x[self.N-1]) * x[0]        - x[1]        + x[self.N]
        for i in range(2, self.N-1):
            d[i]    = x[self.N+1] * (x[i+1] - x[i-2])      * x[i-1]      - x[i]        + x[self.N]
        d[self.N-1] = x[self.N+1] * (x[0]   - x[self.N-3]) * x[self.N-2] - x[self.N-1] + x[self.N]
        return d
        
    def gradient_adjoint(self, la, x):
        mt = np.zeros((self.N + self.M ,self.N + self.M))
        for i in range(self.N):
            for j in range(self.N):
                if (((i-1) % self.N) == j):
                    mt[j][i] += x[self.N+1] * (x[(i+1) % self.N] - x[(i-2) % self.N])
                if (((i+1) % self.N) == j):
                    mt[j][i] += x[self.N+1] * x[(i-1) % self.N]
                if (((i-2) % self.N) == j):
                    mt[j][i] -= x[self.N+1] * x[(i-1) % self.N]
                if ((i     % self.N) == j):
                    mt[j][i] -= 1
            mt[N][i] = 1
            mt[N+1][i] = (x[(i+1) % self.N] - x[(i-2) % self.N]) * x[(i-1) % self.N]
        gr = mt @ la
        return gr

class RungeKutta4:
    def __init__(self, callback, N, dt, t, x):
        self.callback = callback
        self.N = N
        self.dt = dt
        self.t = t
        self.x = x
        self.M = 2

    def nextstep(self):
        k1 = handler(self.callback, self.x)
        k2 = handler(self.callback, self.x + k1*self.dt/2)
        k3 = handler(self.callback, self.x + k2*self.dt/2)
        k4 = handler(self.callback, self.x + k3*self.dt)
        self.t += self.dt
        self.x += (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
        return self.x
    
    def orbit(self,T):
        steps = int(T/self.dt) + 1
        o = np.zeros((steps,self.N + self.M))
        o[0] = self.x
        for i in range(steps):
            o[i] = self.nextstep()
        return o
    
    def nextstep_gradient(self):
        self.nextstep()
        return self.dt * self.callback(self.t, self.x)
    
    def orbit_gradient(self, T):
        steps = int(T/self.dt)
        gr = np.zeros((steps, self.N + self.M))
        gr[0] = self.dt * self.callback(self.t, self.x)
        for i in range(steps):
            gr[i] = self.nextstep_gradient()
        return gr

class Adjoint:
    def __init__(self, dx, dla, N, T, dt, it, x, y):
        self.dx = dx
        self.dla = dla
        self.N = N
        self.T = T
        self.dt = dt
        self.x = x
        self.y = y
        self.it = it
        self.minute_steps = int(T/self.dt)
        self.steps = int(self.minute_steps/it)
        self.M = 2
        
    def orbit(self):
        for i in range(self.minute_steps-1):
            k1 = handler(self.dx, self.x[i])
            k2 = handler(self.dx, self.x[i] + k1*self.dt/2)
            k3 = handler(self.dx, self.x[i] + k2*self.dt/2)
            k4 = handler(self.dx, self.x[i] + k3*self.dt)
            self.x[i+1] = self.x[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
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
        la = np.zeros((self.minute_steps, self.N + self.M))
        for i in range(self.steps-1, -1, -1):
            for j in range(it-1, -1, -1):
                n = self.it*i + j
                if (n < self.it*self.steps - 1):
                    p1 = handler(self.dx, self.x[n])
                    p2 = handler(self.dx, self.x[n] + p1*self.dt/2)
                    p3 = handler(self.dx, self.x[n] + p2*self.dt/2)
                    p4 = handler(self.dx, self.x[n] + p3*self.dt)
                    gr = (p1 + 2*p2 + 2*p3 + p4)/6
    
                    k1 = handler(self.dla, la[n+1], self.x[n+1])
                    k2 = handler(self.dla, la[n+1] - k1*self.dt/2, self.x[n+1] - gr*self.dt/2)
                    k3 = handler(self.dla, la[n+1] - k2*self.dt/2, self.x[n+1] - gr*self.dt/2)
                    k4 = handler(self.dla, la[n+1] - k3*self.dt, self.x[n])
                    la[n] = la[n+1] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6            
            for j in range(self.N):
                la[self.it*i][j] += self.x[self.it*i][j] - self.y[i][j]
        return la[0]

    def gradient_from_x0(self, x0):
        self.x[0] = x0
        self.orbit()
        la = np.zeros((self.minute_steps, self.N + self.M))
        for i in range(self.steps-1, -1, -1):
            for j in range(it-1, -1, -1):
                n = self.it*i + j
                if (n < self.it*self.steps - 1):
                    p1 = handler(self.dx, self.x[n])
                    p2 = handler(self.dx, self.x[n] + p1*self.dt/2)
                    p3 = handler(self.dx, self.x[n] + p2*self.dt/2)
                    p4 = handler(self.dx, self.x[n] + p3*self.dt)
                    gr = (p1 + 2*p2 + 2*p3 + p4)/6
    
                    k1 = handler(self.dla, la[n+1], self.x[n+1])
                    k2 = handler(self.dla, la[n+1] - k1*self.dt/2, self.x[n+1] - gr*self.dt/2)
                    k3 = handler(self.dla, la[n+1] - k2*self.dt/2, self.x[n+1] - gr*self.dt/2)
                    k4 = handler(self.dla, la[n+1] - k3*self.dt, self.x[n])
                    la[n] = la[n+1] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            for j in range(self.N):
                la[self.it*i][j] += self.x[self.it*i][j] - self.y[i][j]
        return la[0]
    
    def cost(self, x0):
        self.x[0] = x0
        self.orbit()
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps):
            cost += (self.x[self.it*i][0:self.N] - self.y[i]) @ (self.x[self.it*i][0:self.N] - self.y[i])
        return cost/2.0 # fixed
    
    def true_cost(self):
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps):
            cost += (self.x[self.it*i][0:self.N] - self.y[i]) @ (self.x[self.it*i][0:self.N] - self.y[i])
        return cost/2.0 # fixed
    
    def numerical_gradient_from_x0(self,x0,h):
        gr = np.zeros(self.N + self.M)
        c1 = self.cost(x0)
        for j in range(self.N + self.M):
            xx = np.copy(x0)
            xx[j] += h
            c = self.cost(xx)
            gr[j] = (c - c1)/h
        return gr
    
    def cbf(self, x0):
        global count, axL, axR
        count += 1
        axLL.scatter(count, x0[self.N], c='b')
        axL.scatter(count, x0[self.N+1], c='b')
        axR.scatter(count, self.cost(x0), c='b')

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
pref = "data/" + str(N) + "/"

M = 2
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

lorenz = Lorenz96(N)

tob = np.loadtxt(pref + "year.1.dat")

obs = np.loadtxt(pref + "observed." + str(it) + ".1.dat")

compare_orbit(tob[0:minute_steps], obs[0:steps])

t = np.arange(0., T, dt)

x_opt = np.zeros(N + M)
x_opt[0:N] = np.loadtxt(pref + "year.2.dat")[np.random.randint(len(tob))]
x_opt[N] = 4 # initial guess for F
x_opt[N+1] = 0.5 # initial guess for a

x = np.zeros((minute_steps, N + M))
scheme = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, it, x, obs)

print("Before assimilation")
print("cost", scheme.cost(x_opt))
compare_orbit3(tob[0:minute_steps], obs[0:steps], scheme.x[:,0:N], 'true_orbit', 'observed', 'initial value')
compare_orbit(tob[0:minute_steps], scheme.x[:,0:N])

print("Analytical and numerical gradient comparison")
gr_anal = scheme.gradient_from_x0(x_opt)
print ("gr_anal", gr_anal)
gr_num = scheme.numerical_gradient_from_x0(x_opt, 0.00001)
print ("gr_num", gr_num)
print ("relative error", (gr_anal - gr_num)/gr_num)

#%%
global axL
global axR
fig , (axLL, axL, axR) = plt.subplots(ncols=3, figsize=(10,4), sharex=False)
res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', callback=scheme.cbf)
print (res)
print ("true x0", tob[0])

for j in range(3):
#for j in range(N):
    fig = plt.figure()
    plt.plot(t, tob[0:minute_steps,j], label='true orbit')
    plt.plot(t, scheme.x[0:minute_steps,j], label='assimilated')
    plt.legend()
    plt.show()

compare_orbit(tob[0:minute_steps], scheme.x[:,0:N])

#%%
fig = plt.figure()
plt.plot(t, [np.linalg.norm(scheme.x[i,0:N] - tob[i])/math.sqrt(N) for i in range(len(t))], label='x norm')
plt.xlabel('t')
plt.ylabel('RMSE')
plt.yscale('symlog')
plt.legend()
plt.show()

print ("RMSE: ", np.mean([np.linalg.norm(scheme.x[i,0:N] - tob[i])/math.sqrt(N) for i in range(int(len(t)*0.4),int(len(t)*0.6))]))

print('4DVar optimal cost: ', res.fun)
scheme_true = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, it, tob, obs)
print('true cost: ', scheme_true.true_cost())
