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
        
    def gradient_adjoint(self, la, x):
        m = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if (((i-1) % self.N) == (j % self.N)):
                    m[j][i] += x[(i+1) % self.N] - x[(i-2) % self.N]
                if (((i+1) % self.N) == (j % self.N)):
                    m[j][i] += x[(i-1) % self.N]
                if (((i-2) % self.N) == (j % self.N)):
                    m[j][i] -= x[(i-1) % self.N]
                if ((i     % self.N) == (j % self.N)):
                    m[j][i] -= 1
        gr = m @ la
        return gr

class RungeKutta4:
    def __init__(self, callback, N, dt, t, x):
        self.callback = callback
        self.N = N
        self.dt = dt
        self.t = t
        self.x = x

    def nextstep(self):
        k1 = handler(self.callback, self.t, self.x)
        k2 = handler(self.callback, self.t + self.dt/2, self.x + k1*self.dt/2)
        k3 = handler(self.callback, self.t + self.dt/2, self.x + k2*self.dt/2)
        k4 = handler(self.callback, self.t + self.dt  , self.x + k3*self.dt)
        self.t += self.dt
        self.x += (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
        return self.x
    
    def orbit(self,T):
        steps = int(T/self.dt) + 1
        o = np.zeros((steps,self.N))
        o[0] = self.x
        for i in range(steps):
            o[i] = self.nextstep()
        return o
    
    def nextstep_gradient(self):
        self.nextstep()
        return self.dt * self.callback(self.t, self.x)
    
    def orbit_gradient(self, T):
        steps = int(T/self.dt)
        gr = np.zeros((steps,N))
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
            self.x[i] += stddev * np.random.randn()
        return self.x

    def true_observed(self, stddev):
        tob = np.copy(self.orbit())
        for i in range(self.steps):
            self.x[i] += stddev * np.random.randn()
        return tob, self.x

    def gradient(self):
        la = np.zeros((self.minute_steps, self.N))
        for i in range(self.steps-1, -1, -1):
            for j in range(self.it-1, -1, -1):
                n = self.it*i + j
                print (n)
                p1 = handler(self.dx, self.x[n-1])
                p2 = handler(self.dx, self.x[n-1] + p1*self.dt/2)
                p3 = handler(self.dx, self.x[n-1] + p2*self.dt/2)
                p4 = handler(self.dx, self.x[n-1] + p3*self.dt)
                gr = (p1 + 2*p2 + 2*p3 + p4)/6
                
                k1 = handler(self.dla, la[n], self.x[n])
                k2 = handler(self.dla, la[n] - k1*self.dt/2, self.x[n] - gr*self.dt/2)
                k3 = handler(self.dla, la[n] - k2*self.dt/2, self.x[n] - gr*self.dt/2)
                k4 = handler(self.dla, la[n] - k3*self.dt, self.x[n-1])
                la[n-1] = la[n] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            n = self.it*i
            la[n-1] += self.x[n-1] - self.y[i-1]
        return la[0]

    def gradient_from_x0(self, x0):
        self.x[0] = x0
        self.orbit()
        la = np.zeros((self.minute_steps, self.N))
        for i in range(self.steps-1, -1, -1):
            for j in range(self.it-1, -1, -1):
                n = self.it*i + j
                print(n)
                p1 = handler(self.dx, self.x[n-1])
                p2 = handler(self.dx, self.x[n-1] + p1*self.dt/2)
                p3 = handler(self.dx, self.x[n-1] + p2*self.dt/2)
                p4 = handler(self.dx, self.x[n-1] + p3*self.dt)
                gr = (p1 + 2*p2 + 2*p3 + p4)/6
    
                k1 = handler(self.dla, la[n], self.x[n])
                k2 = handler(self.dla, la[n] - k1*self.dt/2, self.x[n] - gr*self.dt/2)
                k3 = handler(self.dla, la[n] - k2*self.dt/2, self.x[n] - gr*self.dt/2)
                k4 = handler(self.dla, la[n] - k3*self.dt, self.x[n-1])
                la[n-1] = la[n] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            n = self.it*i
            la[n-1] += self.x[n-1] - self.y[i-1]
        return la[0]
    
    def cost(self, x0):
        self.x[0] = x0
        self.orbit()
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps):
            cost += (self.x[i] - self.y[i]) @ (self.x[i] - self.y[i])
        return cost # fixed
    
    def numerical_gradient_from_x0(self,x0,h):
        gr = np.zeros(N)
        c1 = self.cost(x0)
        for j in range(N):
            xx = np.copy(x0)
            xx[j] += h
            c = self.cost(xx)
            gr[j] = (c - c1)/h
        return gr
#%%
def plot_orbit(dat):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(dat[:,0],dat[:,1],dat[:,2])
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    plt.show()

def compare_orbit(dat1, dat2, labe1, labe2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(dat1[:,0],dat1[:,1],dat1[:,2],label=labe1)
    ax.plot(dat2[:,0],dat2[:,1],dat2[:,2],label=labe2)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    plt.legend()
    plt.show()

def compare_orbit3(dat1, dat2, dat3, labe):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(dat1[:,0],dat1[:,1],dat1[:,2],label='true orbit')
    ax.plot(dat2[:,0],dat2[:,1],dat2[:,2],label='observed')
    ax.plot(dat3[:,0],dat3[:,1],dat3[:,2],label=labe)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    plt.legend()
    plt.show()
    
#%%
from scipy.optimize import minimize

N = 40
F = 8
year = 0.03


day = 365 * year
dt = 0.05
T = day * 0.2
it = 5
minute_steps = int(T/dt)
steps = int(minute_steps/it)

stddev = 1

lorenz = Lorenz96(N, F)

tob = np.loadtxt("data/year." + str(it) + ".1.dat")

obs = np.loadtxt("data/observed." + str(it) + ".1.dat")

compare_orbit(tob[0:minute_steps], obs[0:minute_steps], 'true orbit', 'observed')

t = np.arange(0., T-dt, dt)

x_opt = np.loadtxt("data/assimilation_xzero.2.dat")

x = np.zeros((minute_steps,N))
scheme = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, it, x, obs)



scheme.cost(x_opt)
print("x")
plot_orbit(scheme.x)
compare_orbit(tob[0:minute_steps], scheme.x, 'true_orbit', 'initial value')
#compare_orbit3(tob, scheme.y, scheme.x, 'initial value')

#%%
res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B')
print (res)

#scheme.cost(res.x)
#plot_orbit(scheme.x)

for j in range(3):
#for j in range(N):
    plt.plot(t, tob[0:minute_steps,j], label='true orbit')
    plt.plot(t, scheme.x[0:minute_steps,j], label='assimilated')
    plt.legend()
    plt.show()

compare_orbit(tob[0:minute_steps], scheme.x, 'true_orbit', 'assimilated')
#%%
#compare_orbit3(tob, scheme.y, ans, 'assimilated')

#%%
fig = plt.figure()
plt.plot(t, [np.linalg.norm(scheme.x[i] - tob[i])/math.sqrt(N) for i in range(len(t))], label='x norm')
plt.xlabel('day')
plt.ylabel('RMSE')
plt.legend()
plt.show()

print ("RMSE: ", np.mean([np.linalg.norm(scheme.x[i] - tob[i])/math.sqrt(N) for i in range(6,8)]))
