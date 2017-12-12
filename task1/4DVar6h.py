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
    
    def gradient_discrete(self, R, dt, T, orbit, y):
        ll = np.zeros((T + 2, self.N))
        for t in range(T-1, -1, -1):
            M = np.zeros((self.N,self.N))
            
            M[0][self.N-2] = -dt * orbit[t-1][self.N-1]
            M[0][self.N-1] = dt * (orbit[t-1][1] - orbit[t-1][self.N-2])
            M[0][0] = 1 - dt
            M[0][1] = dt * orbit[t-1][self.N-1]
    
            M[1][self.N-1] = -dt * orbit[t-1][0]
            M[1][0] = dt * (orbit[t-1][2] - orbit[t-1][self.N-1])
            M[1][1] = 1 - dt
            M[1][2] = dt * orbit[t-1][0]
    
            M[self.N-1][self.N-3] = -dt * orbit[t-1][self.N-2]
            M[self.N-1][self.N-2] = dt * (orbit[t-1][0] - orbit[t-1][self.N-3])
            M[self.N-1][self.N-1] = 1 - dt
            M[self.N-1][0] = dt * orbit[t-1][self.N-2]
    
            for i in range(2,self.N-1):
                M[i][i-2] = -dt * orbit[t-1][i-1]
                M[i][i-1] = dt * (orbit[t-1][i+1] - orbit[t-1][i-2])
                M[i][i] = 1 - dt
                M[i][i+1] = dt * orbit[t-1][i-1]
    
            ll[t] = (np.transpose(M)).dot(ll[t+1]) + np.linalg.inv(R).dot(orbit[t] - y[t])
        return ll[0]
    
    def gradient_adjoint(self, la, x, y):
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
        gr = -m @ la - (x - y)
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


class AdjointRungeKutta4:
    def __init__(self, callback, N, T, dt, x, y):
        self.callback = callback
        self.N = N
        self.T = T
        self.dt = dt
        self.x = x
        self.y = y
        self.steps = int(T/self.dt)
        
    def orbit_adjoint(self):
        la = np.zeros((self.steps, N))
        for i in range(self.steps-1, 0, -1):
            k1 = handler(self.callback, i, la[i], self.x[i], self.y[i])
            k2 = handler(self.callback, i - self.dt/2, la[i] + k1*self.dt/2, self.x[i], self.y[i])
            k3 = handler(self.callback, i - self.dt/2, la[i] + k2*self.dt/2, self.x[i], self.y[i])
            k4 = handler(self.callback, i - self.dt  , la[i] + k3*self.dt, self.x[i], self.y[i])
            la[i-1] = la[i] - (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
        return la
    
    def minimizer_gradient(self):
        la = self.orbit_adjoint()
        return la[0]


class Adjoint:
    def __init__(self, dx, dla, N, T, dt, x, y):
        self.dx = dx
        self.dla = dla
        self.N = N
        self.T = T
        self.dt = dt
        self.x = x
        self.y = y
        self.steps = int(T/self.dt)
        
    def orbit(self):
        for i in range(self.steps-1):
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
        la = np.zeros((self.steps, self.N))
        for i in range(self.steps-1, 0, -1):
            k1 = handler(self.dla, la[i], self.x[i], self.y[i])
            k2 = handler(self.dla, la[i] - k1*self.dt/2, self.x[i], self.y[i])
            k3 = handler(self.dla, la[i] - k2*self.dt/2, self.x[i], self.y[i])
            k4 = handler(self.dla, la[i] - k3*self.dt,   self.x[i], self.y[i])
            la[i-1] = la[i] - (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
        return la[0]

    def gradient_from_x0(self, x0):
        self.x[0] = x0
        self.orbit()
        la = np.zeros((self.steps, self.N))
        for i in range(self.steps-1, 0, -1):
            k1 = handler(self.dla, la[i], self.x[i], self.y[i])
            k2 = handler(self.dla, la[i] - k1*self.dt/2, self.x[i], self.y[i])
            k3 = handler(self.dla, la[i] - k2*self.dt/2, self.x[i], self.y[i])
            k4 = handler(self.dla, la[i] - k3*self.dt,   self.x[i], self.y[i])
            la[i-1] = la[i] - (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
        return la[0]
    
    def cost(self, x0):
        self.x[0] = x0
        self.orbit()
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps):
            cost += (self.x[i] - self.y[i]) @ (self.x[i] - self.y[i])
        return cost
    
    def numerical_gradient_from_x0(self,x0,h):
        gr = np.zeros(N)
        c1 = self.cost(x0)
        for j in range(N):
            xx = np.copy(x0)
            xx[j] += h
            c = self.cost(xx)
            gr[j] = (c - c1)/h
        return gr

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
year = 0.01
day = 365 * year
dt = 0.05
T = day * 0.2
it = 5
minute_steps = int(T/dt)
steps = int(minute_steps/it)
stddev = 1

lorenz = Lorenz96(N, F)

tob = np.loadtxt("data/year.1.dat")

obs = np.loadtxt("data/observed." + str(it) + ".1.dat")

t = np.arange(0., T, dt)

x_opt = np.loadtxt("data/assimilation_xzero.2.dat")

x = np.zeros((minute_steps,N))
scheme = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, x, obs)

plot_orbit(obs)

scheme.cost(x_opt)
print("x")
plot_orbit(scheme.x)
compare_orbit(tob, scheme.x, 'true_orbit', 'initial value')
#compare_orbit3(tob, scheme.y, scheme.x, 'initial value')

res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B')
print (res)

#scheme.cost(res.x)
#plot_orbit(scheme.x)

for j in range(N):
    plt.plot(t, scheme.y[0:minute_steps,0], label='true orbit')
    plt.plot(t, tob[0:minute_steps:,0], label='assimilated')
    plt.legend()
    plt.show()

compare_orbit(tob[0:minute_steps], scheme.x, 'true_orbit', 'assimilated')
#%%
#compare_orbit3(tob, scheme.y, ans, 'assimilated')