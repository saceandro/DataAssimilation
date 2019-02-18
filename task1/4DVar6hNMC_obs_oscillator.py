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
    
    def mt_m_dx(self, dx):
        mt = np.zeros((self.N,self.N))
        mt[0][1] = -1
        mt[1][0] = 1
                    
        m = np.zeros((self.N,self.N))
        m[0][1] = 1
        m[1][0] = -1
        
        return mt @ (m @ dx)
        
    def m_dx(self, dx):
        m = np.zeros((self.N,self.N))
        m[0][1] = 1
        m[1][0] = -1
        return m @ dx

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

class RungeKutta4:
    def __init__(self, callback, N, dt, t, x):
        self.callback = callback
        self.N = N
        self.dt = dt
        self.t = t
        self.x = x

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
        la = np.zeros((self.minute_steps, self.N))
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
            la[self.it*i] += 10*(self.x[self.it*i] - self.y[i])
        return la[0]

    def gradient_from_x0(self, x0):
        self.x[0] = x0
        self.orbit()
        la = np.zeros((self.minute_steps, self.N))
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
            la[self.it*i] += 10*(self.x[self.it*i] - self.y[i])
        return la[0]
    
    def cost(self, x0):
        self.x[0] = x0
        self.orbit()
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps):
#            print (10*(self.x[self.it*i] - self.y[i]) @ (self.x[self.it*i] - self.y[i]))
            cost += 10*(self.x[self.it*i] - self.y[i]) @ (self.x[self.it*i] - self.y[i])
        print()
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

#%%
def plot_orbit(dat):
    fig = plt.figure()
    plt.plot(dat[:,0],dat[:,1])
    plt.show()

def compare_orbit(dat1, dat2, labe1, labe2):
    fig = plt.figure()
    plt.plot(dat1[:,0],dat1[:,1],label=labe1)
    plt.plot(dat2[:,0],dat2[:,1],label=labe2)
    plt.legend()
    plt.show()

def compare_orbit3(dat1, dat2, dat3, labe):
    fig = plt.figure()
    plt.plot(dat1[:,0],dat1[:,1],label='true orbit')
    plt.plot(dat2[:,0],dat2[:,1],label='observed')
    plt.plot(dat3[:,0],dat3[:,1],label=labe)
    plt.legend()
    plt.show()
    
#%%
from scipy.optimize import minimize

N = 2
year = 0.01


day = 365 * year
dt = 0.01

# T = day * 0.2
T = 0.1
print("T", T)
print("day", T/0.2)
it = 5
minute_steps = int(T/dt)
steps = int(minute_steps/it)

stddev = 1

oscillator = Linear(N)

tob = np.loadtxt("oscillator_data/year.1.dat")

obs = np.loadtxt("oscillator_data/observed." + str(it) + ".1.dat")

compare_orbit(tob[0:minute_steps], obs[0:steps], 'true orbit', 'observed')

t = np.arange(0., T, dt)

# x_opt = np.loadtxt("oscillator_data/assimilation_xzero.2.dat")
x_opt = np.random.uniform(-10,10,2)
print ("init x0", x_opt)


x = np.zeros((minute_steps,N))
scheme = Adjoint(oscillator.gradient, oscillator.gradient_adjoint, N, T, dt, it, x, obs)


print("cost", scheme.cost(x_opt))
print("x")
plot_orbit(scheme.x)
compare_orbit(tob[0:minute_steps], scheme.x, 'true_orbit', 'initial value')
compare_orbit(obs[0:steps], scheme.x, 'obs', 'initial_value')
#compare_orbit3(tob, scheme.y, scheme.x, 'initial value')


gr_anal = scheme.gradient_from_x0(x_opt)
print ("gr_anal", gr_anal)
gr_num = scheme.numerical_gradient_from_x0(x_opt, 0.001)
print ("gr_num", gr_num)
print ("relative error", (gr_anal - gr_num)/gr_num)

#%%
res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B')
print (res)

print ("true x0", tob[0])
print("cost", scheme.cost(res.x))
plot_orbit(scheme.x)

for j in range(N):
#for j in range(N):
    plt.plot(t, tob[0:minute_steps,j], label='true orbit')
    plt.plot(t, scheme.x[0:minute_steps,j], label='assimilated')
    plt.legend()
    plt.show()

compare_orbit(tob[0:minute_steps], scheme.x, 'true_orbit', 'assimilated')
compare_orbit(obs[0:steps], scheme.x, 'obs', 'assimilated')
#%%
#compare_orbit3(tob, scheme.y, ans, 'assimilated')

#%%
fig = plt.figure()
plt.plot(t, [np.linalg.norm(scheme.x[i] - tob[i])/math.sqrt(N) for i in range(len(t))], label='x norm')
plt.xlabel('day')
plt.ylabel('RMSE')
plt.yscale('symlog')
plt.legend()
plt.show()

print ("RMSE: ", np.mean([np.linalg.norm(scheme.x[i] - tob[i])/math.sqrt(N) for i in range(int(len(t)*0.9),int(len(t)))]))
