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
    def gradient(self, x, la):
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
    
    def mt_m_dx(self, dx):
        mt = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if (((i-1) % self.N) == (j % self.N)):
                    mt[j][i] += dx[(i+1) % self.N] - dx[(i-2) % self.N]
                if (((i+1) % self.N) == (j % self.N)):
                    mt[j][i] += dx[(i-1) % self.N]
                if (((i-2) % self.N) == (j % self.N)):
                    mt[j][i] -= dx[(i-1) % self.N]
                if ((i     % self.N) == (j % self.N)):
                    mt[j][i] -= 1
                    
        m = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if (((i-1) % self.N) == (j % self.N)):
                    m[i][j] += dx[(i+1) % self.N] - dx[(i-2) % self.N]
                if (((i+1) % self.N) == (j % self.N)):
                    m[i][j] += dx[(i-1) % self.N]
                if (((i-2) % self.N) == (j % self.N)):
                    m[i][j] -= dx[(i-1) % self.N]
                if ((i     % self.N) == (j % self.N)):
                    m[i][j] -= 1
       
        return mt @ (m @ dx)
        
    def m_dx(self, dx):
        m = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if (((i-1) % self.N) == (j % self.N)):
                    m[i][j] += dx[(i+1) % self.N] - dx[(i-2) % self.N]
                if (((i+1) % self.N) == (j % self.N)):
                    m[i][j] += dx[(i-1) % self.N]
                if (((i-2) % self.N) == (j % self.N)):
                    m[i][j] -= dx[(i-1) % self.N]
                if ((i     % self.N) == (j % self.N)):
                    m[i][j] -= 1       
        return m @ dx

class RungeKutta4:
    def __init__(self, callback, N, dt, t, x, la):
        self.callback = callback
        self.N = N
        self.dt = dt
        self.t = t
        self.x = x
        self.la = la

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


class AdjointRungeKutta4:
    def __init__(self, callback, N, T, dt, x, y):
        self.callback = callback
        self.N = N
        self.T = T
        self.dt = dt
        self.x = x
        self.y = y
        self.steps = int(T/self.dt) + 1
        
    def orbit_adjoint(self):
        la = np.zeros((self.steps, N))
        for i in range(self.steps-1, 0, -1):
            k1 = handler(self.callback, i, la[i], self.x[i-1])
            k2 = handler(self.callback, i - self.dt/2, la[i] + k1*self.dt/2, self.x[i-1])
            k3 = handler(self.callback, i - self.dt/2, la[i] + k2*self.dt/2, self.x[i-1])
            k4 = handler(self.callback, i - self.dt  , la[i] + k3*self.dt, self.x[i-1])
            la[i-1] = la[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6 + self.x[i-1] - self.y[i-1]
        return la
    
    def minimizer_gradient(self):
        la = self.orbit_adjoint()
        return la[0]

class Adjoint:
    x_old = 0
    la_old = 0
    def __init__(self, dx, dla, N, T, dt, y, x_new, la_new):
        self.dx = dx
        self.dla = dla
        self.N = N
        self.T = T
        self.dt = dt
        self.y = y
        self.x_new = x_new
        self.la_new = la_new
        self.steps = int(T/self.dt) + 1
        self.x_old = np.zeros((self.steps, N))
        self.la_old = np.zeros((self.steps, N))
        
    def gradient(self):
        self.la_new[self.steps-1] = np.zeros(self.N)
        for i in range(self.steps-1, 0, -1):
            self.la_new[i] += self.x_old[i] - self.y[i]
            gr = handler(self.dx, self.x_old[i], self.la_old[i]) # approximate gradient
            
            k1 = handler(self.dla, self.la_new[i], self.x_old[i])
            k2 = handler(self.dla, self.la_new[i] - k1*self.dt/2, self.x_old[i] - gr*self.dt/2)
            k3 = handler(self.dla, self.la_new[i] - k2*self.dt/2, self.x_old[i] - gr*self.dt/2)
            k4 = handler(self.dla, self.la_new[i] - k3*self.dt, self.x_old[i] - gr*self.dt)
            
            self.la_new[i-1] = self.la_new[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            
        self.la_old = np.copy(self.la_new)
        return self.la_new[0] + self.x_old[0] - self.y[0]

    def gradient_from_x0(self, x0):
        self.x_new[0] = x0
        self.la_new[self.steps-1] = np.zeros(self.N)
        for i in range(self.steps-1, 0, -1):
            self.la_new[i] += self.x_old[i] - self.y[i]
            gr = handler(self.dx, self.x_old[i], self.la_old[i]) # approximate gradient
            
            k1 = handler(self.dla, self.la_new[i], self.x_old[i])
            k2 = handler(self.dla, self.la_new[i] - k1*self.dt/2, self.x_old[i] - gr*self.dt/2)
            k3 = handler(self.dla, self.la_new[i] - k2*self.dt/2, self.x_old[i] - gr*self.dt/2)
            k4 = handler(self.dla, self.la_new[i] - k3*self.dt, self.x_old[i] - gr*self.dt)
            
            self.la_new[i-1] = self.la_new[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            
        self.la_old = np.copy(self.la_new)
        return self.la_new[0] + self.x_old[0] - self.y[0]

    def orbit(self):
        for i in range(self.steps-1):
            gr = handler(self.dla, self.la_new[i], self.x_new[i]) # approximate gradient
            
            k1 = handler(self.dx, self.x_new[i], self.la_new[i])
            k2 = handler(self.dx, self.x_new[i] + k1*self.dt/2, self.la_new[i] - gr*self.dt/2) # calculated using approximate la
            k3 = handler(self.dx, self.x_new[i] + k2*self.dt/2, self.la_new[i] - gr*self.dt/2)
            k4 = handler(self.dx, self.x_new[i] + k3*self.dt, self.la_new[i] - gr*self.dt)
            self.x_new[i+1] = self.x_new[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            
        self.x_old = np.copy(self.x_new)
        return self.x_new
    
    def true_orbit(self):
        for i in range(self.steps-1):
            k1 = handler(self.dx, self.x_new[i], np.zeros(self.N))
            k2 = handler(self.dx, self.x_new[i] + k1*self.dt/2, np.zeros(self.N)) # calculated using approximate la
            k3 = handler(self.dx, self.x_new[i] + k2*self.dt/2, np.zeros(self.N))
            k4 = handler(self.dx, self.x_new[i] + k3*self.dt, np.zeros(self.N))
            self.x_new[i+1] = self.x_new[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
        return self.x_new
    
    def observed(self, stddev):
        self.true_orbit()
        for i in range(self.steps):
            for j in range(self.N):
                self.y[i,j] = self.x_new[i,j] + stddev * np.random.randn() # fixed
        return self.y

    def true_observed(self, stddev):
        self.true_orbit()
        for i in range(self.steps):
            for j in range(self.N):
                self.y[i,j] = self.x_new[i,j] + stddev * np.random.randn() # fixed
        return self.x_new, self.y
    
    def cost(self, x0):
        self.x_new[0] = x0
        self.gradient()
        self.orbit()
        
        la_sum = (self.la_new[0] + self.la_new[self.steps-1])/2 # integration using trapezoidal rule
        for i in range(1,self.steps-1):
            la_sum += self.la_new[i]
        la_sum *= self.T/self.steps
        
        cost = 0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps):
            cost += (self.x_new[i] - self.y[i]) @ (self.x_new[i] - self.y[i])
        return (cost + la_sum)/2.0 # fixed
    
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

N = 7
F = 8
T = 0.15
dt = 0.01
steps = int(T/dt) + 1

lorenz = Lorenz96(N, F)

t = np.arange(0., T + dt, dt)
y1 = np.zeros((steps, N))
la_new1 = np.zeros((steps, N))

x_true = np.zeros((steps, N))
x_true[0] = F * np.ones(N)
x_true[0][0] += 2
x_true[0][1] += 1
x_true[0][2] += 2

scheme1 = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, y1, x_true, la_new1)

tob, obs = scheme1.true_observed(1)
# print("y")
# plot_orbit(obs)
compare_orbit(tob, obs, 'true_orbit', 'observed_orbit')

x_new2 = np.zeros((steps, N))
la_new2 = np.zeros((steps, N))
x_opt = F * np.ones(N)
x_opt[0] += 1
x_opt[1] += 4
x_opt[2] += 5
x_new2[0] = np.copy(x_opt)

y2 = np.copy(obs)

scheme2 = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, y2, x_new2, la_new2)

scheme2.cost(x_opt)
print("x")

compare_orbit(tob, scheme2.x_new, 'true_orbit', 'initial value')
# compare_orbit3(tob, init_true, scheme2.x_new, 'initial value')

#gr_anal = scheme.gradient_from_x0(x_opt)
#print (gr_anal)
#gr_num = scheme.numerical_gradient_from_x0(x_opt, 0.001)
#print (gr_num)

#scheme.x[0] = F * np.ones(N)
#scheme.x[0][0] += 0.01
#
#objective_func_gradient = scheme.gradient()
#
#print (objective_func_gradient)
#
#plt.plot(t,[item[0] for item in y])
#plt.show()


#print (scheme.gradient_from_x0(x_opt))

for rep in range(1):
    # res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', options={'gtol': 1e-6, 'disp': True})
    res = minimize(scheme2.cost, x_opt, jac=scheme2.gradient_from_x0, method='L-BFGS-B', options={'gtol': 1e-6, 'disp': True})
    print (res)

    ans = np.copy(scheme2.x_new)
    plot_orbit(ans)
    
    for j in range(N):
        plt.plot(t,[item[j] for item in tob], label='true')
        plt.plot(t,[item[j] for item in ans], label='assimilated')
        plt.legend()
        plt.show()
    
    compare_orbit(tob, ans, 'true_orbit', 'assimilated')
    
    
#%%
#    
#dt = 0.0001
#T = 0.0001
#t = np.arange(0., T + dt, dt)
#
#x0 = F * np.ones(N)
#x0[0] += 2
#x0[1] += 1
#x0[2] += 2
#
#x0_copy = np.copy(x0)
#
#dx0 = np.zeros(N)
#for i in range(N):
#    dx0[i] = 0.0001
#
#x0_plus_dx0 = x0 + dx0
#x0_plus_dx0_copy = np.copy(x0_plus_dx0)
#
#rk4 = RungeKutta4(lorenz.gradient, N, dt, t, x0_copy)
#x1 = rk4.nextstep()
#
#rk4_delta = RungeKutta4(lorenz.gradient, N, dt, t, x0_plus_dx0_copy)
#x1_plus_dx1 = rk4_delta.nextstep()
#
#m_dx0 = (x1_plus_dx1 - x0_plus_dx0 - (x1 - x0))/dt
#
#lhs = m_dx0 @ m_dx0
#
#mdx0 = lorenz.m_dx(dx0)
#lhs2 = mdx0 @ mdx0
#
#mt_m_dx0 = lorenz.mt_m_dx(dx0)
#rhs = dx0 @ mt_m_dx0
#
#print('lhs', lhs)
#print('lhs2',lhs2)
#print('rhs', rhs)
