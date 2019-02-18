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
    def __init__(self, dx, dla, N, T, dt, x, y):
        self.dx = dx
        self.dla = dla
        self.N = N
        self.T = T
        self.dt = dt
        self.x = x
        self.y = y
        self.steps = int(T/self.dt) + 1
        
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
        la = np.zeros((self.steps, self.N))
        for i in range(self.steps-1, 0, -1):
            la[i] += self.x[i] - self.y[i]
            p1 = handler(self.dx, self.x[i-1])
            p2 = handler(self.dx, self.x[i-1] + p1*self.dt/2)
            p3 = handler(self.dx, self.x[i-1] + p2*self.dt/2)
            p4 = handler(self.dx, self.x[i-1] + p3*self.dt)
            gr = (p1 + 2*p2 + 2*p3 + p4)/6
            
            k1 = handler(self.dla, la[i], self.x[i])
            k2 = handler(self.dla, la[i] - k1*self.dt/2, self.x[i] - gr*self.dt/2)
            k3 = handler(self.dla, la[i] - k2*self.dt/2, self.x[i] - gr*self.dt/2)
            k4 = handler(self.dla, la[i] - k3*self.dt, self.x[i-1])
            la[i-1] = la[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
        la[0] += self.x[0] - self.y[0]
        return la[0]

    def gradient_from_x0(self, x0):
        self.x[0] = x0
        self.orbit()
        la = np.zeros((self.steps, self.N))
        for i in range(self.steps-1, 0, -1):
            la[i] += self.x[i] - self.y[i]
            p1 = handler(self.dx, self.x[i-1])
            p2 = handler(self.dx, self.x[i-1] + p1*self.dt/2)
            p3 = handler(self.dx, self.x[i-1] + p2*self.dt/2)
            p4 = handler(self.dx, self.x[i-1] + p3*self.dt)
            gr = (p1 + 2*p2 + 2*p3 + p4)/6

            k1 = handler(self.dla, la[i], self.x[i])
            k2 = handler(self.dla, la[i] - k1*self.dt/2, self.x[i] - gr*self.dt/2)
            k3 = handler(self.dla, la[i] - k2*self.dt/2, self.x[i] - gr*self.dt/2)
            k4 = handler(self.dla, la[i] - k3*self.dt, self.x[i-1])
            la[i-1] = la[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
        la[0] += self.x[0] - self.y[0]
        return la[0]
    
    def cost(self, x0):
        self.x[0] = x0
        self.orbit()
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps):
            cost += (self.x[i] - self.y[i]) @ (self.x[i] - self.y[i])
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
N = 4
F = 8
#x0 = F * np.ones(N)
#x0[3] += 0.01
#lorenz = Lorenz96(N, F)
#scheme = RungeKutta4(lorenz.gradient, N, 0.001, 0, x0)
#o = scheme.orbit(10.)
#plot_orbit(o)
#
#from scipy.integrate import odeint
#def lorenz96(x,t):
#    # compute state derivatives
#    d = np.zeros(N)
#    # first the 3 edge cases: i=1,2,N
#    d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
#    d[1] = (x[2] - x[N-1]) * x[0]- x[1]
#    d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
#    # then the general case
#    for i in range(2, N-1):
#        d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
#    # add the forcing term
#    d = d + F
#    
#    # return the state derivatives
#    return d
#
## pdb.set_trace()
#x0 = F*np.ones(N) # initial state (equilibrium)
#x0[0] += 0.01 # add small perturbation to 20th variable
#t = np.arange(0.0, 2.0+0.01, 0.01)
#
#x = odeint(lorenz96, x0, t)
#
#plot_orbit(x)
#
##%%
#N = 36
#F = 8
#
#lorenz = Lorenz96(N, F)
#
#t = np.arange(0.0, 10., 0.01)
#
#x0 = F * np.ones(N)
#x0[19] += 0.01
#scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, x0)
#o1 = scheme.orbit(10.)
#
#x0 = F * np.ones(N)
#x0[19] += 0.011
#scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, x0)
#o2 = scheme.orbit(10.)
#
#diff = o2 - o1
#errors = [math.sqrt(np.inner(item, item)) for item in diff]
#lerrors = [math.log(item) for item in errors]
#fig = plt.figure()
#plt.plot(t,lerrors)
#grad=(lerrors[200]-lerrors[0])/(t[200]-t[100])
#print(grad)
#
#plt.show()
#
#x1errors = [item[1] for item in diff]
#plt.plot(t,x1errors)
#plt.show()
#
#plt.plot(t,[item[2] for item in o1])
#plt.plot(t,[item[2] for item in o2])
#plt.show()
#
#
##%%
#
#def avg(a, samplesize, timesize, size):
#    b = np.zeros((timesize, size))
#    for t in range(timesize):
#        for i in range(size):
#            sum = 0
#            for j in range(samplesize):
#                sum += a[j][t][i]
#            b[t][i] = sum/samplesize
#    return b            
#
#N = 36
#F = 8
#
#lorenz = Lorenz96(N, F)
#
#t = np.arange(0.0, 10., 0.01)
#o = []
#
#x0 = F * np.ones(N)
#x0[19] += 0.01
#scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, x0)
#o.append(scheme.orbit(10.))
#
#
#randindex = np.random.randint(36, size=10)
#randaugment = 0.001 * np.random.rand(1,10)[0]
#
#for i in range(10):
#    x0 = F * np.ones(N)
#    x0[19] += 0.01
#    x0[randindex[i]] += randaugment[i]
#    scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, x0)
#    o.append(scheme.orbit(10.))
#
#ansemble = avg(o, 10, 1000, N)
#print (ansemble)
#
#fig = plt.figure()
#plt.plot(t, [item[0] for item in o[0]], label="original")
#plt.plot(t, [item[0] for item in ansemble], label="ansemble")
#plt.legend()
#plt.show()
#
#for j in range(N):    
#    fig = plt.figure()
#    for i in range(11):
#        plt.plot(t, [item[j] for item in o[i]])
#    plt.show()
#
#
##%%
#N = 36
#F = 8
#
#lorenz = Lorenz96(N, F)
#
#t = np.arange(0.0, 100., 0.01)
#
#x0 = F * np.ones(N)
#x0[19] += 0.01
#scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, x0)
#gr = scheme.orbit_gradient(100.)
#
#log_gr_norm = 0
#for i in range(len(t)):
#    log_gr_norm += math.log(abs(gr[i][19]))
#log_gr_norm /= len(t)
#
#print (log_gr_norm)
#
#
###%%
##from scipy.optimize import minimize
##
##N = 4
##F = 8
##T = 1.
##
##lorenz = Lorenz96(N, F)
##
##t = np.arange(0.0, T, 0.01)
##
##xb = F*np.ones(N)
##B = np.zeros((N,N), float)
##np.fill_diagonal(B, 1)
##R = np.zeros((N,N), float)
##np.fill_diagonal(R, 1)
##
##
##x0 = F * np.ones(N)
##x0[3] += 0.01
##scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, x0)
##y = scheme.orbit(T)
##
##
##fig = plt.figure()
##plt.plot(t, [item[0] for item in y], label="answer")
##plt.legend()
##plt.show()
##
##randindex = np.random.randint(N, size=10)
##randaugment = 0.001 * np.random.rand(1,10)[0]
##
##for sample in range(1):    
##    fig = plt.figure()
##    plt.plot(t, [item[0] for item in y], label="answer")
##    
##    
##    itera = 0
##    def cost_func(xzero):
##    #    print(xzero)
##        scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, xzero)
##        x = scheme.orbit(T)
##        plt.plot(t, [item[0] for item in x], label="iter: " + str(iter))
##        cost=0
##    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
##        for i in range(len(t)):
##            cost += (x[i] - y[i]).dot(np.linalg.inv(R)) @ (x[i] - y[i])
##        print (cost)
##        return 0.001*cost
##    
##    def gradient_from_xzero(xzero):
##        scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, xzero)
##        orb = scheme.orbit(T)
##        gr = 0.001 * lorenz.gradient_discrete(R, 0.01, 100, orb, y)
##        print("gr", gr)
##        return gr
##    
##    x_opt = F*np.ones(N)
##    x_opt[3] += 0.01
##    x_opt[randindex[sample]] += randaugment[sample]
##    
##    la = np.ones(N)
##    m = lorenz.adjoint_matrix(1,y)
##    print ("adjoint matrix", m)
##    print ("grad", lorenz.gradient_adjoint(1, la, y, y, 0.01, 100))
##    
##    res = minimize(cost_func, x_opt, jac=gradient_from_xzero, method='L-BFGS-B', options={'gtol': 1e-6, 'disp': True})
##    print ("x_opt: ", res.x)
##    plt.legend()
##    plt.show()
##    
#
##%%
#from scipy.optimize import minimize
#
#N = 4
#F = 8
#T = 1.
#
#lorenz = Lorenz96(N, F)
#
#t = np.arange(0.0, T, 0.01)
#
#xb = F*np.ones(N)
#B = np.zeros((N,N), float)
#np.fill_diagonal(B, 1)
#R = np.zeros((N,N), float)
#np.fill_diagonal(R, 1)
#
#
#x0 = F * np.ones(N)
#x0[3] += 0.01
#scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, x0)
#y = scheme.orbit(T)
#
#la = F * np.zeros(N)
#shceme = RungeKutta4()
#
#
#fig = plt.figure()
#plt.plot(t, [item[0] for item in y], label="answer")
#plt.legend()
#plt.show()
#
#randindex = np.random.randint(N, size=10)
#randaugment = 0.001 * np.random.rand(1,10)[0]
#
#for sample in range(1):    
#    fig = plt.figure()
#    plt.plot(t, [item[0] for item in y], label="answer")
#    
#    
#    itera = 0
#    def cost_func(xzero):
#    #    print(xzero)
#        scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, xzero)
#        x = scheme.orbit(T)
#        plt.plot(t, [item[0] for item in x], label="iter: " + str(iter))
#        cost=0
#    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
#        for i in range(len(t)):
#            cost += (x[i] - y[i]).dot(np.linalg.inv(R)) @ (x[i] - y[i])
#        print (cost)
#        return 0.001*cost
#    
#    def gradient_from_xzero(xzero):
#        scheme = RungeKutta4(lorenz.gradient, N, 0.01, 0, xzero)
#        orb = scheme.orbit(T)
#        gr = 0.001 * lorenz.gradient_discrete(R, 0.01, 100, orb, y)
#        print("gr", gr)
#        return gr
#    
#    x_opt = F*np.ones(N)
#    x_opt[3] += 0.01
#    x_opt[randindex[sample]] += randaugment[sample]
#    
#    la = np.ones(N)
#    m = lorenz.adjoint_matrix(1,y)
#    print ("adjoint matrix", m)
#    print ("grad", lorenz.gradient_adjoint(1, la, y, y, 0.01, 100))
#    
#    res = minimize(cost_func, x_opt, jac=gradient_from_xzero, method='L-BFGS-B', options={'gtol': 1e-6, 'disp': True})
#    print ("x_opt: ", res.x)
#    plt.legend()
#    plt.show()
#    
##%%
#
#from scipy.optimize import minimize
#
#N = 7
#F = 8
#T = 0.01
#dt = 0.01
#steps = int(T/dt) + 1
#
#lorenz = Lorenz96(N, F)
#
#t = np.arange(0., T + dt, dt)
#x = np.zeros((steps, N))
#y = np.zeros((steps, N))
#
#x[0] = F * np.ones(N)
#x[0][0] += 2
#x[0][1] += 1
#x[0][2] += 2
#
#scheme = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, x, y)
#
#tob, obs = scheme.true_observed(1)
#scheme.y = np.copy(obs)
##scheme.y = np.copy(tob)
#print("y")
#plot_orbit(scheme.y)
#
#x_opt = F * np.ones(N)
##x_opt[0] += 2
##x_opt[1] += 1
##x_opt[2] += 2
#
#x_opt[0] += 1
#x_opt[1] += 2
#x_opt[2] += 1
#
#scheme.cost(x_opt)
#print("x")
#plot_orbit(scheme.x)
#
#compare_orbit(tob, scheme.x, 'true_orbit', 'initial value')
##compare_orbit3(tob, scheme.y, scheme.x, 'initial value')
#
#gr_anal = scheme.gradient_from_x0(x_opt)
#print (gr_anal)
#gr_num = scheme.numerical_gradient_from_x0(x_opt, 0.001)
#print (gr_num)
#
##scheme.x[0] = F * np.ones(N)
##scheme.x[0][0] += 0.01
##
##objective_func_gradient = scheme.gradient()
##
##print (objective_func_gradient)
##
##plt.plot(t,[item[0] for item in y])
##plt.show()
#
#
##print (scheme.gradient_from_x0(x_opt))
#
#for rep in range(1):
#    # res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', options={'gtol': 1e-6, 'disp': True})
#    res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B')
#    print (res)
#    
#    #scheme.cost(res.x)
#    #plot_orbit(scheme.x)
#    
#    x2 = np.zeros((steps, N))
#    x2[0] = np.copy(res.x)
#    scheme2 = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, x2, y)
#    ans = np.copy(scheme2.orbit())
#    plot_orbit(ans)
#    
#    for j in range(N):
#        plt.plot(t,[item[j] for item in scheme.y], label='true')
#        plt.plot(t,[item[j] for item in ans], label='assimilated')
#        plt.legend()
#        plt.show()
#    
#    compare_orbit(tob, ans, 'true_orbit', 'assimilated')
#    
#    
##%%
#    
#x0 = F * np.ones(N)
#x0[0] += 2
#x0[1] += 1
#x0[2] += 2
#
#dx0 = np.zeros(N)
#for i in range(N):
#    dx0[i] = 0.0001
#
#x0_plus_dx0 = np.copy(x0 + dx0)
#
#rk4 = RungeKutta4(lorenz.gradient, N, dt, t, x0_plus_dx0)
#x1_plus_dx1 = rk4.nextstep()
#
#m_dx0 = ((x1_plus_dx1 - (x0 + dx0)) - (tob[1] - tob[0]))/dt
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
#
##%%
#lm = Linear(N, F)
#    
#x0 = F * np.ones(N)
#x0[0] += 2
#x0[1] += 1
#x0[2] += 2
#
#dx0 = np.zeros(N)
#for i in range(N):
#    dx0[i] = 0.001
#
#x0_plus_dx0 = np.copy(x0 + dx0)
#
#rk4 = RungeKutta4(lm.gradient, N, dt, t, x0)
#x1 = rk4.nextstep()
#
#rk4_delta = RungeKutta4(lm.gradient, N, dt, t, x0_plus_dx0)
#x1_plus_dx1 = rk4_delta.nextstep()
#
#m_dx0 = ((x1_plus_dx1 - (x0 + dx0)) - (x1 - x0))/dt
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

#%%

from scipy.optimize import minimize

N = 2
T = 0.1
dt = 0.01
steps = int(T/dt) + 1

lm = Linear(N)

t = np.arange(0., T + dt, dt)
x = np.zeros((steps, N))
y = np.zeros((steps, N))

x[0] = np.zeros(N)
x[0][0] += 3
x[0][1] += 4

scheme = Adjoint(lm.gradient, lm.gradient_adjoint, N, T, dt, x, y)

tob, obs = scheme.true_observed(1)
scheme.y = np.copy(obs)
#scheme.y = np.copy(tob)
print("y")
plot_orbit(scheme.y)

x_opt = np.zeros(N)
x_opt[0] += -4
x_opt[1] += -3

scheme.cost(x_opt)
print("x")
plot_orbit(scheme.x)

compare_orbit(tob, scheme.x, 'true_orbit', 'initial value')
#compare_orbit3(tob, scheme.y, scheme.x, 'initial value')

gr_anal = scheme.gradient_from_x0(x_opt)
print (gr_anal)
gr_num = scheme.numerical_gradient_from_x0(x_opt, 0.001)
print (gr_num)
print ((gr_anal - gr_num)/gr_num)

print('before assimilation')
for j in range(N):
    plt.plot(t,[item[j] for item in scheme.y], label='true')
    plt.plot(t,[item[j] for item in scheme.x], label='initial_state')
    plt.legend()
    plt.show()


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
    res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B')
    print (res)
    
    #scheme.cost(res.x)
    #plot_orbit(scheme.x)
    
    x2 = np.zeros((steps, N))
    x2[0] = np.copy(res.x)
    scheme2 = Adjoint(lm.gradient, lm.gradient_adjoint, N, T, dt, x2, y)
    ans = np.copy(scheme2.orbit())
    plot_orbit(ans)
    
    for j in range(N):
        plt.plot(t,[item[j] for item in scheme.y], label='true')
        plt.plot(t,[item[j] for item in ans], label='assimilated')
        plt.legend()
        plt.show()
    
    compare_orbit(tob, ans, 'true_orbit', 'assimilated')
 

#%%
#compare_orbit3(tob, scheme.y, ans, 'assimilated')
    
#%%
fig = plt.figure()
plt.plot(t, [np.linalg.norm(ans[i] - tob[i])/math.sqrt(N) for i in range(len(t))], label='x norm')
plt.xlabel('day')
plt.ylabel('RMSE')
plt.yscale('linear')
plt.legend()
plt.show()

print ("RMSE: ", np.mean([np.linalg.norm(ans[i] - tob[i])/math.sqrt(N) for i in range(int(0.7*len(t)),len(t))]))
