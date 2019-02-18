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


class RK4Matrix:
    def __init__(self, M, N, dt):
        self.M = M
        self.N = N
        self.dt = dt
        
    def nextstep(self, gradient, t, x, pa):
        k1 = handler(gradient, t, x, pa)
        k2 = handler(gradient, t + self.dt/2, x + k1*self.dt/2, pa)
        k3 = handler(gradient, t + self.dt/2, x + k2*self.dt/2, pa)
        k4 = handler(gradient, t + self.dt  , x + k3*self.dt, pa)
        return x + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6            

class Var3Dstep:
    def __init__(self, model, intmodelx, intmodelP, N, dt, R, H):
        self.model = model
        self.intmodelx = intmodelx
        self.intmodelP = intmodelP
        self.N = N
        self.dt = dt
        self.R = R
        self.H = H

    def predict(self, xa, Pa, t):
        xf = self.intmodelx.nextstep(self.model.gradient, t, xa)
        Pf = Pa # P is fixed to B
        return xf, Pf
        
    def update(self, xf, Pf, y):
        innov = self.H @ (y - xf)
        InnovCov = self.H @ Pf @ np.transpose(self.H) + self.R
        Gain = Pf @ np.transpose(self.H) @ np.linalg.inv(InnovCov)
        xa = xf + Gain @ innov
        Pa = Pf # P is fixed to B
        return xa, Pa
        
    def step(self, xa, Pa, y, t, it):
        for i in range(it):
            xa, Pa = self.predict(xa, Pa, t) # LHS means xf, Pf
        return self.update(xa, Pa, y) # RHS means xf, Pf

class KF:
    def __init__(self, kfstep, T, dt, xf, Pf, xa, Pa, y, it):
        self.kfstep = kfstep
        self.xf = xf
        self.Pf = Pf
        self.xa = xa
        self.Pa = Pa
        self.y = y
        self.it = it
        self.steps = int(T/dt/it)
        
    def filtering(self):
        t = np.arange(0., T, dt)
        k = 0
        self.xa[0], self.Pa[0] = self.kfstep.update(self.xf[0], self.Pf[0], y[0])
        for i in range(1, self.steps):
            self.xf[k+1], self.Pf[k+1] = self.kfstep.predict(self.xa[i-1], self.Pa[i-1], t[k])
            k += 1
            for j in range(1, self.it):
                self.xf[k+1], self.Pf[k+1] = self.kfstep.predict(self.xf[k], self.Pf[k], t[k])
                k += 1
            self.xa[i], self.Pa[i] = self.kfstep.update(self.xf[k], self.Pf[k], y[i])
        return self.xf, self.Pf, self.xa, self.Pa


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
M = 40

lorenz = Lorenz96(N, F)
rk4 = RK4(N, dt)
rk4matrix = RK4Matrix(N, N, dt)

t = np.arange(0., T-dt, dt)

t_day = np.copy(t)/0.2
t_day_every6h = []
for i in range(0,minute_steps,it):
    t_day_every6h.append(t_day[i])

xf = np.zeros((minute_steps, N))
xf[0] = np.loadtxt("data/assimilation_xzero.2.dat")

xa = np.zeros((steps, N))

Pf = np.zeros((minute_steps, N, N))
np.fill_diagonal(Pf[0], 1)

Pa = np.zeros((steps, N, N))

R = np.zeros((M, M))
np.fill_diagonal(R, 1)

H = np.zeros((N, N))

var3d_step = Var3Dstep(lorenz, rk4, rk4matrix, N, dt, R, H)

y = np.loadtxt("data/observed." + str(it) + ".1.dat")
var3d = KF(var3d_step, T, dt, xf, Pf, xa, Pa, y, it)

#%%
samples = 100
h24 = 0.2
h24_minute_steps = int(h24/dt)
h24_steps = int(h24_minute_steps/it)
xa_index = np.random.choice(steps-48*h24_minute_steps, samples, replace=False)

eps = np.zeros((samples, N))
count2 = 0
for i in xa_index:
    xa0 = np.zeros((2*h24_minute_steps, N))
    xa0[0] = np.copy(xa[i])
    xa24 = np.zeros((h24_minute_steps, N))
    xa24[0] = np.copy(xa[i + h24_steps])
    
    for j in range(1, 2*h24_minute_steps):
        xa0[j], Pa[j] = var3d_step.predict(xa0[j-1], Pf[0], t[i + j])
    
    for j in range(1, h24_minute_steps):
        xa24[j], Pa[j] = var3d_step.predict(xa0[j-1], Pf[0], t[i + j])
    
    eps[count2] = xa0[-1] - xa24[-1]
    count2 += 1
    
#%%
alpha = 0.02
B = alpha * np.cov(eps,rowvar=False)
z
#%%
tob = np.loadtxt("data/year." + str(it) + ".1.dat")

obs = np.loadtxt("data/observed." + str(it) + ".1.dat")

compare_orbit(tob[0:minute_steps], obs[0:minute_steps], 'true orbit', 'observed')


x_opt = np.loadtxt("data/assimilation_xzero.2.dat")

x = np.zeros((minute_steps,N))
scheme = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, x, obs)

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
