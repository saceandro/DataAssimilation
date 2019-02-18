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
    def gradient(self, x):
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

class Adjoint:
    def __init__(self, dx, dla, N, T, dt, it, x, y, q):
        self.dx = dx
        self.dla = dla
        self.N = N
        self.T = T
        self.dt = dt
        self.x = x
        self.y = y
        self.q = q
        self.it = it
        self.minute_steps = int(T/self.dt)
        self.steps = int(self.minute_steps/it)
        
    def orbit(self):
        for i in range(self.minute_steps-1):
            k1 = handler(self.dx, self.x[i]                )
            k2 = handler(self.dx, self.x[i] + k1*self.dt/2.)
            k3 = handler(self.dx, self.x[i] + k2*self.dt/2.)
            k4 = handler(self.dx, self.x[i] + k3*self.dt   )
            self.x[i+1] = self.x[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6 + self.q[i] # discrete system noise is added
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
    
    def calc_la(self, la):
        for i in range(self.steps-1, -1, -1):
            for j in range(it-1, -1, -1):
                n = self.it*i + j
                if (n < self.it*self.steps - 1):
                    p1 = handler(self.dx, self.x[n]                )
                    p2 = handler(self.dx, self.x[n] + p1*self.dt/2.)
                    p3 = handler(self.dx, self.x[n] + p2*self.dt/2.)
                    p4 = handler(self.dx, self.x[n] + p3*self.dt   )
                    gr = (p1 + 2*p2 + 2*p3 + p4)/6
    
                    k1 = handler(self.dla, la[n+1], self.x[n] + gr*self.dt)
                    k2 = handler(self.dla, la[n+1] - k1*self.dt/2, self.x[n] + gr*self.dt/2)
                    k3 = handler(self.dla, la[n+1] - k2*self.dt/2, self.x[n] + gr*self.dt/2)
                    k4 = handler(self.dla, la[n+1] - k3*self.dt, self.x[n])
                    la[n] = la[n+1] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            la[self.it*i] += self.x[self.it*i] - self.y[i]
        return la

    def gradient_from_x0(self, x0_q):
        self.x[0] = x0_q[:self.N]
        for i in range(self.minute_steps):
            self.q[i] = x0_q[self.N*(i+1) : self.N*(i+2)]
        self.orbit()
        la = np.zeros((self.minute_steps + 1, self.N))
        self.calc_la(la)
        
        gr = np.zeros(self.N * (self.minute_steps + 1))
        gr[:self.N] = la[0]
        for n in range(self.minute_steps):
            gr[self.N*(n+1) : self.N*(n+2)] = la[n+1] + 100. * self.q[n]
        return gr
    
    def cost(self, x0_q):
        self.x[0] = x0_q[:self.N]
        for i in range(self.minute_steps):
            self.q[i] = x0_q[self.N*(i+1) : self.N*(i+2)]
        self.orbit()
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for n in range(self.minute_steps):
            cost += 100. * self.q[n] @ self.q[n]
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
    
    def numerical_gradient_from_x0(self, x0_q, h):
        gr = np.zeros(self.N * (self.minute_steps + 1))
        c1 = self.cost(x0_q)
        for j in range(N * (self.minute_steps + 1)):
            xx = np.copy(x0_q)
            xx[j] += h
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
#pref = "data/" + str(N) + "/"
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

nosysnoise = np.loadtxt(pref + "nosysnoise.1.dat")

obs = np.loadtxt(pref + "observed." + str(it) + ".1.dat")

compare_orbit(tob[0:minute_steps], obs[0:steps])

t = np.arange(0., T, dt)
t_it = np.arange(0., T, dt*it)

x_q_opt = np.zeros(N * (minute_steps + 1))
x_q_opt[:N] = np.loadtxt(pref + "assimilation_xzero.2.dat")

x = np.zeros((minute_steps, N))
q = np.zeros((minute_steps, N))
scheme = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, it, x, obs, q)


print("Before assimilation")
print("cost", scheme.cost(x_q_opt))
compare_orbit3(tob[0:minute_steps], obs[0:steps], scheme.x, 'true_orbit', 'observed', 'initial value')
compare_orbit(tob[0:minute_steps], scheme.x)
#
#print("Analytical and numerical gradient comparison")
#gr_anal = scheme.gradient_from_x0(x_q_opt)
#print ("gr_anal", "%f\t"*N*(minute_steps+1) % tuple(gr_anal))
#gr_num = scheme.numerical_gradient_from_x0(x_q_opt, 0.001)
#print ("gr_num", "%f\t"*N*(minute_steps+1) % tuple(gr_num))
#print ("relative error", "%f\t"*N*(minute_steps+1) % tuple((gr_anal - gr_num)/gr_num))
#
#print("variable")
#for i in range(N):
#    print ("%e\t%e\t%e" % (gr_anal[i], gr_num[i], ((gr_anal - gr_num)/gr_num)[i]))
#print("")
#print("sysnoise")
#for i in range(N, N*(minute_steps+1)):
#    print ("%e\t%e\t%e" % (gr_anal[i], gr_num[i], ((gr_anal - gr_num)/gr_num)[i]))

#%%
fig = plt.figure()
res = minimize(scheme.cost, x_q_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', callback=scheme.cbf)
print (res)
print ("true x0", tob[0])

#%%
#for j in range(3):
for j in range(N):
    fig = plt.figure()
    plt.plot(t, tob[0:minute_steps,j], label='true orbit')
    plt.plot(t, scheme.x[0:minute_steps,j], label='assimilated')
    plt.plot(t_it, obs[0:steps,j], label='observed')
    plt.plot(t, nosysnoise[0:minute_steps,j], label='nosysnoise')
    plt.legend()
    plt.show()

compare_orbit(tob[0:minute_steps], scheme.x)
compare_orbit3(tob[0:minute_steps], scheme.x, obs[0:steps], 'true', 'assim', 'obs')

#%%
fig = plt.figure()
plt.plot(t, [np.linalg.norm(scheme.x[i] - tob[i])/math.sqrt(N) for i in range(len(t))], label='x norm')
plt.xlabel('t')
plt.ylabel('RMSE')
plt.yscale('symlog')
plt.legend()
plt.show()

print ("RMSE: ", np.mean([np.linalg.norm(scheme.x[i] - tob[i])/math.sqrt(N) for i in range(int(len(t)*0.4),int(len(t)*0.6))]))

print('4DVar optimal cost: ', res.fun)
q2 = np.zeros(N * (steps + 1))
scheme_true = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, it, tob, obs, q2)
print('true cost: ', scheme_true.true_cost())
