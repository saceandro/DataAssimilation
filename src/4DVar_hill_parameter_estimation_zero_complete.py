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
import scipy.stats

count = 0

def handler(func, *args):
    return func(*args)

#%%
class Hill:
    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.N = 1
        self.M = 4 # x0=x, x1=a, x2=b, x3=K, x4=n

    def gradient(self, t, x):
        u = self.ctrl(t)
        
        u_n = math.pow(u, x[4])
        K_n = math.pow(x[3], x[4])

        d = np.zeros(self.N + self.M)
        d[0] = x[1] * x[0] + x[2] * u_n / (K_n + u_n)
                
        return d
        
#    def gradient(self, t, x):
#        u = self.ctrl(t)
#        d = np.zeros(self.N + self.M)
#        if (u > 0):
#            d[0] = x[1] * x[0] + x[2] / (1.0 + math.pow(x[3]/u, x[4]))
#        else:
#            d[0] = x[1] * x[0]
#        return d

#    def gradient_adjoint(self, la, t, x):
#        u = self.ctrl(t)
#
#        u_n = math.pow(u, x[4])
#        K_n = math.pow(x[3], x[4])
#        
#        d = np.zeros(self.N + self.M)
#        d[0] = x[1] * la[0]
#        d[1] = x[0] * la[0]
#        d[2] = u_n / (K_n + u_n)
#        d[3] = -x[2] * x[4] / x[3] * u_n * K_n / (u_n + K_n)**2
#        if (u > 0):
#            d[4] = x[2] * u_n * K_n * math.log(u/x[3]) / (u_n + K_n)**2
#        return d

    def gradient_adjoint(self, la, t, x):
        u = self.ctrl(t)
        
        d = np.zeros(self.N + self.M)
        d[0] = x[1] * la[0]
        d[1] = x[0] * la[0]
        
#        if (u > 1e-10):
        if(u > 0):
            try:
                l_K_over_u = math.log(x[3]/u)
                n_l_K_over_u = x[4] * l_K_over_u                
                cosh_n_l_K_over_u = math.cosh(n_l_K_over_u)
                d[2] = la[0] / (1.0 + math.exp(n_l_K_over_u))
                d[3] = - x[2] * x[4] / 2.0 / x[3] / (1.0 + cosh_n_l_K_over_u) * la[0] # else d[3] = 0. fixed
                d[4] = - x[2] * l_K_over_u / 2.0 / (1.0 + cosh_n_l_K_over_u) * la[0]  # else d[4] = 0. fixed
            except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
                d[2] = 0.
                d[3] = 0.
                d[4] = 0.
        return d


#    def gradient_adjoint(self, la, t, x):
#        u = self.ctrl(t)
#        
#        d = np.zeros(self.N + self.M)
#        d[0] = x[1] * la[0]
#        d[1] = x[0] * la[0]
#        
#        if (u > 0):
#            print('u, K:', u, x[3])
#
#            l_u_over_K = math.log(u/x[3])
#            n_l_u_over_K = x[4] * l_u_over_K
#            
#            cosh_n_l_u_over_K = math.cosh(n_l_u_over_K)
#
#            d[2] = la[0] / (1.0 + math.exp(-n_l_u_over_K))
#            d[3] = - x[2] * x[4] / 2.0 / x[3] / (1.0 + cosh_n_l_u_over_K) * la[0] # else d[3] = 0. fixed
#            d[4] = x[2] * l_u_over_K / 2.0 / (1.0 + cosh_n_l_u_over_K) * la[0]  # else d[4] = 0. fixed
#        return d

#    def gradient_adjoint(self, la, t, x):
#        u = self.ctrl(t)
##        print(u, x[3])
#        l_K_over_u = math.log(x[3]/u)
#        n_l_K_over_u = x[4] * l_K_over_u
#        cosh_n_l_K_over_u = math.cosh(n_l_K_over_u)
#        
#        d = np.zeros(self.N + self.M)
#        d[0] = x[1] * la[0]
#        d[1] = x[0] * la[0]
#        d[2] = la[0] / (1.0 + math.exp(n_l_K_over_u))
#        d[3] = - x[2] * x[4] / 2.0 / x[3] / (1.0 + cosh_n_l_K_over_u) * la[0] # fixed
#        d[4] = - x[2] * l_K_over_u / 2.0 / (1.0 + cosh_n_l_K_over_u) * la[0]  # fixed
#        return d

def ramp(t):
    return t

def impulse(t):
    return scipy.stats.norm.pdf(t, 0, 0.1)

def rect(t):
    if (t <= 5):
        return 1.
    else:
        return 0.
    
def zero(t):
    return 0.

def sin(t):
    return 1.0 + math.sin(t)

def cos(t):
    return 1.0 + math.cos(t)

def cos2t(t):
    return 1.0 + math.cos(2.*t)

class Adjoint:
    def __init__(self, dx, dla, N, T, dt, it, x, y, stddev):
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
        self.M = 4
        self.stddev = stddev
        
    def orbit(self):
        self.t = 0.
        for i in range(self.minute_steps):
            time = float(i) / self.minute_steps * self.T
            k1 = handler(self.dx, time             , self.x[i])
            k2 = handler(self.dx, time + self.dt/2., self.x[i] + k1*self.dt/2)
            k3 = handler(self.dx, time + self.dt/2., self.x[i] + k2*self.dt/2)
            k4 = handler(self.dx, time + self.dt   , self.x[i] + k3*self.dt)
            self.x[i+1] = self.x[i] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            self.t += self.dt
        return self.x
    
    def observed(self):
        self.orbit()
        for i in range(self.steps + 1):
            for j in range(self.N):
                self.x[i,j] += self.stddev * np.random.randn() # fixed
        return self.x

    def true_observed(self):
        tob = np.copy(self.orbit())
        for i in range(self.steps + 1):
            for j in range(self.N):
                self.x[i,j] += self.stddev * np.random.randn() # fixed
        return tob, self.x
    
    def gradient(self):
        self.t = self.T
        la = np.zeros((self.minute_steps + 1, self.N + self.M))
        for j in range(self.N):
            la[self.minute_steps][j] = (self.x[self.minute_steps][j] - self.y[self.steps][j]) / self.stddev**2
        for i in range(self.steps-1, -1, -1):
            for j in range(it-1, -1, -1):
                n = self.it*i + j
                time = float(n) / self.minute_steps * self.T
                if (n < self.it*self.steps):
                    p1 = handler(self.dx, time             , self.x[n])
                    p2 = handler(self.dx, time + self.dt/2., self.x[n] + p1*self.dt/2)
                    p3 = handler(self.dx, time + self.dt/2., self.x[n] + p2*self.dt/2)
                    p4 = handler(self.dx, time + self.dt   , self.x[n] + p3*self.dt)
                    gr = (p1 + 2*p2 + 2*p3 + p4)/6
    
                    k1 = handler(self.dla, la[n+1]               , time + self.dt   , self.x[n+1])
                    k2 = handler(self.dla, la[n+1] - k1*self.dt/2, time + self.dt/2., self.x[n+1] - gr*self.dt/2)
                    k3 = handler(self.dla, la[n+1] - k2*self.dt/2, time + self.dt/2., self.x[n+1] - gr*self.dt/2)
                    k4 = handler(self.dla, la[n+1] - k3*self.dt  , time             , self.x[n])
                    la[n] = la[n+1] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            for j in range(self.N):
                la[self.it*i][j] += (self.x[self.it*i][j] - self.y[i][j]) / self.stddev**2
        return la[0]

    def gradient_from_x0(self, x0):
        self.t = self.T
        self.x[0] = x0
        self.orbit()
        la = np.zeros((self.minute_steps + 1, self.N + self.M))
        for j in range(self.N):
            la[self.minute_steps][j] = (self.x[self.minute_steps][j] - self.y[self.steps][j]) / self.stddev**2
        for i in range(self.steps-1, -1, -1):
            for j in range(it-1, -1, -1):
                n = self.it*i + j
                time = float(n) / self.minute_steps * self.T
                if (n < self.it*self.steps):
                    p1 = handler(self.dx, time             , self.x[n])
                    p2 = handler(self.dx, time + self.dt/2., self.x[n] + p1*self.dt/2)
                    p3 = handler(self.dx, time + self.dt/2., self.x[n] + p2*self.dt/2)
                    p4 = handler(self.dx, time + self.dt   , self.x[n] + p3*self.dt)
                    gr = (p1 + 2*p2 + 2*p3 + p4)/6
    
                    k1 = handler(self.dla, la[n+1]               , time + self.dt   , self.x[n+1])
                    k2 = handler(self.dla, la[n+1] - k1*self.dt/2, time + self.dt/2., self.x[n+1] - gr*self.dt/2)
                    k3 = handler(self.dla, la[n+1] - k2*self.dt/2, time + self.dt/2., self.x[n+1] - gr*self.dt/2)
                    k4 = handler(self.dla, la[n+1] - k3*self.dt  , time             , self.x[n])
                    la[n] = la[n+1] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            for j in range(self.N):
                la[self.it*i][j] += (self.x[self.it*i][j] - self.y[i][j]) / self.stddev**2
        return la[0]
    
    def cost(self, x0):
        self.x[0] = x0
        self.orbit()
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps + 1):
            cost += ((self.x[self.it*i][0:self.N] - self.y[i]) @ (self.x[self.it*i][0:self.N] - self.y[i])) / self.stddev**2
        return cost/2.0 # fixed
    
    def true_cost(self):
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(self.steps + 1):
            cost += ((self.x[self.it*i][0:self.N] - self.y[i]) @ (self.x[self.it*i][0:self.N] - self.y[i])) / self.stddev**2
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
        global count, axLL, axL, axR, axRR, axRRR
        count += 1
        axLL.scatter(count, x0[self.N], c='b')
        axL.scatter(count, x0[self.N+1], c='b')
        axR.scatter(count, x0[self.N+2], c='b')
        axRR.scatter(count, x0[self.N+3], c='b')
        axRRR.scatter(count, self.cost(x0), c='b')

#%%
def plot_orbit(t, dat, lab):
    fig = plt.figure()
    plt.plot(t, dat, label=lab)
    plt.legend()
    plt.show()

def compare_orbit(t1, t2, dat1, dat2, lab1, lab2):
    fig = plt.figure()
    plt.plot(t1, dat1, label=lab1)
    plt.plot(t2, dat2, label=lab2)
    plt.legend()
    plt.show()

def compare_orbit3(t1, t2, t3, dat1, dat2, dat3, label1, label2, label3):
    fig = plt.figure()
    plt.plot(t1, dat1, label=label1)
    plt.plot(t2, dat2, label=label2)
    plt.plot(t3, dat3, label=label3)
    plt.legend()
    plt.show()

    
#%%
from scipy.optimize import minimize
np.seterr(invalid='raise')

N = 1
stddev = 0.1

ctrl = cos
pref = "/Users/konta/bitbucket/androsace/dacamp/hill/data/" + ctrl.__name__ + "/" + str(N) + "/"

M = 4
dt = 0.01

T = 10.
it = 5
minute_steps = int(T/dt)
steps = int(minute_steps/it)

hill = Hill(ctrl)

tob = np.loadtxt(pref + "true.1.dat", ndmin=2)
tob2 = np.loadtxt(pref + "true.2.dat", ndmin=2)
#covariance_tob = np.cov(np.transpose(np.asarray(tob[0:minute_steps])))
#root_mean_trace_cov_tob = np.sqrt(np.trace(covariance_tob)/N)

RMSE_natural_variability = np.mean([np.linalg.norm(tob2[i] - tob[i])/math.sqrt(N) for i in range(0,len(tob))])
#RMSE_natural_variability_T = np.mean([np.linalg.norm(tob2[i] - tob[i])/math.sqrt(N) for i in range(0,minute_steps)])

obs = np.loadtxt(pref + "observed." + str(it) + ".1.dat", ndmin=2)
#covariance_obs = np.cov(np.transpose(np.asarray(obs[0:steps])))
#root_mean_trace_cov_obs = np.sqrt(np.trace(covariance_obs)/N)

t = np.arange(0., T, dt)
t_it = np.arange(0., T, dt*it)

t_plus_1 = np.arange(0., T+dt, dt)
t_it_plus_1 = np.arange(0., T+dt, dt*it)

compare_orbit(t_plus_1, t_it_plus_1, tob[0:minute_steps+1], obs[0:steps+1], 'true orbit', 'observed')

x_opt = np.zeros(N + M)
x_opt[0:N] = tob2[np.random.randint(len(tob))]
#x_opt[0] = 0
x_opt[N] = -4.  # initial guess for a (a_true = -3.)
x_opt[N+1] = 11.  # initial guess for b (b_true = 12.)
x_opt[N+2] = 1.7  # initial guess for K (K_true = 0.3132)
x_opt[N+3] = 2.4 # initial guess for n (n_true = 1.276)

x = np.zeros((minute_steps + 1, N + M))
scheme = Adjoint(hill.gradient, hill.gradient_adjoint, N, T, dt, it, x, obs, stddev)

print("Before assimilation")
print("cost", scheme.cost(x_opt))
compare_orbit3(t_plus_1, t_plus_1, t_it_plus_1, tob[0:minute_steps+1, 0], scheme.x[0:minute_steps+1, 0], obs[0:steps+1, 0], 'true_orbit', 'initial value', 'observed')
compare_orbit(t_plus_1, t_plus_1, tob[0:minute_steps+1, 0], scheme.x[0:minute_steps+1,0], 'true_orbit', 'assimilated')

print("Analytical and numerical gradient comparison")
gr_anal = scheme.gradient_from_x0(x_opt)
print ("gr_anal", gr_anal)
gr_num = scheme.numerical_gradient_from_x0(x_opt, 0.000001)
print ("gr_num", gr_num)
if not(0 in gr_num):
    print ("relative error", (gr_anal - gr_num)/gr_num)

#%%
global axLL, axL, axR, axRR, axRRR
fig , (axLL, axL, axR, axRR, axRRR) = plt.subplots(ncols=5, figsize=(10,4), sharex=False)
bnds = ((0, None), (None, None), (None, None), (0.1, None), (0, None))
res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', bounds=bnds, callback=scheme.cbf, options={'disp': None, 'maxls': 40, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
print (res)
print ("true x0", tob[0])

compare_orbit3(t_plus_1, t_plus_1, t_it_plus_1, tob[0:minute_steps+1, 0], scheme.x[0:minute_steps+1,0], obs[0:steps+1, 0], 'true orbit', 'assimilated', 'observed')

#%%
fig = plt.figure()
plt.plot(t, [np.linalg.norm(scheme.x[i,0] - tob[i,0])/math.sqrt(N) for i in range(len(t))], label='x norm')
plt.xlabel('t')
plt.ylabel('RMSE')
#plt.yscale('symlog')
plt.legend()
plt.show()

print ("RMSE:", np.mean([np.linalg.norm(scheme.x[i,0] - tob[i,0])/math.sqrt(N) for i in range(int(len(t)*0.4),int(len(t)*0.6))]))
#print ("RootMeanTr(cov(true)):", root_mean_trace_cov_tob)
#print ("RootMeanTr(cov(obs)):", root_mean_trace_cov_obs)
print ("RMSE_natural_variability:", RMSE_natural_variability)
#print ("RMSE_natural_variability_T:", RMSE_natural_variability_T)

print('4DVar optimal cost: ', res.fun)
scheme_true = Adjoint(hill.gradient, hill.gradient_adjoint, N, T, dt, it, tob, obs, stddev)
print('true cost: ', scheme_true.true_cost())

#%%
#slight_different_x0 = np.zeros(N + M)
#slight_different_x0[0:N] = np.copy(tob[0])
#slight_different_x0[N] = 8
#slight_different_x0[N+1] = 1
#slight_different_x0[N+2] = -1
#slight_different_x0[0] += 0.01
#rk4 = RungeKutta4(lorenz.gradient, N, dt, t, slight_different_x0)
#test_T = 2.0
#slight_different_orb = rk4.orbit(test_T)
#curr_steps = int(test_T/dt)
#
#compare_orbit(tob[0:curr_steps], slight_different_orb[0:curr_steps,0:N])
