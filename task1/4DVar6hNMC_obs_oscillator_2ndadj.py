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
class Linear:
    def __init__(self):
        self.N = 2
        self.m = np.zeros((self.N, self.N))
        self.m[0][1] = 1.
        self.m[1][0] = -1.
        
    def gradient(self, x):
        d = np.zeros(self.N)
        d[0] = x[1]
        d[1] = -x[0]
        return d
            
    def gradient_tl(self, xi, x):
        return self.m @ xi
    
    def gradient_tl_T(self, nu, x):
        return self.m.transpose() @ nu
    
    def gradient_adjoint(self, la, x):
        return self.m.transpose() @ la
    
    def gradient_secondadj(self, nu, x, la, xi):
        return self.gradient_tl_T(nu, x) + xi
        
class Lorenz96:
    def __init__(self, N):
        self.N = N # number of variables
        self.M = 3 # number of parameters
        self.m = np.zeros((self.N + self.M, self.N + self.M))
    def gradient(self,x):
        d = np.zeros(self.N + self.M)
        d[0]        = x[self.N+1] * (x[1]   - x[self.N-2]) * x[self.N-1] + x[self.N+2] * x[0]        + x[self.N]
        d[1]        = x[self.N+1] * (x[2]   - x[self.N-1]) * x[0]        + x[self.N+2] * x[1]        + x[self.N]
        for i in range(2, self.N-1):
            d[i]    = x[self.N+1] * (x[i+1] - x[i-2])      * x[i-1]      + x[self.N+2] * x[i]        + x[self.N]
        d[self.N-1] = x[self.N+1] * (x[0]   - x[self.N-3]) * x[self.N-2] + x[self.N+2] * x[self.N-1] + x[self.N]
        return d

    def tl(self, x):
        self.m.fill(0.)
        for i in range(self.N):
            for j in range(self.N):
                if (((i-1) % self.N) == j):
                    self.m[i][j] += x[self.N+1] * (x[(i+1) % self.N] - x[(i-2) % self.N])
                if (((i+1) % self.N) == j):
                    self.m[i][j] += x[self.N+1] * x[(i-1) % self.N]
                if (((i-2) % self.N) == j):
                    self.m[i][j] -= x[self.N+1] * x[(i-1) % self.N]
                if ((i     % self.N) == j):
                    self.m[i][j] += x[self.N+2]
            self.m[i][N] = 1
            self.m[i][N+1] = (x[(i+1) % self.N] - x[(i-2) % self.N]) * x[(i-1) % self.N]
            self.m[i][N+2] = x[i]

    def gradient_tl(self, xi, x):
        self.tl(x)
        return self.m @ xi
        
    def gradient_tl_T(self, nu, x):
        self.tl(x)
        return self.m.transpose() @ nu
    
    def gradient_adjoint(self, la, x):
        # same as gradient_tl_T
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
                    mt[j][i] += x[self.N+2]
            mt[N][i] = 1
            mt[N+1][i] = (x[(i+1) % self.N] - x[(i-2) % self.N]) * x[(i-1) % self.N]
            mt[N+2][i] = x[i]
        gr = mt @ la
        return gr
    
    def gradient_secondadj(self, nu, x, la, xi):
        self.tl(x)
        gr = np.zeros(self.N + self.M)
        for i in range(self.N):
            gr[i] = (x[(i+1) % self.N] - x[(i-2) % self.N]) * (xi[(i-1) % self.N] * la[self.N+1] + xi[self.N+1] * la[(i-1) % self.N])\
            + x[self.N+1] * (xi[(i-1) % self.N] * (la[(i+1) % self.N] - la[(i-2) % self.N]) + la[(i-1) % self.N] * (xi[(i+1) % self.N] - xi[(i-2) % self.N]))\
            + x[(i-1) % self.N] * ((xi[(i+1) % self.N] - xi[(i-2) % self.N]) * la[self.N+1] + (la[(i+1) % self.N] - la[(i-2) % self.N]) * xi[self.N+1])\
            + la[self.N+2] * xi[i] + xi[self.N+2] * la[i]
        return gr + self.gradient_tl_T(nu, x)

class Adjoint:
    def __init__(self, dx, dla, dxi, dnu, N, T, dt, it, x, y, la, xi, nu):
        self.dx = dx
        self.dla = dla
        self.dxi = dxi
        self.dnu = dnu
        self.N = N
        self.T = T
        self.dt = dt
        self.x = x
        self.y = y
        self.la = la
        self.xi = xi
        self.nu = nu
        self.it = it
        self.minute_steps = int(T/self.dt)
        self.steps = int(self.minute_steps/it)
        self.M = 0
        self.xgr = np.zeros((self.minute_steps, self.N + self.M))
        self.lagr = np.zeros((self.minute_steps, self.N + self.M))
        self.xigr = np.zeros((self.minute_steps, self.N + self.M))
        
    def orbit(self):
        for i in range(self.minute_steps-1):
            k1 = handler(self.dx, self.x[i])
            k2 = handler(self.dx, self.x[i] + k1*self.dt/2)
            k3 = handler(self.dx, self.x[i] + k2*self.dt/2)
            k4 = handler(self.dx, self.x[i] + k3*self.dt)
            self.xgr[i] = (k1 + 2*k2 + 2*k3 + k4)/6.
            self.x[i+1] = self.x[i] + self.xgr[i] * self.dt
        return self.x

    def neighboring(self):
        for i in range(self.minute_steps-1):
            k1 = handler(self.dxi, self.xi[i]                , self.x[i]                           )
            k2 = handler(self.dxi, self.xi[i] + k1*self.dt/2., self.x[i] + self.xgr[i] * self.dt/2.)
            k3 = handler(self.dxi, self.xi[i] + k2*self.dt/2., self.x[i] + self.xgr[i] * self.dt/2.)
            k4 = handler(self.dxi, self.xi[i] + k3*self.dt   , self.x[i+1])
            self.xigr[i] = (k1 + 2*k2 + 2*k3 + k4)/6.
            self.xi[i+1] = self.xi[i] + self.xigr[i] * self.dt
        return self.xi
    
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
    
    def gradient_from_x0(self, x0):
        self.x[0] = x0
        self.orbit()
        self.la.fill(0.)
        for i in range(self.steps-1, -1, -1):
            for j in range(it-1, -1, -1):
                n = self.it*i + j
                if (n < self.it*self.steps - 1):
                    p1 = handler(self.dx, self.x[n])
                    p2 = handler(self.dx, self.x[n] + p1*self.dt/2)
                    p3 = handler(self.dx, self.x[n] + p2*self.dt/2)
                    p4 = handler(self.dx, self.x[n] + p3*self.dt)
                    gr = (p1 + 2*p2 + 2*p3 + p4)/6
    
                    k1 = handler(self.dla, self.la[n+1], self.x[n+1])
                    k2 = handler(self.dla, self.la[n+1] - k1*self.dt/2, self.x[n+1] - gr*self.dt/2)
                    k3 = handler(self.dla, self.la[n+1] - k2*self.dt/2, self.x[n+1] - gr*self.dt/2)
                    k4 = handler(self.dla, self.la[n+1] - k3*self.dt, self.x[n])
                    self.lagr[n] = (k1 + 2*k2 + 2*k3 + k4)/6.
                    self.la[n] = self.la[n+1] + self.lagr[n] * self.dt
            for j in range(self.N):
                self.la[self.it*i][j] += self.x[self.it*i][j] - self.y[i][j]
        return self.la[0]
    
    def hessian_vector_product(self, xi0):
        self.xi[0] = np.copy(xi0)
        self.neighboring()
        self.nu.fill(0.)
        for i in range(self.steps-1, -1, -1):
            for j in range(it-1, -1, -1):
                n = self.it*i + j
                if (n < self.it*self.steps - 1):
                    k1 = handler(self.dnu, self.nu[n+1]                , self.x[n+1]                         , self.la[n+1]                          , self.xi[n+1])
                    k2 = handler(self.dnu, self.nu[n+1] - k1*self.dt/2., self.x[n+1] - self.xgr[n]*self.dt/2., self.la[n+1] + self.lagr[n]*self.dt/2., self.xi[n+1] - self.xigr[n]*self.dt/2.)
                    k3 = handler(self.dnu, self.nu[n+1] - k2*self.dt/2., self.x[n+1] - self.xgr[n]*self.dt/2., self.la[n+1] + self.lagr[n]*self.dt/2., self.xi[n+1] - self.xigr[n]*self.dt/2.)
                    k4 = handler(self.dnu, self.nu[n+1] - k3*self.dt   , self.x[n]                           , self.la[n+1] + self.lagr[n]*self.dt   , self.xi[n])
                    self.nu[n] = self.nu[n+1] + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6
            for j in range(self.N):
                self.nu[self.it*i][j] += self.xi[self.it*i][j]        
        return self.nu[0]
    
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
    
    def numerical_gradient_from_x0(self, x0, h):
        gr = np.zeros(self.N + self.M)
        c1 = self.cost(x0)
        for i in range(self.N + self.M):
            xx = np.copy(x0)
            xx[i] += h
            c = self.cost(xx)
            gr[i] = (c - c1)/h
        return gr
    
    def numerical_hessian_from_x0(self, x0, h):
        hess = np.zeros((self.N + self.M, self.N + self.M))
        gr1 = np.copy(self.gradient_from_x0(x0))
#        print("gr1", gr1)
        for i in range(self.N + self.M):
            for j in range(self.N + self.M):
                xx = np.copy(x0)
                xx[j] += h
                gr2 = np.copy(self.gradient_from_x0(xx))
#                print("gr2", gr2)
#                print("(gr2 - gr1)", (gr2 - gr1))
                hess[j,i] = (gr2[i] - gr1[i])/h
#                print("")
#            print("")
        return hess
    
    def numerical_hessian_from_x0_2(self, x0, h):
        hess = np.zeros((self.N + self.M, self.N + self.M))
        gr1 = np.copy(self.numerical_gradient_from_x0(x0, h))
        for i in range(self.N + self.M):
            for j in range(self.N + self.M):
                xx = np.copy(x0)
                xx[j] += h
                gr2 = np.copy(self.numerical_gradient_from_x0(xx, h))
                hess[j,i] = (gr2[i] - gr1[i])/h
        return hess
        
    
    def cbf(self, x0):
#        global axL, axR, axRR
        global count, trace, cost_trace
        count += 1
        cos = self.cost(x0)
#        axLL.scatter(count, x0[self.N], c='b')
#        axL.scatter(count, x0[self.N+1], c='b')
#        axR.scatter(count, x0[self.N+2], c='b')
#        axRR.scatter(count, cos, c='b')
        cost_trace.append(cos)
        for i in range(self.N):
            trace[i].append(x0[i])

#%%
def plot_orbit(dat):
    fig = plt.figure()
    plt.plot(dat[:,0],dat[:,1])
    plt.show()

def compare_orbit(dat1, dat2):
    fig = plt.figure()
    plt.plot(dat1[:,0], dat1[:,1], label='true orbit')
    plt.plot(dat2[:,0], dat2[:,1], label='assimilated')
    plt.legend()
    plt.show()

def compare_orbit3(dat1, dat2, dat3, label1, label2, label3):
    fig = plt.figure()
    plt.plot(dat1[:,0], dat1[:,1], label=label1)
    plt.plot(dat2[:,0], dat2[:,1], label=label2)
    plt.plot(dat3[:,0], dat3[:,1], label=label3)
    plt.legend()
    plt.show()
    
#%%
from scipy.optimize import minimize

N = 2
pref = "/Users/konta/bitbucket/androsace/dacamp/task1/oscillator_sysnoise_data/" + str(N) + "/"

M = 0
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

lorenz = Linear()

tob = np.loadtxt(pref + "year.1.dat")
tob2 = np.loadtxt(pref + "year.2.dat")
#covariance_tob = np.cov(np.transpose(np.asarray(tob[0:minute_steps])))
#root_mean_trace_cov_tob = np.sqrt(np.trace(covariance_tob)/N)

RMSE_natural_variability = np.mean([np.linalg.norm(tob2[i] - tob[i])/math.sqrt(N) for i in range(0,len(tob))])
#RMSE_natural_variability_T = np.mean([np.linalg.norm(tob2[i] - tob[i])/math.sqrt(N) for i in range(0,minute_steps)])

obs = np.loadtxt(pref + "observed." + str(it) + ".1.dat")
#covariance_obs = np.cov(np.transpose(np.asarray(obs[0:steps])))
#root_mean_trace_cov_obs = np.sqrt(np.trace(covariance_obs)/N)

compare_orbit(tob[0:minute_steps], obs[0:steps])

t = np.arange(0., T, dt)
t_it = np.arange(0., T, dt*it)

x_opt = np.zeros(N + M)
x_opt[0:N] = np.loadtxt(pref + "year.2.dat")[np.random.randint(len(tob))]
#x_opt[0:N] = np.copy(tob[0])

x = np.zeros((minute_steps, N + M))
la = np.zeros((minute_steps, N + M))
xi = np.zeros((minute_steps, N + M))
nu = np.zeros((minute_steps, N + M))
scheme = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, lorenz.gradient_tl, lorenz.gradient_secondadj, N, T, dt, it, x, obs, la, xi, nu)

print("Before assimilation")
initial_cost = scheme.cost(x_opt)
print("cost", initial_cost)
compare_orbit3(tob[0:minute_steps], obs[0:steps], scheme.x[:,0:N], 'true_orbit', 'observed', 'initial value')
compare_orbit(tob[0:minute_steps], scheme.x[:,0:N])

print("Analytical and numerical gradient comparison")
gr_anal = scheme.gradient_from_x0(x_opt)
print ("gr_anal", gr_anal)
gr_num = scheme.numerical_gradient_from_x0(x_opt, 0.00001)
print ("gr_num", gr_num)
print ("relative error", (gr_anal - gr_num)/gr_num)

#%%
#fig , (axLL, axL, axR, axRR) = plt.subplots(ncols=4, figsize=(10,4), sharex=False)
cost_trace = [initial_cost]
trace=[]
for i in range(N):
    trace.append([obs[0,i]])
res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', callback=scheme.cbf)

#%%
print (res)
print ("true x0", tob[0])

estimated_x0 = np.copy(res.x)

#%%    

#for j in range(3):
for j in range(N):
    fig = plt.figure()
    plt.plot(t, tob[0:minute_steps,j], label='true orbit')
    plt.plot(t, scheme.x[0:minute_steps,j], label='assimilated')
    plt.plot(t_it, obs[0:steps, j], label='observed')
    plt.legend()
    plt.show()

compare_orbit3(tob[0:minute_steps], scheme.x[:,0:N], obs[0:steps], 'true orbit', 'assimilated', 'observed')


#%%
fig = plt.figure()
plt.plot(t, [np.linalg.norm(scheme.x[i,0:N] - tob[i])/math.sqrt(N) for i in range(len(t))], label='x norm')
plt.xlabel('t')
plt.ylabel('RMSE')
plt.yscale('symlog')
plt.legend()
plt.show()

print ("RMSE:", np.mean([np.linalg.norm(scheme.x[i,0:N] - tob[i])/math.sqrt(N) for i in range(int(len(t)*0.4),int(len(t)*0.6))]))
#print ("RootMeanTr(cov(true)):", root_mean_trace_cov_tob)
#print ("RootMeanTr(cov(obs)):", root_mean_trace_cov_obs)
print ("RMSE_natural_variability:", RMSE_natural_variability)
#print ("RMSE_natural_variability_T:", RMSE_natural_variability_T)

print('4DVar optimal cost: ', res.fun)
#scheme_true = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, N, T, dt, it, tob, obs)
#print('true cost: ', scheme_true.true_cost())


#%%
hessian_T = np.zeros((N + M, N + M))
xi0 = np.zeros(N + M)
for i in range(N + M):
    xi0.fill(0.)
    xi0[i] = 1.
    hessian_T[i] = np.copy(scheme.hessian_vector_product(xi0))

hessian = hessian_T.transpose()
hessian_inv = np.linalg.inv(hessian)
print("hessian", hessian)
#print("hessian_inverse", hessian_inv)
variance = np.diag(hessian_inv)
#print("variance", variance)

hess_num = scheme.numerical_hessian_from_x0(estimated_x0, 0.001)
print ("hessian_num", hess_num)

hess_num2 = scheme.numerical_hessian_from_x0_2(estimated_x0, 0.001)
print ("hessian_num2", hess_num2)


rel_error = (hessian - hess_num)/ hessian
print ("relative error", (hessian - hess_num)/hessian)

rel_error_2 = (hessian - hess_num2)/ hessian
print ("relative error", rel_error_2)

hess_num2_inv = np.linalg.inv(hess_num2)

#%%
for j in range(N):
    plt.plot([obs[0,j] for i in range(len(trace[j]))], 'r')
    plt.plot(trace[j], 'b')
    plt.errorbar(len(trace[j])-1, trace[j][-1], yerr=hessian_inv[j,j], fmt='b')
    plt.legend()
    plt.xlabel('minimizer iteration')
    plt.ylabel('x_{' + str(j) + '}')
    plt.show()

true_param = np.array([8., 1., -1.])

#fig = plt.figure()
#plt.plot([true_param[0] for i in range(len(f_trace))], 'r')
#plt.plot(f_trace, 'b')
#plt.errorbar(len(f_trace)-1, f_trace[-1], yerr=hessian_inv[N,N], fmt='b')
#plt.legend()
#plt.xlabel('minimizer iteration')
#plt.ylabel('F')
#plt.show()
#
#fig = plt.figure()
#plt.plot([true_param[1] for i in range(len(a_trace))], 'r')
#plt.plot(a_trace, 'b')
#plt.errorbar(len(a_trace)-1, a_trace[-1], yerr=hessian_inv[N+1,N+1], fmt='b')
#plt.legend()
#plt.xlabel('minimizer iteration')
#plt.ylabel('a')
#plt.show()
#
#fig = plt.figure()
#plt.plot([true_param[2] for i in range(len(b_trace))], 'r')
#plt.plot(b_trace, 'b')
#plt.errorbar(len(b_trace)-1, b_trace[-1], yerr=hessian_inv[N+2,N+2], fmt='b')
#plt.legend()
#plt.xlabel('minimizer iteration')
#plt.ylabel('b')
#plt.show()

fig = plt.figure()
plt.plot(cost_trace, 'b')
plt.xlabel('minimizer iteration')
plt.ylabel('cost')
plt.show()

print("-----------------------------------------------------------------")
#%%
for j in range(N):
    plt.plot([obs[0,j] for i in range(len(trace[j]))], 'r')
    plt.plot(trace[j], 'b')
    plt.errorbar(len(trace[j])-1, trace[j][-1], yerr=hess_num2_inv[j,j], fmt='b')
    plt.legend()
    plt.xlabel('minimizer iteration')
    plt.ylabel('x_{' + str(j) + '}')
    plt.show()

true_param = np.array([8., 1., -1.])

#fig = plt.figure()
#plt.plot([true_param[0] for i in range(len(f_trace))], 'r')
#plt.plot(f_trace, 'b')
#plt.errorbar(len(f_trace)-1, f_trace[-1], yerr=hess_num2_inv[N,N], fmt='b')
#plt.legend()
#plt.xlabel('minimizer iteration')
#plt.ylabel('F')
#plt.show()
#
#fig = plt.figure()
#plt.plot([true_param[1] for i in range(len(a_trace))], 'r')
#plt.plot(a_trace, 'b')
#plt.errorbar(len(a_trace)-1, a_trace[-1], yerr=hess_num2_inv[N+1,N+1], fmt='b')
#plt.legend()
#plt.xlabel('minimizer iteration')
#plt.ylabel('a')
#plt.show()
#
#fig = plt.figure()
#plt.plot([true_param[2] for i in range(len(b_trace))], 'r')
#plt.plot(b_trace, 'b')
#plt.errorbar(len(b_trace)-1, b_trace[-1], yerr=hess_num2_inv[N+2,N+2], fmt='b')
#plt.legend()
#plt.xlabel('minimizer iteration')
#plt.ylabel('b')
#plt.show()
#
#fig = plt.figure()
#plt.plot(cost_trace, 'b')
#plt.xlabel('minimizer iteration')
#plt.ylabel('cost')
#plt.show()


#param_rmse = np.linalg.norm(res.x[N:N+M] - true_param)/math.sqrt(M)
#print("True_Param:", true_param)
#print("Estimated_Param:", res.x[N:N+M])
#print ("RMSE_Param:", param_rmse)


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
