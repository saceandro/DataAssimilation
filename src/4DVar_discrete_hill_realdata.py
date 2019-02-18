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

def handler(func, *args):
    return func(*args)

#%%
class Hill:
    def __init__(self):
        self.N = 1
        self.M = 4 # x0=x, x1=a, x2=b, x3=K, x4=n

    def gradient(self, dt, u, x, x_next):
        u_n = math.pow(u, x[4])
        K_n = math.pow(x[3], x[4])
        
        x_next[0] = (1. + dt * x[1]) * x[0] + dt * x[2] * u_n / (K_n + u_n)
        x_next[1] = x[1]
        x_next[2] = x[2]
        x_next[3] = x[3]
        x_next[4] = x[4]
        
        return x_next
        
    def gradient_adjoint(self, dt, u, la, x):
        d = np.zeros(self.N + self.M)
        
        d[0] = x[1]
        d[1] = x[0]
        
        try:
            K_over_u = x[3]/u
            l_K_over_u = math.log(K_over_u)
            cosh_n_l_K_over_u_plus_1_inv = 1. / (1. + math.cosh(x[4] * l_K_over_u))
            d[2] = 1. / (1. + math.pow(K_over_u, x[4]))
            d[3] = - x[2] * x[4] / x[3] / 2. * cosh_n_l_K_over_u_plus_1_inv
            d[4] = - x[2] / 2. * l_K_over_u * cosh_n_l_K_over_u_plus_1_inv
        except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError) as e:
#            print('gradient_adjoint exception:', e)
            d[2] = 0
            d[3] = 0
            d[4] = 0
        
        return la + dt * la[0] * d

    def gradient_neighboring(self, dt, u, xi, x):
        d = np.zeros(self.N + self.M)
        
        try:
            K_over_u = x[3]/u
            l_K_over_u = math.log(K_over_u)
            cosh_n_l_K_over_u_plus_1_inv = 1. / (1. + math.cosh(x[4] * l_K_over_u))

            d[0] = xi[2] / (1. + math.pow(K_over_u, x[4]))\
                 - xi[3] * x[2] * x[4] / x[3] / 2. * cosh_n_l_K_over_u_plus_1_inv\
                 - xi[4] * x[2] / 2. * l_K_over_u * cosh_n_l_K_over_u_plus_1_inv
        except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError) as e:
#            print('gradient_neighboring exception:', e)
            d[0] = 0
        
        d[0] += x[1] * xi[0] + x[0] * xi[1]
        
        return xi + dt * d
        
    def gradient_secondadj(self, dt, u, nu, x, la, xi):
        d = np.zeros(self.N + self.M)
        
        d[0] = x[1] * nu[0] + xi[1] * la[0]
        d[1] = x[0] * nu[0] + xi[0] * la[0]
        
        try:
            K_over_u = x[3]/u
            l_K_over_u = math.log(K_over_u)
            n_l_K_over_u = x[4] * l_K_over_u                
            sinh_n_l_K_over_u = math.sinh(n_l_K_over_u)
            cosh_n_l_K_over_u_plus_1_inv = 1. / (1. + math.cosh(n_l_K_over_u))
            
            term2_3 = -x[4] / x[3] / 2. * cosh_n_l_K_over_u_plus_1_inv
            term2_4 = -l_K_over_u / 2. * cosh_n_l_K_over_u_plus_1_inv
            term3_3 = x[2] * x[4] / x[3]**2 / 2. * cosh_n_l_K_over_u_plus_1_inv * ( 1. + x[4] * sinh_n_l_K_over_u * cosh_n_l_K_over_u_plus_1_inv )
            term3_4 = -x[2] / x[3] / 2. * cosh_n_l_K_over_u_plus_1_inv * ( 1. - x[4] * l_K_over_u * sinh_n_l_K_over_u * cosh_n_l_K_over_u_plus_1_inv )
            term4_4 = x[2] / 2. * l_K_over_u**2 * sinh_n_l_K_over_u * cosh_n_l_K_over_u_plus_1_inv**2
            
            d[2] = nu[0] / (1. + math.pow(K_over_u, x[4]))\
                 + (xi[3] * term2_3 + xi[4] * term2_4) * la[0]
            d[3] = x[2] * term2_3 * nu[0]\
                 + (xi[2] * term2_3 + xi[3] * term3_3 + xi[4] * term3_4) * la[0]
            d[4] = x[2] * term2_4 * nu[0]\
                 + (xi[2] * term2_4 + xi[3] * term3_4 + xi[4] * term4_4) * la[0]
        except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError) as e:
#            print('gradient_secondadj exception', e)
            d[2] = 0
            d[3] = 0
            d[4] = 0
            
        return nu + dt * d

#def ramp(t):
#    return t
#
#def impulse(t):
#    return scipy.stats.norm.pdf(t, 0, 0.1)
#
#def rect(t):
#    if (t <= 5):
#        return 1.
#    else:
#        return 0.
#    
#def zero(t):
#    return 0.
#
#def sin(t):
#    return 1.0 + math.sin(t)
#
#def cos(t):
#    return 1.0 + math.cos(t)

#%%
class Adjoint:
    def __init__(self, dx, dla, dxi, dnu, N, dt, obs_variance, t, u, x, y, la, xi, nu):
        self.dx = dx
        self.dla = dla
        self.dxi = dxi
        self.dnu = dnu
        self.N = N
        self.M = 4
#        self.T = T
        self.t = t
        self.dt = dt
        self.u = u
        self.x = x
        self.y = y
        self.la = la
        self.xi = xi
        self.nu = nu
#        self.it = it
#        self.minute_steps = int(T/self.dt)
#        self.steps = int(self.minute_steps/it)
        self.obs_variance = obs_variance
        
    def orbit(self):
        for i in range(len(self.t)-1):
            handler(self.dx, self.dt[i], self.u[i], self.x[i], self.x[i+1])
        return self.x
    
    def neighboring(self):
        for i in range(len(self.t)-1):
            self.xi[i+1] = handler(self.dxi, self.dt[i], self.u[i], self.xi[i], self.x[i])
        return self.xi
        
    def gradient(self):
        self.la.fill(0.)
        self.la[len(self.t)-1, :self.N] = (self.x[len(self.t)-1, 0:self.N] - self.y[len(self.t)-1])/self.obs_variance
        for i in range(len(self.t)-2, -1, -1):
            self.la[i] = handler(self.dla, self.dt[i], self.u[i], self.la[i+1], self.x[i]) # x should be current one.
            self.la[i, :self.N] += (self.x[i, :self.N] - self.y[i])/self.obs_variance
        return self.la[0]

    def gradient_from_x0(self, x0):
        self.x[0] = np.copy(x0)
        self.orbit()
        return self.gradient()

    def hessian_vector_product(self, xi0):
        self.xi[0] = np.copy(xi0)
        self.neighboring()
        self.nu.fill(0.)
        self.nu[len(self.t)-1, :self.N] = self.xi[len(self.t)-1, :self.N]/self.obs_variance
        for i in range(len(self.t)-2, -1, -1):
            self.nu[i] = handler(self.dnu, self.dt[i], self.u[i], self.nu[i+1], self.x[i], self.la[i+1], self.xi[i]) # x and xi should be current one.
            self.nu[i, :self.N] += self.xi[i, :self.N]/self.obs_variance
        return self.nu[0]
    
    def cost(self, x0):
        self.x[0] = np.copy(x0)
        self.orbit()
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(len(self.t)):                                                                
#            print ((self.x[self.it*i] - self.y[i]) @ (self.x[self.it*i] - self.y[i]))
            cost += (self.x[i, 0:self.N] - self.y[i]) @ (self.x[i, 0:self.N] - self.y[i])
        return cost/self.obs_variance/2.0 # fixed
    
    def true_cost(self, tob):
        cost=0
    #    cost = (xzero - xb) * (np.linalg.inv(B)) * (xzero - xb)
        for i in range(len(self.t)):
            cost += (tob[i] - self.y[i]) @ (tob[i] - self.y[i])
        return cost/self.obs_variance/2.0 # fixed
    
    def numerical_gradient_from_x0(self,x0,h):
        gr = np.zeros(self.N + self.M)
        c1 = self.cost(x0)
        for j in range(self.N + self.M):
            xx = np.copy(x0)
            xx[j] += h
            c = self.cost(xx)
            gr[j] = (c - c1)/h
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
        global a_trace, b_trace, K_trace, n_trace, cost_trace, trace
        cos = self.cost(x0)
        a_trace.append(x0[self.N])
        b_trace.append(x0[self.N+1])
        K_trace.append(x0[self.N+2])
        n_trace.append(x0[self.N+3])
        cost_trace.append(cos)
        for i in range(self.N):
            trace[i].append(x0[i])
        

#%%
#def compare_orbit(dat1, dat2):
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.plot(dat1[:,0],dat1[:,1],dat1[:,2],label='true orbit')
#    ax.plot(dat2[:,0],dat2[:,1],dat2[:,2],label='assimilated')
#    ax.set_xlabel('$x_0$')
#    ax.set_ylabel('$x_1$')
#    ax.set_zlabel('$x_2$')
#    plt.legend()
#    plt.show()
#
#def compare_orbit3(dat1, dat2, dat3, label1, label2, label3):
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.plot(dat1[:,0],dat1[:,1],dat1[:,2],label=label1)
#    ax.plot(dat2[:,0],dat2[:,1],dat2[:,2],label=label2)
#    ax.plot(dat3[:,0],dat3[:,1],dat3[:,2],label=label3)
#    ax.set_xlabel('$x_0$')
#    ax.set_ylabel('$x_1$')
#    ax.set_zlabel('$x_2$')
#    plt.legend()
#    plt.show()

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

#ctrl = impulse

N = 1
M = 4

#pref = "/Users/konta/bitbucket/androsace/dacamp/hill_discrete/data/"+ ctrl.__name__ +"/1/"
#pref = "/Users/konta/bitbucket/androsace/dacamp/hill_discrete_sysnoise/data/"+ ctrl.__name__ +"/1/"

#dt = 0.01

#T = 2.
#print("T", T)
#it = 5
#minute_steps = int(T/dt)
#steps = int(minute_steps/it)

stddev = 1.


lorenz = Hill()

#tob = np.loadtxt(pref + "true.1.dat", ndmin=2)

u = np.loadtxt("/Users/konta/Documents/journal/sysbio/experimental_data_Tsuchiya_et_al.codeconverted.extracted.csv", delimiter=",", usecols=(1), ndmin=2)[1:] #pERK
#obs = np.loadtxt("/Users/konta/Documents/journal/sysbio/experimental_data_Tsuchiya_et_al.codeconverted.extracted.csv", delimiter=",", usecols=(11), ndmin=2)[1:] # c-Jun
#obs = np.loadtxt("/Users/konta/Documents/journal/sysbio/experimental_data_Tsuchiya_et_al.codeconverted.extracted.csv", delimiter=",", usecols=(16), ndmin=2)[1:] # c-Fos
obs = np.loadtxt("/Users/konta/Documents/journal/sysbio/experimental_data_Tsuchiya_et_al.codeconverted.extracted.csv", delimiter=",", usecols=(21), ndmin=2)[1:] # Egr1

#t = np.arange(0., T, dt)
#t_plus_1 = np.arange(0., T+dt, dt)
#t_it = np.arange(0., T, dt*it)
#t_it_plus_1 = np.arange(0., T+dt, dt*it)
t = np.loadtxt("/Users/konta/Documents/journal/sysbio/experimental_data_Tsuchiya_et_al.codeconverted.extracted.csv", delimiter=",", usecols=(0))[1:] # t (min)
dt = np.zeros(len(t)-1)
for i in range(len(t)-1):
    dt[i] = t[i+1] - t[i]

x_opt = np.zeros(N + M)
#x_opt[0:N] = np.loadtxt(pref + "assimilation_xzero.2.dat", ndmin=2)
#x_opt[0:N] = np.copy(tob[0])
x_opt[0:N] = np.zeros(N)
x_opt[N] = -0.1
x_opt[N+1] = 0.1
x_opt[N+2] = 0.3
x_opt[N+3] = 1.2

x = np.zeros((len(t), N + M))
la = np.zeros((len(t), N + M))
xi = np.zeros((len(t), N + M))
nu = np.zeros((len(t), N + M))

scheme = Adjoint(lorenz.gradient, lorenz.gradient_adjoint, lorenz.gradient_neighboring, lorenz.gradient_secondadj, N, dt, stddev, t, u, x, obs, la, xi, nu)
# dx, dla, dxi, dnu, N, dt, obs_variance, t, u, x, y, la, xi, nu
print("Before assimilation")
print("cost", scheme.cost(x_opt))
compare_orbit(t, t, obs[:,0], scheme.x[:,0], "observed", "initial value")

print("Analytical and numerical gradient comparison")
gr_anal = scheme.gradient_from_x0(x_opt)
print ("gr_anal", gr_anal)
gr_num = scheme.numerical_gradient_from_x0(x_opt, 0.001)
print ("gr_num", gr_num)
if (not (0 in gr_num)):
    print ("relative error", (gr_anal - gr_num)/gr_num)

#%%
fig = plt.figure()
a_trace = [x_opt[N]]
b_trace = [x_opt[N+1]]
K_trace = [x_opt[N+2]]
n_trace = [x_opt[N+3]]
cost_trace = [scheme.cost(x_opt)]
trace=[]
for i in range(N):
    trace.append([x_opt[i]])

bnds = ((0, None), (None, None), (None, None), (0.1, None), (0, None))
res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', bounds=bnds, callback=scheme.cbf, options={'disp': None, 'maxls': 40, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
#res = minimize(scheme.cost, x_opt, jac=scheme.gradient_from_x0, method='L-BFGS-B', callback=scheme.cbf)
plt.show()
print (res)
estimated_x0 = np.copy(res.x)

for j in range(N):
    fig = plt.figure()
    plt.plot(t, scheme.x[:,j], label='assimilated')
    plt.plot(t, obs[:,j], label='observed')
    plt.legend()
    plt.show()

#%%
#fig = plt.figure()
#plt.plot(t_plus_1, [np.linalg.norm(scheme.x[i,0:N] - tob[i])/math.sqrt(N) for i in range(len(t_plus_1))], label='x norm')
#plt.xlabel('t')
#plt.ylabel('RMSE')
##plt.yscale('symlog')
#plt.legend()
#plt.show()

#print ("RMSE: ", np.mean([np.linalg.norm(scheme.x[i,0:N] - tob[i])/math.sqrt(N) for i in range(int(len(t_plus_1)*0.4),int(len(t_plus_1)*0.6))]))

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
std_deviation = np.array([math.sqrt(variance[i]) for i in range(N + M)])
#print("variance", variance)

hess_num = scheme.numerical_hessian_from_x0(estimated_x0, 0.001)
print ("hessian_num", hess_num)

#hess_num2 = scheme.numerical_hessian_from_x0_2(estimated_x0, 0.001)
#print ("hessian_num2", hess_num2)


rel_error = (hessian - hess_num)/ hessian
print ("relative error", rel_error)

#rel_error_2 = (hessian - hess_num2)/ hessian
#print ("relative error2", rel_error_2)

abs_error = (hessian - hess_num)
print ("absolute error", abs_error)

#abs_error_2 = (hessian - hess_num2)
#print ("absolute error2", abs_error_2)

hess_num_inv = np.linalg.inv(hess_num)

for j in range(N):
#    plt.plot([tob[0,j] for i in range(len(trace[j]))], 'r')
    plt.plot(trace[j], 'b')
    plt.errorbar(len(trace[j])-1, trace[j][-1], yerr=std_deviation[j], fmt='b')
    plt.legend()
    plt.xlabel('minimizer iteration')
    plt.ylabel('x_{' + str(j) + '}')
    plt.show()

#true_param = np.array([-3., 12., 0.3132, 1.276])

fig = plt.figure()
#plt.plot([true_param[0] for i in range(len(a_trace))], 'r')
plt.plot(a_trace, 'b')
plt.errorbar(len(a_trace)-1, a_trace[-1], yerr=std_deviation[N], fmt='b')
plt.legend()
plt.xlabel('minimizer iteration')
plt.ylabel('a')
plt.show()

fig = plt.figure()
#plt.plot([true_param[1] for i in range(len(b_trace))], 'r')
plt.plot(b_trace, 'b')
plt.errorbar(len(b_trace)-1, b_trace[-1], yerr=std_deviation[N+1], fmt='b')
plt.legend()
plt.xlabel('minimizer iteration')
plt.ylabel('b')
plt.show()

fig = plt.figure()
#plt.plot([true_param[2] for i in range(len(K_trace))], 'r')
plt.plot(K_trace, 'b')
plt.errorbar(len(K_trace)-1, K_trace[-1], yerr=std_deviation[N+2], fmt='b')
plt.legend()
plt.xlabel('minimizer iteration')
plt.ylabel('K')
plt.show()

fig = plt.figure()
#plt.plot([true_param[3] for i in range(len(n_trace))], 'r')
plt.plot(n_trace, 'b')
plt.errorbar(len(n_trace)-1, n_trace[-1], yerr=std_deviation[N+3], fmt='b')
plt.legend()
plt.xlabel('minimizer iteration')
plt.ylabel('n')
plt.show()


fig = plt.figure()
plt.plot(cost_trace, 'b')
plt.xlabel('minimizer iteration')
plt.ylabel('cost')
plt.show()


##%%
#print('4DVar optimal cost: ', res.fun)
#print('true cost: ', scheme.true_cost(tob))