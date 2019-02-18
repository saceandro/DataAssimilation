#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:01:40 2017

@author: yk
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.optimize import minimize
import math

np.seterr(all='warn', over='raise')

#%%
class Model:
    def __init__(self, N, M):
        self.N = N    # Dimension of state.      dim(x)
        self.M = M    # Dimension of parameters. dim(p)
        self.dxdt = np.zeros(self.N)  # dxdt of continuous model
        self.jacobian = np.zeros((self.N, self.N + self.M))  # Jacobian of continuous model
        self.hessian = np.zeros((self.N, self.N + self.M, self.N + self.M))  # Hessian of continuous model
    
    def calc_dxdt(self, t, x, p):
        """
        Return dxdt.
        Next state is calculated as follows,
        x(t+1) = x(t) + dt * dxdt(t, x(t), p)
        """
        # Fill in dxdt here.
        self.dxdt[0]        = p[1] * (x[1]   - x[self.N-2]) * x[self.N-1] + p[0] - x[0]
        self.dxdt[1]        = p[1] * (x[2]   - x[self.N-1]) * x[0]        + p[0] - x[1]
        for i in range(2, self.N-1):
            self.dxdt[i]    = p[1] * (x[i+1] - x[i-2])      * x[i-1]      + p[0] - x[i]
        self.dxdt[self.N-1] = p[1] * (x[0]   - x[self.N-3]) * x[self.N-2] + p[0] - x[self.N-1]
        
        return self.dxdt
    
    def calc_jacobian(self, t, x, p):
        """
        Return Jacobian.
        Tangent linear model is calculated as follows,
        dx(t+1) = (I + dt * jacobian(t ,x(t), p)) @ dx(t).
        """        
        # Fill in Jacobian here.
        for i in range(self.N):
            for j in range(self.N + self.M):
                self.jacobian[i,j] = p[1] * ((((i+1) % self.N) == j) - (((i-2) % self.N) == j)) * x[(i-1) % self.N]\
                                   + p[1] * (x[(i+1) % self.N]       - x[(i-2) % self.N])       * (((i-1) % self.N) == j)\
                                   + (self.N+1 == j)    * (x[(i+1) % self.N]       - x[(i-2) % self.N])       * x[(i-1) % self.N]\
                                   + (self.N   == j)\
                                   - (i        == j)
        return self.jacobian

    def calc_hessian(self, t, x, p):
        """
        Return Hessian.
        """
        # Fill in Hessian here.
        for i in range(self.N):
            for j in range(self.N + self.M):
                for k in range(self.N + self.M):
                    self.hessian[i,j,k] = ((self.N+1)==j) * ((((i+1)%self.N)==k) - (((i-2)%self.N)==k)) * x[(i-1)%self.N]\
                                        + ((self.N+1)==j) * (x[(i+1)%self.N]     - x[(i-2)%self.N])     * (((i-1)%self.N)==k)\
                                        + ((self.N+1)==k) * ((((i+1)%self.N)==j) - (((i-2)%self.N)==j)) * x[(i-1)%self.N]\
                                        + p[1]            * ((((i+1)%self.N)==j) - (((i-2)%self.N)==j)) * (((i-1)%self.N)==k)\
                                        + ((self.N+1)==k) * (x[(i+1)%self.N]     - x[(i-2)%self.N])     * (((i-1)%self.N)==j)\
                                        + p[1]            * ((((i+1)%self.N)==k) - (((i-2)%self.N)==k)) * (((i-1)%self.N)==j)
        return self.hessian
        

class Adjoint(Model):
    def __init__(self, N, M, dt, t, obs_variance, obs, rng):
        super().__init__(N, M)
        self.dt = dt
        self.t = t      # e.g. t = [0, 0.01, 0.02, ...] when dt = 0.01
        self.steps = len(t)
        self.obs_variance = obs_variance  # Observation variance
        self.obs = obs  # dim(obs)  = steps x N  , obs[i,j] == np.nan means loss of jth variable observation at t[i]. (Especially no observation made at time t[i] if all(obs[i,:] == np.nan) == True.)
        self.x = np.zeros((self.steps, N))      # State variables.
        self.p = np.zeros(M)      # Parameters.
        self.dx = np.zeros((self.steps, N))     # Frechet derivative of x.
        self.dp = np.zeros(M)     # Frechet derivative of p.
        self.la = np.zeros((self.steps, N+M))   # Lagrange multipliers.
        self.dla = np.zeros((self.steps, N+M))  # Frechet derivative of la.
        self.cost_trace = []
        self.x0_p_trace = []
        self.cov = np.zeros((N+M, N+M))
        self.sigma = np.zeros(N+M)
        self.__rng = rng

    def __next_x(self, t, x, p):
        """
        Return next state.
        """
        return x + self.calc_dxdt(t, x, p) * self.dt
    
    def __next_dx(self, t, x, p, dx, dp):
        """
        Return next neighboring state.
        """
        return dx + self.calc_jacobian(t, x, p) @ np.concatenate((dx, dp)) * self.dt
    
    def __prev_la(self, t, x, p, la):
        """
        Return previous lagrange multiplier w/o observation term.
        """
        return la + self.calc_jacobian(t, x, p).transpose() @ la[:self.N] * self.dt
    
    def __prev_dla(self, t, x, p, dx, dp, la, dla):
        """
        Return previous neighboring lagrange multiplier w/o observation term.
        """
        return self.__prev_la(t, x, p, dla) + (self.calc_hessian(t, x, p) @ np.concatenate((dx, dp))).transpose() @ la[:self.N] * self.dt
        
    def orbit(self):
        """
        Return state orbit.
        """
        for i in range(self.steps-1):
            self.x[i+1] = self.__next_x(self.t[i], self.x[i], self.p)
        return self.x
    
    def neighboring(self):
        """
        Return neighboring state orbit.
        """
        for i in range(self.steps-1):
            self.dx[i+1] = self.__next_dx(self.t[i], self.x[i], self.p, self.dx[i], self.dp)
        return self.dx
    
    def gradient(self, x0_p):
        """
        Return gradient wrt initial state and parameters x0_p = (x[0], pa).
        """
        self.x[0] = x0_p[:self.N]
        self.p    = x0_p[self.N:]
        self.orbit()
        for j in range(self.N):
            if not np.isnan(self.obs[self.steps-1, j]):
                self.la[self.steps-1, j] = (self.x[self.steps-1, j] - self.obs[self.steps-1, j]) / self.obs_variance
            else:
                self.la[self.steps-1, j] = 0.
        for i in range(self.steps-2, -1, -1):
            self.la[i] = self.__prev_la(self.t[i], self.x[i], self.p, self.la[i+1]) # x should be current one.
            for j in range(self.N):
                if not np.isnan(self.obs[i, j]):
                    self.la[i, j] += (self.x[i, j] - self.obs[i, j]) / self.obs_variance
        return self.la[0]

    def hessian_vector_product(self, dx0_dp):
        """
        Return hessian vector product wrt initial state and parameters dx0_dp = (dx[0], dp).
        """
        self.dx[0] = dx0_dp[:self.N]
        self.dp    = dx0_dp[self.N:]
        self.neighboring()
        for j in range(self.N):
            if not np.isnan(self.obs[self.steps-1, j]):
                self.dla[self.steps-1, j] = self.dx[self.steps-1, j] / self.obs_variance
            else:
                self.dla[self.steps-1, j] = 0.
        for i in range(self.steps-2, -1, -1):
            self.dla[i] = self.__prev_dla(self.t[i], self.x[i], self.p, self.dx[i], self.dp, self.la[i+1], self.dla[i+1])  # x and xi should be current one.
            for j in range(self.N):
                if not np.isnan(self.obs[i, j]):
                    self.dla[i, j] += self.dx[i, j] / self.obs_variance
        return self.dla[0]
    
    def cost(self, x0_p):
        """
        Return cost function wrt initial state and parameters x0_p = (x[0], p).
        """
        self.x[0] = np.copy(x0_p[:self.N])
        self.p    = np.copy(x0_p[self.N:])
        self.orbit()
        __cost=0
        for i in range(self.steps):
            __cost_t=0
            for j in range(self.N):
                if not np.isnan(self.obs[i, j]):
                    __cost_t += (self.x[i, j] - self.obs[i, j]) ** 2
            __cost += __cost_t
        return __cost/self.obs_variance/2.0
        
    def numerical_gradient(self, x0_p, h):
        """
        Return numerically calculated gradient
        wrt initial state and parameters x0_p = (x[0], p).
        """
        __gr = np.zeros(self.N + self.M)
        __c1 = self.cost(x0_p)
        for j in range(self.N + self.M):
            __xx = np.copy(x0_p)
            __xx[j] += h
            __c = self.cost(__xx)
            __gr[j] = (__c - __c1)/h
        return __gr
    
    def numerical_covariance(self, x0_p, h):
        """
        Return numerically calculated hessian vector product
        wrt initial state and parameters dx0_dp = (dx[0], dp)
        using analytically calculated gradient.
        """
        __hess = np.zeros((self.N + self.M, self.N + self.M))
        __gr1 = np.copy(self.gradient(x0_p))
        for i in range(self.N + self.M):
            for j in range(self.N + self.M):
                __xx = np.copy(x0_p)
                __xx[j] += h
                __gr2 = np.copy(self.gradient(__xx))
                __hess[j,i] = (__gr2[i] - __gr1[i])/h
        return np.linalg.inv(__hess)
    
    def numerical_covariance2(self, x0_p, h):
        """
        Return numerically calculated hessian vector product
        wrt initial state and parameters dx0_dp = (dx[0], dp)
        using numerically calculated gradient.
        """
        __hess = np.zeros((self.N + self.M, self.N + self.M))
        __gr1 = np.copy(self.numerical_gradient(x0_p, h))
        for i in range(self.N + self.M):
            for j in range(self.N + self.M):
                __xx = np.copy(x0_p)
                __xx[j] += h
                __gr2 = np.copy(self.numerical_gradient(__xx, h))
                __hess[j,i] = (__gr2[i] - __gr1[i])/h
        return np.linalg.inv(__hess)

    def cbf(self, x0_p):
        """
        Minimizer callback function to store iteration trace.
        """
        self.cost_trace.append(self.cost(x0_p))
        self.x0_p_trace.append(list(x0_p))
            
    def minimize(self, x0_p, bounds=None):
        """
        Minimize cost function wrt x0_p = (x[0], p) using L-BFGS-B.
        Give bounds for x0_p such as ((None, None), (-3, None), (None, 2), (-5, 10)).
        (Default no bounds.)
        """
        self.cbf(x0_p)
        return minimize(self.cost, x0_p, jac=self.gradient, method='L-BFGS-B', bounds=bounds, callback=self.cbf, options={'disp': None, 'maxls': 40, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})

    def calc_covariance(self):
        """
        Return covariance of initial state and parameters.
        Sqrt of each diagonal element represents 1 sigma confidence interval.
        """
        __hess = np.zeros((self.N + self.M, self.N + self.M))
        __dx0_dp = np.zeros(self.N + self.M)
        for i in range(self.N + self.M):
            __dx0_dp.fill(0.)
            __dx0_dp[i] = 1.
            __hess[i] = np.copy(self.hessian_vector_product(__dx0_dp))
        self.cov = np.linalg.inv(__hess)
        __var = np.diag(self.cov)
        self.sigma = np.array([math.sqrt(__var[i]) for i in range(self.N + self.M)])
        return self.cov

    def plot_trace(self):
        """
        Plot trace of each initial state and final confidence interval (1 sigma).
        """
        self.x0_p_trace = np.array(self.x0_p_trace)
        __minimizer_it = len(self.x0_p_trace)
        fig = plt.figure()
        plt.plot(self.cost_trace, 'b')
        plt.xlabel('minimizer iteration')
        plt.ylabel('cost')
        plt.show()
        for j in range(self.N):
            fig = plt.figure()
            plt.plot(self.x0_p_trace[:,j], 'b')
            plt.errorbar(__minimizer_it-1, self.x0_p_trace[-1,j], yerr=self.sigma[j], fmt='b')
            plt.legend()
            plt.xlabel('minimizer iteration')
            plt.ylabel('$x_{' + str(j) + '}$')
            plt.show()
        for j in range(self.N, self.N + self.M):
            fig = plt.figure()
            plt.plot(self.x0_p_trace[:,j], 'b')
            plt.errorbar(__minimizer_it-1, self.x0_p_trace[-1,j], yerr=self.sigma[j], fmt='b')
            plt.legend()
            plt.xlabel('minimizer iteration')
            plt.ylabel('$p_{' + str(j-self.N) + '}$')
            plt.show()

    def plot_trace_twin(self, true_cost, true_x0, true_p):
        """
        Plot trace of each initial state and final confidence interval (1 sigma).
        """
        self.x0_p_trace = np.array(self.x0_p_trace)
        __minimizer_it = len(self.x0_p_trace)
        fig = plt.figure()
        plt.plot([true_cost for i in range(__minimizer_it)], 'r')
        plt.plot(self.cost_trace, 'b')
        plt.xlabel('minimizer iteration')
        plt.ylabel('cost')
        plt.show()
        for j in range(self.N):
            fig = plt.figure()
            plt.plot([true_x0[j] for i in range(__minimizer_it)], 'r')
            plt.plot(self.x0_p_trace[:,j], 'b')
            plt.errorbar(__minimizer_it-1, self.x0_p_trace[-1,j], yerr=self.sigma[j], fmt='b')
            plt.legend()
            plt.xlabel('minimizer iteration')
            plt.ylabel('$x_{' + str(j) + '}$')
            plt.show()
        for j in range(self.N, self.N + self.M):
            fig = plt.figure()
            plt.plot([true_p[j-self.N] for i in range(__minimizer_it)], 'r')
            plt.plot(self.x0_p_trace[:,j], 'b')
            plt.errorbar(__minimizer_it-1, self.x0_p_trace[-1,j], yerr=self.sigma[j], fmt='b')
            plt.legend()
            plt.xlabel('minimizer iteration')
            plt.ylabel('$p_{' + str(j-self.N) + '}$')
            plt.show()
    
    def assimilate(self, trials=10, bounds=None):
        """
        Assimilate and calculate confidence intervals.
        Return the minimum cost result in the designated number of trials.
        """
        __mincost = np.inf
        __minres = None
        __min_cost_trace = None
        __min_x0_p_trace = None
        for i in range(trials):
            print("\niter:", i)
            __x0_p = self.__rng.randn(self.N + self.M)
            self.cost_trace = []
            self.x0_p_trace = []
            try:
                __res = self.minimize(__x0_p, bounds)
            except FloatingPointError as e:
                print('Exception occurred in minimizer:', e)
                continue
            if __res.fun < __mincost:
                print("Smaller cost found.")
                __mincost = __res.fun
                __minres = __res
                __min_cost_trace = np.copy(self.cost_trace)
                __min_x0_p_trace = np.copy(self.x0_p_trace)
        self.gradient(__minres.x)
        self.calc_covariance()
        self.cost_trace = __min_cost_trace
        self.x0_p_trace = __min_x0_p_trace
        self.plot_trace()
        return __minres
    
    def twin_experiment(self, true_cost, true_x0, true_p, trials=10, bounds=None):
        """
        Assimilate and calculate confidence intervals.
        Return the minimum cost result in the designated number of trials.
        Estimated result is shown compared to the true initial state and parameters.
        """
        __mincost = np.inf
        __minres = None
        __min_cost_trace = None
        __min_x0_p_trace = None
        for i in range(trials):
            print("\niter:", i)
            __x0_p = self.__rng.randn(self.N + self.M)
            self.cost_trace = []
            self.x0_p_trace = []
            try:
                __res = self.minimize(__x0_p, bounds)
            except FloatingPointError as e:
                print('Exception occurred in minimizer:', e)
                continue
            if __res.fun < __mincost:
                print("Smaller cost found.")
                __mincost = __res.fun
                __minres = __res
                __min_cost_trace = np.copy(self.cost_trace)
                __min_x0_p_trace = np.copy(self.x0_p_trace)
        self.gradient(__minres.x)
        self.calc_covariance()
        self.cost_trace = __min_cost_trace
        self.x0_p_trace = __min_x0_p_trace
        self.plot_trace_twin(true_cost, true_x0, true_p)
        return __minres

#    def check_covariance(self, true_cost, true_x0, true_p, trials=10, bounds=None):
#        __mincost = np.inf
#        __minres = None
#        __min_cost_trace = None
#        __min_x0_p_trace = None
#        __x0_p_stack = np.empty((0, self.N + self.M), float)
#        for i in range(trials):
#            print("\niter:", i)
#            __x0_p = self.__rng.randn(self.N + self.M)
#            self.cost_trace = []
#            self.x0_p_trace = []
#            try:
#                __res = self.minimize(__x0_p, bounds)
#            except FloatingPointError as e:
#                print('Exception occurred in minimizer:', e)
#                continue
#            __x0_p_stack = np.vstack((__x0_p_stack, __res.x.reshape(1, self.N + self.M)))
#            if __res.fun < __mincost:
#                print("Smaller cost found.")
#                __mincost = __res.fun
#                __minres = __res
#                __min_cost_trace = np.copy(self.cost_trace)
#                __min_x0_p_trace = np.copy(self.x0_p_trace)
#        self.gradient(__minres.x)
#        self.calc_covariance()
#        self.cost_trace = __min_cost_trace
#        self.x0_p_trace = __min_x0_p_trace
#        self.plot_trace_twin(true_cost, true_x0, true_p)
#        print("x0_p_stack", __x0_p_stack)
#        print("experiment cov:", np.cov(__x0_p_stack.transpose()))
#        return __minres
