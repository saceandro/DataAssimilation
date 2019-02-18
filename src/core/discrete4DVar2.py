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

class Adjoint:
    def __init__(self, model, dt, t, obs_variance, obs, rng):
        self.model = model                                      # Model used for assimilation.
        self.__N = self.model.N
        self.__M = self.model.M
        self.dt = dt
        self.t = t                                              # e.g. t = [0, 0.01, 0.02, ...] when dt = 0.01
        self.steps = len(t)
        self.obs_variance = obs_variance                        # Observation variance
        self.obs = obs                                          # dim(obs)  = steps x N  , obs[i,j] == np.nan means loss of jth variable observation at t[i]. (Especially no observation made at time t[i] if all(obs[i,:] == np.nan) == True.)
        self.x = np.zeros((self.steps, self.__N))               # State variables.
        self.p = np.zeros(self.__M)                             # Parameters.
        self.dx = np.zeros((self.steps, self.__N))              # Frechet derivative of x.
        self.dp = np.zeros(self.__M)                            # Frechet derivative of p.
        self.la = np.zeros((self.steps, self.__N + self.__M))   # Lagrange multipliers.
        self.dla = np.zeros((self.steps, self.__N + self.__M))  # Frechet derivative of la.
        self.cov = np.zeros((self.__N + self.__M, self.__N + self.__M))  # Initial state and parameter covariance calculated by 2nd order adjoint method.
        self.sigma = np.zeros(self.__N + self.__M)                       # 1 sigma confidence interval for initial state and parameter covariance
        self.__cost_trace = []
        self.__x0_p_trace = []
        self.__rng = rng

    def __next_x(self, t, x, p):
        """
        Return next state.
        """
        return x + self.model.calc_dxdt(t, x, p) * self.dt
    
    def __next_dx(self, t, x, p, dx, dp):
        """
        Return next neighboring state.
        """
        return dx + self.model.calc_jacobian(t, x, p) @ np.concatenate((dx, dp)) * self.dt
    
    def __prev_la(self, t, x, p, la):
        """
        Return previous lagrange multiplier w/o observation term.
        """
        return la + self.model.calc_jacobian(t, x, p).transpose() @ la[:self.__N] * self.dt
    
    def __prev_dla(self, t, x, p, dx, dp, la, dla):
        """
        Return previous neighboring lagrange multiplier w/o observation term.
        """
        return self.__prev_la(t, x, p, dla) + (self.model.calc_hessian(t, x, p) @ np.concatenate((dx, dp))).transpose() @ la[:self.__N] * self.dt
        
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
        self.x[0] = x0_p[:self.__N]
        self.p    = x0_p[self.__N:]
        self.orbit()
        for j in range(self.__N):
            if not np.isnan(self.obs[self.steps-1, j]):
                self.la[self.steps-1, j] = (self.x[self.steps-1, j] - self.obs[self.steps-1, j]) / self.obs_variance
            else:
                self.la[self.steps-1, j] = 0.
        for i in range(self.steps-2, -1, -1):
            self.la[i] = self.__prev_la(self.t[i], self.x[i], self.p, self.la[i+1]) # x should be current one.
            for j in range(self.__N):
                if not np.isnan(self.obs[i, j]):
                    self.la[i, j] += (self.x[i, j] - self.obs[i, j]) / self.obs_variance
        return self.la[0]

    def hessian_vector_product(self, dx0_dp):
        """
        Return hessian vector product wrt initial state and parameters dx0_dp = (dx[0], dp).
        """
        self.dx[0] = dx0_dp[:self.__N]
        self.dp    = dx0_dp[self.__N:]
        self.neighboring()
        for j in range(self.__N):
            if not np.isnan(self.obs[self.steps-1, j]):
                self.dla[self.steps-1, j] = self.dx[self.steps-1, j] / self.obs_variance
            else:
                self.dla[self.steps-1, j] = 0.
        for i in range(self.steps-2, -1, -1):
            self.dla[i] = self.__prev_dla(self.t[i], self.x[i], self.p, self.dx[i], self.dp, self.la[i+1], self.dla[i+1])  # x and xi should be current one.
            for j in range(self.__N):
                if not np.isnan(self.obs[i, j]):
                    self.dla[i, j] += self.dx[i, j] / self.obs_variance
        return self.dla[0]
    
    def cost(self, x0_p):
        """
        Return cost function wrt initial state and parameters x0_p = (x[0], p).
        """
        self.x[0] = np.copy(x0_p[:self.__N])
        self.p    = np.copy(x0_p[self.__N:])
        self.orbit()
        __cost=0
        for i in range(self.steps):
            __cost_t=0
            for j in range(self.__N):
                if not np.isnan(self.obs[i, j]):
                    __cost_t += (self.x[i, j] - self.obs[i, j]) ** 2
            __cost += __cost_t
        return __cost/self.obs_variance/2.0
        
    def numerical_gradient(self, x0_p, h):
        """
        Return numerically calculated gradient
        wrt initial state and parameters x0_p = (x[0], p).
        """
        __gr = np.zeros(self.__N + self.__M)
        __c1 = self.cost(x0_p)
        for j in range(self.__N + self.__M):
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
        __hess = np.zeros((self.__N + self.__M, self.__N + self.__M))
        __gr1 = np.copy(self.gradient(x0_p))
        for i in range(self.__N + self.__M):
            for j in range(self.__N + self.__M):
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
        __hess = np.zeros((self.__N + self.__M, self.__N + self.__M))
        __gr1 = np.copy(self.numerical_gradient(x0_p, h))
        for i in range(self.__N + self.__M):
            for j in range(self.__N + self.__M):
                __xx = np.copy(x0_p)
                __xx[j] += h
                __gr2 = np.copy(self.numerical_gradient(__xx, h))
                __hess[j,i] = (__gr2[i] - __gr1[i])/h
        return np.linalg.inv(__hess)

    def cbf(self, x0_p):
        """
        Minimizer callback function to store iteration trace.
        """
        self.__cost_trace.append(self.cost(x0_p))
        self.__x0_p_trace.append(list(x0_p))
            
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
        __hess = np.zeros((self.__N + self.__M, self.__N + self.__M))
        __dx0_dp = np.zeros(self.__N + self.__M)
        for i in range(self.__N + self.__M):
            __dx0_dp.fill(0.)
            __dx0_dp[i] = 1.
            __hess[i] = np.copy(self.hessian_vector_product(__dx0_dp))
        self.cov = np.linalg.inv(__hess)
        __var = np.diag(self.cov)
        self.sigma = np.array([math.sqrt(__var[i]) for i in range(self.__N + self.__M)])
        return self.cov

    def plot_trace(self):
        """
        Plot trace of each initial state and final confidence interval (1 sigma).
        """
        self.__x0_p_trace = np.array(self.__x0_p_trace)
        __minimizer_it = len(self.__x0_p_trace)
        fig = plt.figure()
        plt.plot(self.__cost_trace, 'b')
        plt.xlabel('minimizer iteration')
        plt.ylabel('cost')
        plt.show()
        for j in range(self.__N):
            fig = plt.figure()
            plt.plot(self.__x0_p_trace[:,j], 'b')
            plt.errorbar(__minimizer_it-1, self.__x0_p_trace[-1,j], yerr=self.sigma[j], fmt='b')
            plt.legend()
            plt.xlabel('minimizer iteration')
            plt.ylabel('$x_{' + str(j) + '}$')
            plt.show()
        for j in range(self.__N, self.__N + self.__M):
            fig = plt.figure()
            plt.plot(self.__x0_p_trace[:,j], 'b')
            plt.errorbar(__minimizer_it-1, self.__x0_p_trace[-1,j], yerr=self.sigma[j], fmt='b')
            plt.legend()
            plt.xlabel('minimizer iteration')
            plt.ylabel('$p_{' + str(j-self.__N) + '}$')
            plt.show()

    def plot_trace_twin(self, true_cost, true_x0, true_p):
        """
        Plot trace of each initial state and final confidence interval (1 sigma).
        """
        self.__x0_p_trace = np.array(self.__x0_p_trace)
        __minimizer_it = len(self.__x0_p_trace)
        fig = plt.figure()
        plt.plot([true_cost for _ in range(__minimizer_it)], 'r')
        plt.plot(self.__cost_trace, 'b')
        plt.xlabel('minimizer iteration')
        plt.ylabel('cost')
        plt.show()
        for j in range(self.__N):
            fig = plt.figure()
            plt.plot([true_x0[j] for _ in range(__minimizer_it)], 'r')
            plt.plot(self.__x0_p_trace[:,j], 'b')
            plt.errorbar(__minimizer_it-1, self.__x0_p_trace[-1,j], yerr=self.sigma[j], fmt='b')
            plt.legend()
            plt.xlabel('minimizer iteration')
            plt.ylabel('$x_{' + str(j) + '}$')
            plt.show()
        for j in range(self.__N, self.__N + self.__M):
            fig = plt.figure()
            plt.plot([true_p[j-self.__N] for _ in range(__minimizer_it)], 'r')
            plt.plot(self.__x0_p_trace[:,j], 'b')
            plt.errorbar(__minimizer_it-1, self.__x0_p_trace[-1,j], yerr=self.sigma[j], fmt='b')
            plt.legend()
            plt.xlabel('minimizer iteration')
            plt.ylabel('$p_{' + str(j-self.__N) + '}$')
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
            __x0_p = self.__rng.randn(self.__N + self.__M)
            self.__cost_trace = []
            self.__x0_p_trace = []
            try:
                __res = self.minimize(__x0_p, bounds)
            except FloatingPointError as e:
                print('Exception occurred in minimizer:', e)
                continue
            if __res.fun < __mincost:
                print("Smaller cost found.")
                __mincost = __res.fun
                __minres = __res
                __min_cost_trace = np.copy(self.__cost_trace)
                __min_x0_p_trace = np.copy(self.__x0_p_trace)
        self.gradient(__minres.x)
        self.calc_covariance()
        self.__cost_trace = __min_cost_trace
        self.__x0_p_trace = __min_x0_p_trace
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
            __x0_p = self.__rng.randn(self.__N + self.__M)
            self.__cost_trace = []
            self.__x0_p_trace = []
            try:
                __res = self.minimize(__x0_p, bounds)
            except FloatingPointError as e:
                print('Exception occurred in minimizer:', e)
                continue
            if __res.fun < __mincost:
                print("Smaller cost found.")
                __mincost = __res.fun
                __minres = __res
                __min_cost_trace = np.copy(self.__cost_trace)
                __min_x0_p_trace = np.copy(self.__x0_p_trace)
        self.gradient(__minres.x)
        self.calc_covariance()
        self.__cost_trace = __min_cost_trace
        self.__x0_p_trace = __min_x0_p_trace
        self.plot_trace_twin(true_cost, true_x0, true_p)
        return __minres
