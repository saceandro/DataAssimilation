#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:23:06 2018

@author: konta
"""

import numpy as np

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
        
        return self.dxdt
    
    def calc_jacobian(self, t, x, p):
        """
        Return Jacobian.
        Tangent linear model is calculated as follows,
        dx(t+1) = (I + dt * jacobian(t ,x(t), p)) @ dx(t).
        """        
        # Fill in Jacobian here.

        return self.jacobian

    def calc_hessian(self, t, x, p):
        """
        Return Hessian.
        """
        # Fill in Hessian here.

        return self.hessian

#%%
class Lorenz96:       # Example code for Lorenz96 model
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
