#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 18:58:15 2018

@author: konta
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math

#%%
def plot_orbit3d(dat, label):
    """
    Plot first three state dimensions.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(dat[:,0],dat[:,1],dat[:,2], label=label)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    plt.show()

def plot_orbit2_3d(dat1, dat2, label1, label2):
    """
    Plot first three state dimensions for two data.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(dat1[:,0],dat1[:,1],dat1[:,2],label=label1)
    ax.plot(dat2[:,0],dat2[:,1],dat2[:,2],label=label2)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    plt.legend()
    plt.show()

def plot_orbit3_3d(dat1, dat2, dat3, label1, label2, label3):
    """
    Plot first three state dimensions for three data.
    """
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

def plot_orbit2_1d(t, assim, obs):
    """
    Plot each state dimension for true state, assimilated state, and observation state.
    """
    for j in range(len(assim[0])):
        mask = np.array(list(map(np.isnan, obs[:,j])))
        fig = plt.figure()
        plt.plot(t, assim[:,j], label='assimilated')
        plt.plot(t[~mask], obs[:,j][~mask], label='observed')
        plt.xlabel('t')
        plt.ylabel('$x_{' + str(j) + '}$')
        plt.legend()
        plt.show()

def plot_orbit3_1d(t, tob, assim, obs):
    """
    Plot each state dimension for true state, assimilated state, and observation state.
    """
    for j in range(len(tob[0])):
        mask = np.array(list(map(np.isnan, obs[:,j])))
        fig = plt.figure()
        plt.plot(t, tob[:,j], label='true orbit')
        plt.plot(t, assim[:,j], label='assimilated')
        plt.plot(t[~mask], obs[:,j][~mask], label='observed')
        plt.xlabel('t')
        plt.ylabel('$x_{' + str(j) + '}$')
        plt.legend()
        plt.show()
        
def plot_RMSE(t, tob, assim):
    """
    Plot RMSE of states.
    """
    fig = plt.figure()
    plt.plot(t, [np.linalg.norm(assim[i] - tob[i])/math.sqrt(len(tob[0])) for i in range(len(t))], label='RMSE')
    plt.xlabel('t')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()
