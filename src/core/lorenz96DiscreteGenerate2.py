#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:17:20 2017

@author: yk
"""
import sys
import os
import pathlib
import numpy as np
import math

import util
import model4DVar
import discrete4DVar

#%%
N = 5
M = 2
true_p = np.array([8., 1.])
obs_variance = 1.
it = 5
dt = 0.01

pref = "/Users/konta/bitbucket/androsace/assimilation/data/lorenz_discrete/N_"\
     + str(N) + "/M_" + str(M) + "/p1_" + str(true_p[0]) + "/p2_" + str(true_p[1])\
     + "/obsvar_" + str(obs_variance) +  "/obsiter_" + str(it) + "/dt_" + str(dt) + "/"

seeds = 2
year = 2
day = 365 * year
T = day * 0.2
_t = np.arange(0., T, dt)
t = np.concatenate((_t, np.array([T+dt])))
steps = len(t)

lorenz = model4DVar.Lorenz96(N, M)

seeds = np.arange(0, seeds, 1)
for seed in seeds:
    pref_seed = pref + "seed_" + str(seed) + "/"
    pathlib.Path(pref_seed).mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    
    scheme = discrete4DVar.Adjoint(lorenz, dt, t, obs_variance, np.zeros((steps, N)), rng)
    scheme.p = true_p
    
    x0 = true_p[0] * np.ones(N)
    x0[rng.randint(N)] += 0.1*rng.randn()
    
    scheme.x[0] = x0
    true_orbit = scheme.orbit()
    observed = true_orbit + math.sqrt(obs_variance) * rng.randn(steps, N)

    print ("observation RMSE: ", np.mean([np.linalg.norm(observed[i] - true_orbit[i])/math.sqrt(N) for i in range(steps)]))
        
    with open(pref_seed + 'true.tsv', 'w') as f:
        for i in range(int(steps/2), steps):
            f.write(("%f\t"*N + "\n") % tuple(true_orbit[i]))
        f.close()

    with open(pref_seed + 'observed.tsv', 'w') as f:
        for i in range(int(steps/2), steps, it):
            mask = rng.randint(N, size=N)
            for j in range(N):
                if mask[j] != 0:
                    f.write("%f\t" % observed[i,j])
                else:
                    f.write("NA\t")
            f.write("\n")
            for k in range(it-1):
                f.write("NA\t"*N + "\n")
        f.close()
