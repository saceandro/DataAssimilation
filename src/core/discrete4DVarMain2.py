#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:01:40 2017

@author: yk
"""
import util
import discrete4DVar

#%%
seed = 1
N = 5
M = 2
true_p = np.array([8., 1.])
pref = "/Users/konta/bitbucket/androsace/dacamp/task1/lorenz_discrete_missing_data/" + str(N) + "/"

dt = 0.01
T = 1.
_t = np.arange(0., T, dt)
t = np.concatenate((_t, np.array([T+dt])))
steps = len(t)
it = 5
obs_variance = 1.
trials = 2

tob = np.loadtxt(pref + "true.seed1.dat", ndmin=2)
obs = np.genfromtxt(pref + "observed.missing.iteration" + str(it) + ".seed1.dat", missing_values="NA")
obs_mask = np.array(list(map(np.isnan, obs[:steps])))
obs_mask_row = [all(~obs_mask[i]) for i in range(steps)]

util.plot_orbit2_3d(tob[:steps], obs[:steps][obs_mask_row], "true orbit", "observed")

rng = np.random.RandomState(seed)
scheme = discrete4DVar.Adjoint(N, M, dt, t, obs_variance, obs[:steps], rng)
true_cost = scheme.cost(np.concatenate((tob[0], true_p)))

#%%
minres = scheme.twin_experiment(true_cost, tob[0], true_p, trials)
print(minres)
#%%
print('\nestiamted initial state:', minres.x[:N])
print('true initial state:\t', tob[0])
print('1 sigma confidence interval:\t', scheme.sigma[:N])
print('\nestimated parameters:\t', minres.x[N:])
print('true parameters:\t', true_p)
print('1 sigma confidence interval:\t', scheme.sigma[N:])
print("\ninitial state and parameter covariance:\n", scheme.cov)
print('\noptimal cost:\t', minres.fun)
print('true cost:\t', true_cost)

util.plot_orbit3_1d(t, tob[:steps], scheme.x, obs[:steps])
util.plot_orbit2_3d(tob[:steps], scheme.x, 'true orbit', 'assimilated')
util.plot_RMSE(t, tob, scheme.x)
print ("mean RMSE around interval center: ", np.mean([np.linalg.norm(scheme.x[i] - tob[i])/math.sqrt(N) for i in range(int(len(t)*0.4),int(len(t)*0.6))]))
