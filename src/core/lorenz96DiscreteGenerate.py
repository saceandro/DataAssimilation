#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:17:20 2017

@author: yk
"""
import sys
import os
import pathlib

import util
import discrete4DVar

#%%
N = 5
M = 2
true_p = np.array([8., 1.])
pref = "/Users/konta/bitbucket/androsace/dacamp/task1/lorenz_discrete_missing_data/" + str(N) + "/"

dt = 0.01
year = 2
day = 365 * year
dt = 0.01
T = day * 0.2
_t = np.arange(0., T, dt)
t = np.concatenate((_t, np.array([T+dt])))
steps = len(t)
it = 5
obs_variance = 1.

t_day = np.arange(0.,T/0.2, dt/0.2)

seeds = np.array([1,2])

for seed in seeds:
    rng = np.random.RandomState(seed)
    
    scheme = discrete4DVar.Adjoint(N, M, dt, t, obs_variance, np.zeros((steps, N)), rng)
    scheme.p = true_p
    
    x0 = true_p[0] * np.ones(N)
    x0[rng.randint(N)] += 0.1*rng.randn()
    
    scheme.x[0] = x0
    true_orbit = scheme.orbit()
    observed = true_orbit + math.sqrt(obs_variance) * rng.randn(steps, N)

    print ("observation RMSE: ", np.mean([np.linalg.norm(observed[i] - true_orbit[i])/math.sqrt(N) for i in range(steps)]))
    
    assimilation_xzero = observed[rng.randint(len(observed))]
    
    pathlib.Path(pref).mkdir(parents=True, exist_ok=True)
    
    with open(pref + 'assimilation_xzero.seed' + str(seed) + '.dat', 'w') as ff:
        ff.write(("%f\t"*N + "\n") % tuple(assimilation_xzero))
        ff.close()
        
    with open(pref + 'true.seed' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps):
            f.write(("%f\t"*N + "\n") % tuple(true_orbit[i]))
        f.close()

    with open(pref + 'observed.seed' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps):
            f.write(("%f\t"*N + "\n") % tuple(observed[i]))
        f.close()

    sparse_obs = []
    with open(pref + 'observed.iteration' + str(it) + '.seed' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps, it):
            f.write(("%f\t"*N + "\n") % tuple(observed[i]))
            for j in range(it-1):
                f.write("NA\t"*N + "\n")
            sparse_obs.append(observed[i])
        f.close()

    with open(pref + 'observed.missing.iteration' + str(it) + '.seed' + str(seed) +'.dat', 'w') as f:
        for i in range(int(steps/2), steps, it):
            mask = rng.randint(5, size=N)
            for j in range(N):
                if mask[j] != 0:
                    f.write("%f\t" % observed[i,j])
                else:
                    f.write("NA\t")
            f.write("\n")
            for k in range(it-1):
                f.write("NA\t"*N + "\n")
        f.close()
    
#    observed_every6h = []
#    with open("data/observed6h." + str(seed) + ".dat", "w") as g:    
#        for i in range(int(steps/2),steps,interval):
#            g.write(("%f\t"*N + "\n") % tuple(observed[i,:]))
#            observed_every6h.append(observed[i,:])
#        g.close()
    
#    np.savetxt("data/cov." + str(seed) + ".dat", np.cov(np.transpose(np.asarray(observed_every6h))))
    
#    np.savetxt(pref + "cov." + str(it) + "." + str(seed) + ".dat", np.cov(np.transpose(np.asarray(sparse_obs))))
    
#    plot_orbit(np.asarray(true_orbit_every6h))
#    plot_orbit(np.asarray(observed_every6h))


#t_day_every6h = []
#for i in range(int(steps/2),steps,5):
#    t_day_every6h.append(t_day[i])
#
##%%
#plt.xlabel("day")
#plt.plot(t_day_every6h[0:100],[item[0] for item in observed_every6h[0:100]], label='observed')
#plt.plot(t_day_every6h[0:100],[item[0] for item in true_orbit_every6h[0:100]], label='true_orbit')
#plt.legend()
#plt.show()