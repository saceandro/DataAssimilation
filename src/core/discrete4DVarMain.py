#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:01:40 2017

@author: yk
"""
import util
import discrete4DVar

#%%
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

tob = np.loadtxt(pref + "true.seed1.dat")
obs = np.genfromtxt(pref + "observed.missing.iteration" + str(it) + ".seed1.dat")
obs_mask = np.array(list(map(np.isnan, obs[:steps])))
obs_mask_row = [all(~obs_mask[i]) for i in range(steps)]

util.plot_orbit2_3d(tob[:steps], obs[:steps][obs_mask_row], "true orbit", "observed")

x0_p = np.zeros(N + M)
#x0_p[:N] = np.loadtxt(pref + "assimilation_xzero.seed2.dat")
#x0_p[0:N] = np.copy(tob[0])
#x0_p[:N] = np.random.randn(N)
#x0_p[N] = 4.
#x0_p[N+1] = 0.5
x0_p = np.random.randn(N+M)

scheme = discrete4DVar.Adjoint(N, M, dt, t, obs_variance, obs[:steps])
true_cost = scheme.cost(np.concatenate((tob[0], true_p)))

print("Before assimilation")
print("cost", scheme.cost(x0_p))
util.plot_orbit3_3d(tob[:steps], obs[:steps][obs_mask_row], scheme.x, 'true orbit', 'observed', 'pre-assimilation orbit')
util.plot_orbit2_3d(tob[:steps], scheme.x, 'true orbit', 'pre-assimilation orbit')

print("Analytical and numerical gradient comparison")
gr_anal = scheme.gradient(x0_p)
print ("gr_anal", gr_anal)
gr_num = scheme.numerical_gradient(x0_p, 0.001)
print ("gr_num", gr_num)
if all(gr_num > 0):
    print ("relative error", (gr_anal - gr_num)/gr_num)

#%%
fig = plt.figure()

#bnds = tuple([(-8., 24.) for i in range(N)])
res = scheme.minimize(x0_p)
print (res)
print ("true x0", tob[0])
estimated_x0_p = np.copy(res.x)

util.plot_orbit3_1d(t, tob[:steps], scheme.x, obs[:steps])
util.plot_orbit2_3d(tob[:steps], scheme.x, 'true orbit', 'assimilated')

#%%
util.plot_RMSE(t, tob, scheme.x)
print ("mean RMSE around interval center: ", np.mean([np.linalg.norm(scheme.x[i] - tob[i])/math.sqrt(N) for i in range(int(len(t)*0.4),int(len(t)*0.6))]))

#%%
cov = scheme.calc_covariance()
var = np.diag(cov)
sigma = scheme.sigma

cov_num = scheme.numerical_covariance(estimated_x0_p, 0.001)
cov_num2 = scheme.numerical_covariance2(estimated_x0_p, 0.001)
abs_error = cov - cov_num
abs_error2 = cov - cov_num2
rel_error = abs_error/cov_num
rel_error2 = abs_error2/cov_num2

scheme.plot_trace(true_cost, tob[0], true_p)

#%%
print('true cost: ', true_cost)
print('4DVar optimal cost: ', res.fun)
