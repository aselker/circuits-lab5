#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from ekvfit import ekvfit
from scipy.optimize import minimize

def clip(xs, ys, xbounds, ybounds):
  pairs = [(x, y) for (x, y) in zip(xs, ys) if (xbounds[0] <= x) and (x <= xbounds[1]) and (ybounds[0] <= y) and (y <= ybounds[1])]
  out = list(zip(*pairs))
  return np.array(out[0]), np.array(out[1])

def clip_range(xs, ys, bounds):
  return clip(xs, ys, (-np.inf, np.inf), bounds)

def fit(xs, ys, model, initial_params):
  def err_f(params): return np.mean(np.power(np.log(ys) - np.log(model(xs, params)), 2))
  res = minimize(err_f, x0 = initial_params, method='Nelder-Mead')
  print(res)
  return res.x

# From ekvfit
def ekv_n(vg, params):
  Is, Vt, kappa = params
  return Is * np.power(np.log(1 + np.exp(kappa*(vg - Vt)/(2*0.0258))), 2)

def ekv_p(vg, params):
  return ekv_n(5-np.array(vg), [params[0], -params[1], params[2]])

# Process n-type data
vg_n = []
isat_n = []

with open("data/exp1_nmos.csv") as f:
  c = csv.reader(f, delimiter=",")
  next(c) # Throw away the header
  for row in c:
    vg_n += [float(row[0])]
    isat_n += [float(row[1])] 

vg_n, isat_n = clip_range(vg_n, isat_n, (np.finfo(float).eps, np.inf)) # Clip to positive

params_n = fit(vg_n, isat_n, ekv_n, [7.2197482429849523e-08, 0.50917435504354447, 3.3782458038598859])


#Process p-type data
vg_p = []
isat_p = []

with open("data/exp1_pmos.csv") as f:
  c = csv.reader(f, delimiter=",")
  next(c) # Throw away the header
  for row in c:
    vg_p += [float(row[0])]
    isat_p += [-float(row[1])] 

# Clip to positive, and drop invalid end
vg_p, isat_p = clip(vg_p, isat_p, (-np.inf, 4.6), (np.finfo(float).eps, np.inf))

params_p = fit(vg_p, isat_p, ekv_p, [7.2197482429849523e-08, -0.50917435504354447, 3.3782458038598859])


fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)

ax.semilogy(vg_n, isat_n, 'b.', label="N-type current (experimental)")
ax.semilogy(vg_n, ekv_n(vg_n, params_n), 'g-', label="N-type current (theoretical, Is = %g, Vt0 = %g, κ = %g)" %  (params_n[0], params_n[1], params_n[2]))
ax.semilogy(vg_p, isat_p, 'r.', label="P-type current (experimental)")
ax.semilogy(vg_p, ekv_p(vg_p, params_p), 'y-', label="P-type current (theoretical, Is = %g, Vt0 = %g, κ = %g)" %  (params_p[0], params_p[1], params_p[2]))

plt.title("N- and P-type current-voltage characteristics")
plt.xlabel("Gate voltage (v)")
plt.ylabel("Current (A)")
plt.grid(True)
ax.legend()
plt.savefig("exp1-vi-semilog.pdf")
