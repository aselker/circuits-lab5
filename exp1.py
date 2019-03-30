#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from ekvfit import ekvfit
from scipy.optimize import minimize

vg = []
isat = []

with open("data/exp1_nmos.csv") as f:
  c = csv.reader(f, delimiter=",")
  next(c) # Throw away the header
  for row in c:
    vg += [float(row[0])]
    isat += [float(row[1])] 

def clip_range(xs, ys, bounds):
  pairs = [(x, y) for (x, y) in zip(xs, ys) if (bounds[0] <= y) and (y <= bounds[1])]
  out = list(zip(*pairs))
  return list(out[0]), list(out[1])

vg, isat = clip_range(vg, isat, (np.finfo(float).eps, np.inf)) # Clip to positive

def fit(xs, ys, model, initial_params):
  def err_f(params): return np.mean(np.power(np.log(ys) - np.log(model(xs, params)), 2))
  res = minimize(err_f, x0 = initial_params, method='Nelder-Mead')
  print(res)
  return res.x

# From ekvfit
def ekv_n(vg, params):
  Is, Vt, kappa = params
  return Is * np.power(np.log(1 + np.exp(kappa*(vg - Vt)/(2*0.0258))), 2)

params = fit(vg, isat, ekv_n, [7.2197482429849523e-08, 0.50917435504354447, 3.3782458038598859])

fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)

ax.semilogy(vg, isat, 'b.', label="Saturation current (experimental)")
ax.semilogy(vg, ekv_n(vg, params), 'g-', label="Saturation current (theoretical, Is = %g, Vt0 = %g, κ = %g)" %  (params[0], params[1], params[2]))

plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.grid(True)
ax.legend()
# plt.show()
plt.savefig("exp1-vi-semilog.pdf")
