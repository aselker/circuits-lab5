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

# From ekvfit
def ekv(vg, Is, Vt, kappa):
  return Is * np.power(np.log(1 + np.exp(kappa*(vg - Vt)/(2*0.0258))), 2)

# Error is mean-squared diff between theoretical and practice
def err_f(Is, Vt, kappa): return np.mean(np.power(np.log(isat) - np.log(ekv(np.array(vg), Is, Vt, kappa)), 2))

# Found with ekvfit, but not good enough
initial_params = [7.2197482429849523e-08, 0.50917435504354447, 3.3782458038598859]

res = minimize(lambda args: err_f(args[0], args[1], args[2]), x0 = initial_params, method='Nelder-Mead')
print(res)
params = res.x

fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)

ax.semilogy(vg, isat, 'b.', label="Saturation current (experimental)")
ax.semilogy(vg, ekv(vg, params[0], params[1], params[2]), 'g-', label="Saturation current (theoretical, Is = %g, Vt = %g, κ = %g)" %  (params[0], params[1], params[2]))

plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.grid(True)
ax.legend()
# plt.show()
plt.savefig("exp1-vi-semilog.pdf")
