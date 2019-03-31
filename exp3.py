#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def clip(xs, ys, xbounds, ybounds):
  pairs = [(x, y) for (x, y) in zip(xs, ys) if (xbounds[0] <= x) and (x <= xbounds[1]) and (ybounds[0] <= y) and (y <= ybounds[1])]
  out = list(zip(*pairs))
  return np.array(out[0]), np.array(out[1])

def linterp(xs, ys, target):
  closest = -1
  dist = np.inf
  for (i, x) in enumerate(xs):
    if abs(x-target) < dist:
      dist = abs(x-target)
      closest = i

  i1, i2 = 0, 0
  if (closest == len(xs)-1) or ( closest != 0 and abs(xs[closest-1] - target) < abs(xs[closest+1] - target) ):
    i1 = closest-1
    i2 = closest
  else:
    i1 = closest
    i2 = closest+1
  
  x1 = xs[i1]
  x2 = xs[i2]
  y1 = ys[i1]
  y2 = ys[i2]
  return y1 + (target-x1) * ( (y2-y1) / (x2-x1) )
    

# for each curve:
# Get saturation voltage, somehow
# Make linear fit for sat region to find early voltage and isat
# Make linear fit for ohmic region
# combine slopes to get intrinsic gain

vt_n, vt_p = 0.56, 4.34

vd_n = [[], [], []]
i_n = [[], [], []]

names = ["data/exp3_nmos_0.46.csv", "data/exp3_nmos_0.56.csv", "data/exp3_nmos_5.csv"]
for i in range(3):
  with open(names[i]) as f:
    c = csv.reader(f, delimiter=",")
    next(c) # Throw away the header
    for row in c:
      vd_n[i] += [float(row[0])]
      i_n[i] += [(1 if i==0 else -1) * float(row[1])] 

vg_n = np.array([0.46, 0.56, 5])
# a = np.power(np.log(1 + np.exp(kappa*(vg_n-vt0)/(2*ut))), 2) / np.power(np.log(1 + np.exp(
a = 100
ut = 25.7 / 1000 # roughly, at room temperature
Vt0 = 0.551129 # from exp1
kappa = 0.700688 # from exp1
vdsat_n = kappa*(vg_n-Vt0) - 2 * ut * np.log(np.power(1+np.exp(kappa*(vg_n-Vt0)/(2*ut)), 1/np.sqrt(a)) - 1)
print(vdsat_n)





# Plot n-type characteristics
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)

for i in range(3):
  ax.semilogy(vd_n[i], i_n[i], ['r.', 'g.', 'b.'][i], label="Gate voltage = %s" % (["0.46 v", "0.56 v", "5 v"][i]))

plt.title("N-type drain characteristics")
plt.xlabel("Drain voltage (v)")
plt.ylabel("Current (A)")
plt.grid(True)
ax.legend()
plt.savefig("exp3_drainchars_n.pdf")
