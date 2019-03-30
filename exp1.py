#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from ekvfit import ekvfit

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
  return list(zip(*pairs))

vg, isat = clip_range(vg, isat, (0, np.inf))

Is, Vt, kappa = ekvfit(np.array(vg), np.array(isat), plotting='on')

# From ekvfit
def ekv(vg):
  return Is * np.power(np.log(1 + np.exp(kappa*(vg - Vt)/(2*0.0258))), 2)

fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)

ax.semilogy(vg, isat, 'b.', label="Saturation current (experimental)")
ax.semilogy(vg, [ekv(v) for v in vg], 'g-', label="Saturation current (theoretical)")

plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.grid(True)
ax.legend()
plt.show()
