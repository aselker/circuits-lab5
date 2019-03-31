#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

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
    

# Threshold voltages -- do we need these?
vt_n, vt_p = 0.56, 4.34

# Read data files
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

vd_p = [[], [], []]
i_p = [[], [], []]

names = ["data/exp3_pmos_4.44.csv", "data/exp3_pmos_4.34.csv", "data/exp3_pmos_0.csv"]
for i in range(3):
  with open(names[i]) as f:
    c = csv.reader(f, delimiter=",")
    next(c) # Throw away the header
    for row in c:
      vd_p[i] += [float(row[0])]
      i_p[i] += [-float(row[1])] 



# Find sat currents and voltages
a = 100
ut = 25.7 / 1000 # roughly, at room temperature

vg_n = np.array([0.46, 0.56, 5])
Vt0_n = 0.551129 # from exp1
kappa_n = 0.700688 # from exp1
def vdsat_nf(vg): return kappa_n*(vg-Vt0_n) - 2 * ut * np.log(np.power(1+np.exp(kappa_n*(vg-Vt0_n)/(2*ut)), 1/np.sqrt(a)) - 1)
vdsat_n = vdsat_nf(vg_n)
print("Saturation start voltages (n-type):", vdsat_n)
isat_n = [linterp(vd, i, vdsat) for vd, i, vdsat in zip(vd_n, i_n, vdsat_n)]
print("Saturation start currents (n-type):", isat_n)

vg_p = np.array([4.44, 4.34, 0])
Vt0_p = 0.59509 # from exp1
kappa_p = 0.746349 # from exp1
def vdsat_pf(vg): 
  this_vg = 5-vg
  return 5-(kappa_p*(this_vg-Vt0_p) - 2 * ut * np.log(np.power(1+np.exp(kappa_p*(this_vg-Vt0_p)/(2*ut)), 1/np.sqrt(a)) - 1))
vdsat_p = vdsat_pf(vg_p)
print("Saturation start voltages (p-type):", vdsat_p)
isat_p = [linterp(vd, i, vdsat) for vd, i, vdsat in zip(vd_p, i_p, vdsat_p)]
print("Saturation start currents (p-type):", isat_p)


# Split the data into ohmic and saturation regions
grouped = [[list(g) for _, g in groupby(zip(vd, i), lambda pair: pair[0] >= vdsat)] for vd, i, vdsat in zip(vd_n, i_n, vdsat_n)]
vd_on = [[x[0] for x in y] for y in [grouped[i][0] for i in range(3)]]
i_on = [[x[1] for x in y] for y in [grouped[i][0] for i in range(3)]]
vd_sn = [[x[0] for x in y] for y in [grouped[i][1] for i in range(3)]]
i_sn = [[x[1] for x in y] for y in [grouped[i][1] for i in range(3)]]

grouped = [[list(g) for _, g in groupby(zip(vd, i), lambda pair: pair[0] >= vdsat)] for vd, i, vdsat in zip(vd_p, i_p, vdsat_p)]
vd_op = [[x[0] for x in y] for y in [grouped[i][0] for i in range(3)]]
i_op = [[x[1] for x in y] for y in [grouped[i][0] for i in range(3)]]
vd_sp = [[x[0] for x in y] for y in [grouped[i][1] for i in range(3)]]
i_sp = [[x[1] for x in y] for y in [grouped[i][1] for i in range(3)]]

# Linear fit the saturation region, to find Early voltage and r0
early_n = []
r0_n = []
for vd, i in zip(vd_sn, i_sn):
  fit = np.polyfit(vd, i, 1)
  early_n += [(fit[1] / fit[0])]
  r0_n += [1/fit[1]]

early_p = []
r0_p = []
for vd, i in zip(vd_sp, i_sp):
  fit = np.polyfit(vd, i, 1)
  early_p += [-(fit[1] / fit[0])]
  r0_p += [1/fit[1]]

# Linear fit part of the ohmic region, to find gs
gs_n = []
for vd, i in zip(vd_on, i_on):
  num_to_use = len(vd)//3
  vd = vd[:num_to_use]
  i = i[:num_to_use]
  fit = np.polyfit(vd, i, 1)
  gs_n += [fit[0]]

gs_p = []
for vd, i in zip(vd_op, i_op):
  num_to_use = len(vd)//3
  vd = vd[:num_to_use]
  i = i[:num_to_use]
  fit = np.polyfit(vd, i, 1)
  gs_p += [-fit[0]]

# Find intrinsic gains
ig_n = np.array(r0_n) * np.array(gs_n)
ig_p = np.array(r0_p) * np.array(gs_p)

print("n-type early voltages:", early_n, "\nr0:", r0_n, "\ngs:", gs_n)
print("p-type early voltages:", early_p, "\nr0:", r0_p, "\ngs:", gs_p)


# Plot n-type characteristics
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)

for i in range(3):
  ax.semilogy(vd_n[i], i_n[i], ['r.', 'g.', 'b.'][i], label="Gate voltage = %s" % (["0.46 V", "0.56 V", "5 V"][i]))
ax.semilogy(vdsat_n, isat_n, 'ko', label="Saturation points")

plt.title("N-type drain characteristics")
plt.xlabel("Drain voltage (v)")
plt.ylabel("Current (A)")
plt.grid(True)
ax.legend()
plt.savefig("exp3_drainchars_n.pdf")
plt.cla()

# Plot p-type characteristics
for i in range(3):
  ax.semilogy(vd_p[i], i_p[i], ['r.', 'g.', 'b.'][i], label="Gate voltage = %s" % (["4.44 V", "4.34 V", "0 V"][i]))
ax.semilogy(vdsat_p, isat_p, 'ko', label="Saturation points")

plt.title("P-type drain characteristics")
plt.xlabel("Drain voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
ax.legend()
plt.savefig("exp3_drainchars_p.pdf")
plt.cla()


# Plot both-types characteristics
ax.semilogx(isat_n, early_n, 'bo', label="N-type")
ax.semilogx(isat_p, early_p, 'go', label="P-type")
plt.title("Early voltages and saturation currents")
plt.xlabel("Saturation current (A)")
plt.ylabel("Early voltage (V)")
plt.grid(True)
ax.legend()
plt.savefig("exp3_early.pdf")
plt.cla()

ax.loglog(isat_n, ig_n, 'bo', label="N-type")
ax.loglog(isat_p, ig_p, 'go', label="P-type")
plt.title("Intrinsic gains and saturation currents")
plt.xlabel("Saturation current (A)")
plt.ylabel("Intrinsic gain")
plt.grid(True)
ax.legend()
plt.savefig("exp3_ig.pdf")
plt.cla()
