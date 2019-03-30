#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt

V = []
Vg = []
I = []

with open('data/exp3_nmos_0.46.csv') as f:
    c = csv.reader(f, delimiter=",")
    next(c)
    
    for row in c:
        Vg+= [float(row[0])]
        I += [float(row[1])]
        
fig = plt.figure()
ax = plt.subplot(111)



ax.semilogy(Vg, I, 'b.')
plt.title("Current")
ax.legend()
plt.show()
