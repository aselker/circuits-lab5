#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#def weak_inversion(Vs, S, I0, K, Ut):
#    return S*I0*np.exp((K*5-Vs)/Ut)

def weak_inversion(Vs, Is, K, Vt0, Ut):
    #return Is*np.exp((K*(Vg-Vt0)-Vs)/Ut)*(1-np.exp(-(Vg-Vs)/Ut))
    #return Is*np.exp((K*(Vg-Vt0)-Vs)/Ut)
    Ut = .0257
    return np.log(1+Is*(np.exp((K*(5-Vt0)-Vs)/Ut)-np.exp((K*(5-Vt0)-5)/Ut)))

def make_plots(Vs, I, nmos=True):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.semilogy(Vs,I, 'b.')
    params = curve_fit(weak_inversion, Vs[700:], np.log(1+np.array(I[700:])))#, p0=[7.22e-8,3.4,.5,.0257])
    #params = [[7.22e-8,3.4,.5,.0257]]
    #ax.plot(Vs[650:750],weak_inversion(Vs[650:750],*params[0]), '-k', label="Theoretical Fit (S={!s}, I0={!s}, Ut={!s}, k={!s})".format(params[0][0],params[0][1],params[0][2],params[0][3]))
    print(params)
    theor_x = np.array(Vs[650:])
    #print(theor_x)
    theor_y = np.exp(weak_inversion(theor_x,*params[0]))-1#7.22e-8,3.4,.5,.0257)#

    #print(theor_y)
    ax.plot(theor_x,theor_y, '-k', label="Theoretical Fit (Is={!s}, K={!s}, Vt0={!s}, Ut={!s})".format(params[0][0],params[0][1],params[0][2],params[0][3]))
    plt.xlabel("Source Voltage (V)")
    plt.ylabel("Current (A)")
    plt.legend()
    plt.ylim(top=0.007)
    plt.ylim(bottom=1e-8)
    plt.show()

    plt.figure()
    plt.subplot(111).plot(I[1:], np.divide(np.diff(I),np.diff(Vs)), '.b')
    #plt.show()

with open('data/exp2_nmos.csv') as f:
    c = csv.reader(f, delimiter=",")
    next(c)

    Vs = []
    I = []
    for row in c:
        Vs +=[float(row[0])]
        I  +=[float(row[1])]

    make_plots(Vs,I)
