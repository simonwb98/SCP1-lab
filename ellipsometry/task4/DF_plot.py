# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:23:41 2022

@author: User
"""

import cmath
import matplotlib.pyplot as plt
from pylab import figure, show, legend, xlabel,  ylabel, title, savefig
import numpy as np
from scipy.optimize import curve_fit

color = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#EF4026"]

#phonon modes (in cm^-1) and damping G, A - perpendicular phonons
ATO = 347
ETO = 410
ALO = 570
ELO = 589
G = 14
epsilon = 3.7

def Lorentz(E, A, B1, B2, G):
    output = []
    for E in E:
        output.append(float(A*(B1**2-E**2+1j*E*G/(B2**2-E**2+1j*E*G))))
    return output

def real(wavenumber, EL, ET, GL, GT):
    output = []
    for v in wavenumber:
        output.append(float(((EL**2-v**2)*(ET**2-v**2)+v**2*GL*GT)/((ET**2-v**2)**2+(v*GT)**2)))
    return output

def imaginary(wavenumber, EL, ET, GL, GT):
    output = []
    for v in wavenumber:
        output.append(float(((EL**2-v**2)*v*GT-(ET**2-v**2)*v*GL)/((ET**2-v**2)**2+(v*GT)**2)))
    return output

energy = np.linspace(0, 1000, 1000)
plt.plot(energy, real(energy, ELO, ETO, G, G), color[0], label=r'$\Re{\epsilon_{\parallel}}$')
plt.plot(energy, imaginary(energy, ELO, ETO, G, G), color[1], label=r'$\Im{\epsilon_{\parallel}}$')
plt.plot(energy, real(energy, ALO, ATO, G, G), color[2], label=r'$\Re{\epsilon_{\bot}}$')
plt.plot(energy, imaginary(energy, ALO, ATO, G, G), color[3], label=r'$\Im{\epsilon_{\bot}}$')
plt.legend()
plt.grid()
plt.xlabel(r'$\tilde{\nu} (cm^{-1})$')
plt.ylabel(r'$\epsilon$')
plt.xlim(150, 800)
plt.title('$ZnO$ - Dielectric Tensor components')
plt.savefig('DF_ZnO.png', dpi=200)