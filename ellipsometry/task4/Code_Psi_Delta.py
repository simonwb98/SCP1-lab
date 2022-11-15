# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:31:36 2022

@author: User
"""

import numpy as np
# import math
import matplotlib.pyplot as plt
from pylab import figure, show, legend, xlabel,  ylabel, title, savefig

file = '201207_Task4_cZnO_IRSE_measured&fittedPsi&Delta.txt'

def get_data(input_file):
    with open(input_file, 'r') as data_file:
        wavenumber = []
        psi = []
        delta = []
        model_psi = []
        model_delta = []
        data_file.readline()
        data_file.readline()
        data_file.readline()
        data_file.readline()
        for line in data_file:
            a, b, c, d, e, f = line.split('\t')
            wavenumber.append(float(a))
            delta.append(180 + float(b))
            psi.append(float(c))
            model_delta.append(180 + float(d))
            model_psi.append(float(e))
    return (wavenumber, psi, delta, model_psi, model_delta)

def plot_data(input_file):
    wavenumber, psi, delta, model_psi, model_delta = map(np.array, get_data(input_file))
    color = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#EF4026"]
    # plt.plot(angle, psi, color[1], label= r'$\Psi$', linewidth=1)
    # plt.plot(angle, delta, color[2], label=r'$\Delta$', linewidth=1)
    # plt.plot(angle, model_psi, color[1], linestyle='--', label=r'$\Psi_{model}$', linewidth=1)
    # plt.plot(angle, model_delta, color[2], linestyle='--', label=r'$\Delta_{model}$', linewidth=1)
    
    
 
    # create the general figure
    fig1 = figure()
     
    # and the first axes using subplot populated with data 
    ax1 = fig1.add_subplot(111)
    
    line1, = ax1.plot(wavenumber, psi, color[1])
    line3, = ax1.plot(wavenumber, model_psi, color[1], linestyle='--')
    ylabel(r'$\Psi (^{\circ})$')
     
    # now, the second axes that shares the x-axis with the ax1
    ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
    line2, = ax2.plot(wavenumber, delta, color[2])
    line4, = ax2.plot(wavenumber, model_delta, color[2], linestyle='--')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ylabel('$\Delta (^{\circ})$')
    title(r'$ZnO$ - $\theta=70^{\circ}$')
    # for the legend, remember that we used two different axes so, we need 
    # to build the legend manually
    legend([line1, line2, line3, line4], [r'$\Psi$', r'$\Delta$', r'$\Psi_{model}$', r'$\Delta_{model}$'], loc='upper right')
    xlabel(r'$\tilde{\nu} (cm^{-1})$')
    # plt.xlim(0,2000)
    plt.grid()
    savefig('ZnO', dpi=200)
    show()
    
    # plt.xlabel(r'$\theta (^{\circ})$')
    # plt.ylabel(r'$\Psi (^{\circ})$')
    
    
    # plt.title('$Al_2O_3$ - $\Psi$ and $\Delta$ at $E \cong 2$ eV')
    # plt.grid()
    # plt.legend()
    # plt.show()
plot_data(file)