# -*- coding: utf-8 -*-
"""
Experimental Dielectric Function

"""

import cmath
import matplotlib.pyplot as plt
from pylab import figure, show, legend, xlabel,  ylabel, title, savefig
import numpy as np
from scipy.optimize import curve_fit

color = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#EF4026"]
input_file = '201207_Task3_AlGaAsonGaAs_DielectricFunction.txt'

def get_data(file):
    with open(file, 'r') as data_file:
        energy = []
        er = []
        ei = []
        for i in range(4):
            data_file.readline()
        for line in data_file:
            data = line.split('\t')
            energy.append(float(data[0]))
            er.append(float(data[1]))
            ei.append(float(data[2]))
        return (energy, er, ei)
    
def plot_data(file):
    energy, e1, e2 = get_data(file)
    # create the general figure
    fig1 = figure()
     
    # and the first axes using subplot populated with data 
    ax1 = fig1.add_subplot(111)
    
    line1, = ax1.plot(energy, e1, color[1])
    ylabel(r'$\epsilon_1$')
     
    # now, the second axes that shares the x-axis with the ax1
    ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
    line2, = ax2.plot(energy, e2, color[2])

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ylabel('$\epsilon_2$')
    title(r'Dielectric Function for AlGaAs')
    # for the legend, remember that we used two different axes so, we need 
    # to build the legend manually
    legend([line1, line2], [r'$\epsilon_1$', r'$\epsilon_2$'], loc='upper left')
    xlabel(r'$h\nu (eV)$')
    plt.grid()
    savefig('dielectric_function', dpi=200)
    show()
    
def Lorentz(E, A, B, C):
    return A**2*C*E/((B**2-E**2)**2+C**2*E**2)

def M_0(E, A, B, C):
    output = []
    for E in E:
        output.append(A*cmath.sqrt(B)*(2-cmath.sqrt(1+(E+1j*C)/B) - cmath.sqrt(1-(E+1j*C)/B))/((E+1j*C)**2))
    return output


def M_1(E, A, B, C):
    output = []
    for E in E:
        output.append(-A*B**2/((E+1j*C)**2)*cmath.log((B**2-(E+1j*C)**2)/B**2, cmath.e))
    return output

#parameters for manually fitting the MDF
A1, B1, C1 = 30, 4.85, 0.02
A2, B2, C2 = 6.8, 1.92, 0.001
A3, B3, C3 = 7, 3.1, 0.1

def plot_models(file):
    energy, e1, e2 = get_data(file)
    energy, e2 = np.array(energy), np.array(e2)
    plt.plot(energy, e2, color[1], label='calculated $\epsilon_2$')
    #popt, pcov = curve_fit(np.imag(M_1), energy, e2)
    plt.plot(energy, Lorentz(energy, A1, B1, C1), color[2], linestyle='--', label='Lorentz Oscillator')
    plt.plot(energy, np.imag(M_0(energy, A2, B2, C2)), color[0], linestyle='--', label='$M_0$')
    plt.plot(energy, np.imag(M_1(energy, A3, B3, C3)), color[4], linestyle='--', label='$M_1$')
    total = Lorentz(energy, A1, B1, C1) + np.imag(M_0(energy, A2, B2, C2))+np.imag(M_1(energy, A3, B3, C3))
    plt.grid()
    plt.plot(energy, total, color[5], label='Total')
    #plt.plot(energy, M_1(energy, *popt), 'g--')
    plt.ylabel('$\epsilon_2$')
    plt.xlabel(r'$h\nu$ (eV)')
    # plt.xlim(0.0, 4.0)
    plt.legend(handleheight=2.1)

    plt.savefig('DF', dpi=200)
    
    
#plot_data(input_file)
plot_models(input_file)