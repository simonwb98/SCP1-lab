# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 18:43:27 2021

@author: User
"""
import numpy as np
# import math
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
import scipy.optimize as opt

def get_data(input_file):
    with open(input_file, 'r') as data_file:
        temperature = []
        mobility = []
        data_file.readline()
        for line in data_file:
            try:
                a, b, c, d = line.split('\t')
                temperature.append(float(a))
                mobility.append(float(b))
            except ValueError:
                pass
        return (temperature, mobility)

file = 'Praktikum_µ(T)__n(T)__ZnO-Einkristall.txt'

def ionized_impurity_scattering(T, A, B):
    return np.float64(A)*(B*T)**(3/2)
def polar_optical_scattering(T, B):
    return (B*T)**(-3/2)
def piezoelectric_scattering(T, B):
    return (B*T)**(-1/2)

def plot_data(input_file):
    '''
    Parameters
    ----------
    input_file : file, which stores the mobility - temperature data.
    Plots the mobility as a function of temperature and fits various
    scattering mechanisms. The range for which the scattering fits were 
    obtained were eyeballed for the specific input_file. The plot is on a 
    bilogarithmic graph with canonical tick marks. 

    Returns
    -------
    None.

    '''
    temperature, mobility = get_data(input_file)
    temperature = np.array(temperature)
    mobility = np.array(mobility)
    plt.plot(temperature, mobility, 'ko', label=r'$\mu(T)$')
    #implement various fits:
    fit_function1 = ionized_impurity_scattering
    fit_function2 = polar_optical_scattering
    fit_function3 = piezoelectric_scattering
    #for fitting function, choose equal length lists to be plotted against
    temp_range1 = [e for e in temperature if e <= 37]
    mob_range1 = mobility[0:len(temp_range1)]
    temp_range2 = [e for e in temperature if e > 170]
    start = np.where(temperature == temp_range2[0])
    end = np.where(temperature == temp_range2[-1])
    mob_range2 = mobility[start[0][0]:end[0][0] + 1]
    start2 = np.where(temperature == 50)
    end2 = np.where(temperature == 125.1)
    temp_range3 = temperature[start2[0][0]:end2[0][0]]
    mob_range3 = mobility[start2[0][0]:end2[0][0]]
    #fit parameters and plot
    popt1, pcov1 = opt.curve_fit(fit_function1, temp_range1, mob_range1)
    popt2, pcov2 = opt.curve_fit(fit_function2, temp_range2, mob_range2)
    popt3, pcov3 = opt.curve_fit(fit_function3, temp_range3, mob_range3)
    plt.plot(temperature, fit_function1(*popt1, temperature), 'g--', label='$\propto T^{3/2}$ - ionized impurity scattering')
    plt.plot(temperature, fit_function3(*popt3, temperature), 'y--', label='$\propto T^{-1/2}$ - piezoelectric scattering')
    plt.plot(temperature, fit_function2(*popt2, temperature), 'r--', label='$\propto T^{-3/2}$ - deformation potential scattering')
    # print(*popt1, *popt2, *popt3)
    #plot details
    plt.xlabel(r'$T(K)$')
    plt.xscale('log')
    plt.xticks([10, 50, 100, 200, 300], [10, 50, 100, 200, 300])
    plt.yscale('log')
    plt.yticks([10e1, 10e2],[r'$10^2$',  r'$10^3$'])
    # plt.xlim(200, 320)
    plt.ylim(0, 8e3)
    plt.ylabel(r'$\mu(cm^2/Vs)$')
    plt.title('Temperature dependence of mobility in ZnO bulk single crystal')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    

plot_data(file)

# Task 3

    
    
    
    
    