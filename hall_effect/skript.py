# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:49:12 2021

@author: User
"""
"""Plotting and Fitting script for the Semiconductor Lab A4: Hall effect"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt
import scipy.stats

#Problem 2: Plotting carrier concentration and mobility as a function of T

def get_data(input_file):
    with open(input_file, 'r') as data_file:
        conc = []
        temp = []
        data_file.readline() #ignore header
        for line in data_file:
            try:
                a, b, c, d = line.split('\t')
                if c == '':
                    pass
                else:
                    conc.append(float(d))
                    temp.append(float(c))
            except ValueError:
                pass
    return (temp, conc)

def logistic(x, L, k, b):
    return L / (1 + b*np.exp(-k/x))

def exponential(x, A, B):
    return A*np.exp(B/x)

def plot_data1(input_file):
    temp, conc = get_data(input_file)
    # plt.plot(temp, conc, label= 'Carrier Concentration')
    # plt.xlabel(r'T (K)')
    # plt.ylabel(r'n $(m^{-3})$')
    temp = np.array(temp, dtype='float64')
    conc = np.array(conc, dtype='float64')
    conc_norm = conc/max(conc)
    popt1, pcov = opt.curve_fit(logistic, temp, conc_norm, method="trf")
    popt2, pcov = opt.curve_fit(exponential, [e for e in temp if e < 27], conc_norm[36:], method="trf")
    y_fit1 = logistic(temp, *popt1)
    y_fit2 = exponential(temp, *popt2)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_fit1, conc_norm)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(temp, conc_norm, 'o', label=r'normalized $n (T)$')
    ax.plot(temp, y_fit1, '-', label='logistic curve fit, $R^2=$ %1.3f' %r_value**2)
    E_D = popt2[1]*2*1.38e-23*6.242e21
    # ax.plot(temp, y_fit2, '--', label='exponential curve fit, $E_1 = %1.3f meV$' %E_D)
    # print(*popt)
    # y_fit = logistic(temp, *popt)
    # plt.plot(temp, y_fit, label='logistic curve fit')
    plt.xlabel(r'$T(K)$')
    plt.ylabel(r'normalized $n(T)$')
    plt.legend()
    plt.grid()
    plt.title('Temperature dependence of concentration in ZnO bulk single crystal')
    print(popt1[0]*max(conc), std_err, r_value**2)

    

def plot_data2(input_file):
    temp, conc = get_data(input_file)
    x_axis = []
    y_axis = []
    for i in range(len(temp)):
        x_axis.append(1/temp[i])
        y_axis.append(math.log(conc[i]*temp[i]**(-3/4)))
    plt.plot(x_axis, y_axis, label = 'Plot of' + r' $\ln(n/T^{3/4})$ vs. 1/T')
    plt.xlabel(r'$1/T (1/K)$')
    plt.ylabel(r'$ln\left(\frac{n}{T^{3/4}}\right)$')
    x_axis = np.array(x_axis[20:])
    y_axis = np.array(y_axis[20:])
    fit = np.polyfit(x_axis, y_axis, 1)
    predicted = np.polyval(fit, x_axis)
    plt.plot(x_axis, predicted, 'k:', label = 'linear fit with slope ' + str(round(fit[0], 2)) + ' K')
    print(fit)
    plt.legend()
    plt.title('Estimating the Donator energy level')

    

plot_data2('Praktikum_µ(T)__n(T)__ZnO-Einkristall.txt')
# plot_data1('Praktikum_µ(T)__n(T)__ZnO-Einkristall.txt')
    
    