# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:32:23 2022

@author: User
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt

file = 'Problem_3.txt'

def get_corrected_data(input_file):
    mobility = []
    temperature = []
    concentration = []
    with open(input_file, 'r') as file:
        file.readline()
        for line in file:
            try:
                a, b, c = line.split('\t')
                temperature.append(float(a))
                concentration.append(float(b))
                mobility.append(float(c))
            except ValueError:
                pass
    def correct_mob(conc, mob):
        sheet_mob = np.array(mob[0])
        sheet_conc = np.array(conc[0])
        res = [sheet_mob]
        mob = mob[1:]
        conc = conc[1:]
        numerator = []
        denominator = []
        for i in range(len(mob)):
            numerator.append(mob[i]**2*conc[i] - sheet_mob**2*sheet_conc)
            denominator.append(mob[i]*conc[i] - sheet_mob*sheet_conc)
        numerator, denominator = np.array(numerator), np.array(denominator)
        res.append(list(numerator/denominator))
        return res
    def correct_conc(conc, mob):
        sheet_mob = np.array(mob[0])
        sheet_conc = np.array(conc[0])
        res = [float(sheet_conc)]
        mob = mob[1:]
        conc = conc[1:]
        numerator = []
        denominator = []
        for i in range(len(mob)):
            numerator.append((mob[i]*conc[i] - float(sheet_mob)*float(sheet_conc))**2)
            denominator.append(mob[i]**2*conc[i] - sheet_mob**2*sheet_conc)
        numerator, denominator = np.array(numerator), np.array(denominator)
        res.append(list(numerator/denominator))
        result = []
        for sublist in res:
            if type(sublist) == list:
                for element in sublist:
                    result.append(element)
            else:
                result.append(sublist)
        result.insert(-1, result[0])
        result.pop(0)
        return result
    rev_temp = temperature[::-1]
    rev_conc = correct_conc(concentration, mobility)[::-1]
    return (rev_temp, rev_conc)

def fit_function(T, A, B):
    return A*(T**B)

def plot_correct_conc(input_file):
    temp, conc = get_corrected_data(input_file)
    
    x_axis = []
    y_axis = []
    z_axis = []
    
    for i in range(len(temp)):
        x_axis.append(1/temp[i])
        z_axis.append(conc[i]*temp[i]**(-3/4))
        y_axis.append(math.log(conc[i]*temp[i]**(-3/4)))        
    plt.plot(x_axis, y_axis, label = 'Plot of' + r' $\ln(n/T^{3/4})$ vs. 1/T')
    plt.xlabel(r'$1/T (1/K)$')
    plt.ylabel(r'$ln\left(\frac{n}{T^{3/4}}\right)$')
    x_axis_c = x_axis[:]
    x_axis = np.array(x_axis[15:25])
    y_axis = np.array(y_axis[15:25])
    fit = np.polyfit(x_axis, y_axis, 1)
    predicted = np.polyval(fit, x_axis)
    plt.plot(x_axis, predicted, 'k:', label = 'linear fit with slope ' + str(round(fit[0], 2)) + ' K')
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlim(0.0033, 0.05)
    plt.title('Estimating the Donator energy level')
    plt.suptitle('$ZnO$ thin film')
    plt.clf()
    plt.plot(temp, z_axis, label = '$n$ vs $T$')
    plt.title(r'Carrier Concentration for low temperatures')
    plt.suptitle(r'$ZnO$ thin film')
    plt.xlabel(r'T(K)')
    plt.ylabel(r'$n (cm^{-3})$')
    plt.xlim(40, 150)
    plt.grid()
    plt.yscale('log')
    
    temp = temp[::-1]
    conc = conc[::-1]
    
    temp_range = temp[6:16]
    conc_range = z_axis[6:16]
    popt, pcov = opt.curve_fit(fit_function, temp_range, conc_range)
    plt.plot(temp, 4.13e9*np.array(temp)**(2.91), label=r'Fit $AT^B$ with $B = 2.91$')
    # plt.plot(temp, fit_function(*popt, temp), 'c--', label=r'exponential curve fit')
    plt.legend()
plot_correct_conc(file)