# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:34:37 2021

@author: User
"""

import numpy as np
import math
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
import scipy.optimize as opt
import scipy.stats

def get_data(input_file):
    with open(input_file, 'r') as data_file:
        temperature = []
        concentration = []
        mobility = []
        data_file.readline()
        for line in data_file:
            try:
                a, b, c = line.split('\t')
                temperature.append(float(a))
                concentration.append(float(b))
                mobility.append(float(c))
            except ValueError:
                pass
        return (temperature, concentration, mobility)
    
def get_corrected_data(input_file):
    with open(input_file, 'r') as data_file:
        temperature = []
        concentration = []
        mobility = []
        data_file.readline()
        for line in data_file:
            try:
                a, b, c = line.split('\t')
                temperature.append(float(a))
                concentration.append(float(b))
                mobility.append(float(c))
            except ValueError:
                pass
        return (temperature[::-1], model_concentration(mobility, concentration)[::-1], mobility)
    
file = 'Problem_3.txt'

def model_mobility(mobility, concentration):
    '''
    

    Parameters
    ----------
    mobility : np.array of length N
    concentration : np.array of length N

    Returns
    -------
    np.array for the corrected mobilities

    '''
    sheet_mob = mobility[0]
    sheet_conc = concentration[0]
    numerator = mobility**2*concentration - sheet_mob**2*sheet_conc
    denominator = mobility*concentration - sheet_mob*sheet_conc
    return numerator/denominator

def model_concentration(mobility, concentration):
    sheet_mob = mobility[0]
    sheet_conc = concentration[0]
    res = np.array([sheet_conc])
    mobility = mobility[1:]
    concentration = concentration[1:]
    numerator = (mobility*concentration - sheet_mob*sheet_conc)**2
    denominator = mobility**2*concentration - sheet_mob**2*sheet_conc
    res = res + numerator/denominator
    return res
    

def plot_mobility(input_file):
    temperature, concentration, mobility = get_data(input_file)
    temperature, mobility = np.array(temperature), np.array(mobility)
    plt.plot(temperature, mobility, 'y.-', label='uncorrected mobility $\mu_H$')
    plt.plot(temperature, model_mobility(mobility, concentration), 'r.-', label='corrected mobility $\mu_H$')
    plt.xlabel(r'$T (K)$')
    plt.ylabel(r'$\mu (cm^2/Vs)$')
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks([50, 100, 200, 300], [50, 100, 200, 300])
    plt.title('Comparison of corrected to uncorrected mobilities in PLD ZnO film')
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_concentration(input_file):
    plt.clf()
    temperature, concentration, mobility = get_data(input_file)
    temperature, mobility = np.array(temperature), np.array(mobility)
    rec_temperature = 1/temperature
    plt.plot(rec_temperature, concentration, 'y.-', label='uncorrected concentration $n(T)$')
    plt.plot(rec_temperature, model_concentration(mobility, concentration), 'r.-', label='corrected concentration $n(T)$')
    plt.xlabel(r'$10^3/T$ $ (K^{-1})$')
    plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel(r'$n (cm^{-3})$')
    plt.title('Comparison of corrected to uncorrected concentrations in PLD ZnO film')
    plt.legend()
    plt.grid()
    plt.show()
#     fig = plt.figure()
#     my_file = '\corrected_conc.png'
#     os.chdir(save_in_file)
#     fig.savefig(os.path.join(save_in_file, my_file))  

def ionized_impurity_scattering(T, A, B):
    return np.float64(A)*(B*T)**(3/2)
def polar_optical_scattering(T, B):
    return (np.float64(B)*T)**(-3/2)
def piezoelectric_scattering(T, B):
    return (np.float64(B)*T)**(-1/2)
def grain_boundary_scattering(T, A, B, C, D):
    return np.float64(A)*(C*T)**(-1/2)*D*np.exp(-np.float64(B)/T)
def unknown_scattering(T, A, B):
    return np.float64(A)*(B*T**(5/2))

def plot_mobility_fit(input_file):
    plt.clf()
    temperature, concentration, mobility = get_data(input_file)
    temperature, mobility = np.array(temperature), np.array(mobility)
    
    plt.plot(temperature[2:], model_mobility(mobility, concentration)[2:], 'b.', label='corrected mobility $\mu_H$')

    plt.plot()
    mobility = model_mobility(mobility, concentration)[2:]
    temperature = temperature[2:]
    print(mobility)
    print(temperature)
    #implement various fits:
    fit_function1 = ionized_impurity_scattering
    fit_function2 = polar_optical_scattering
    fit_function3 = piezoelectric_scattering
    #for fitting function, choose equal length lists to be plotted against
    #for ionized impurity scattering
    temp_range1 = [e for e in temperature if e <= 50]
    mob_range1 = mobility[0:len(temp_range1)]
    #for polar optical scattering
    temp_range2 = [e for e in temperature if e > 200]
    start = np.where(temperature == temp_range2[0])
    end = np.where(temperature == temp_range2[-1])
    mob_range2 = mobility[start[0][0]:end[0][0] + 1]
    #for piezoelectric potential scattering
    start2 = np.where(temperature == 50.01)
    end2 = np.where(temperature == 125.0)
    temp_range3 = temperature[start2[0][0]:end2[0][0]]
    mob_range3 = mobility[start2[0][0]:end2[0][0]]
    #for unkown scattering
    temp_range5 = [e for e in temperature if e < 83]
    start5 = np.where(temperature == temp_range5[0])
    end5 = np.where(temperature == temp_range5[-1])
    mob_range5 = mobility[start5[0][0]:end5[0][0] + 1]
    #importing fit parameters of bulk probe:
    # popt1, popt2, popt3 = ((1.535641997475128, 1.8242491677016217), 9.91330110608591e-05, 1.41991286382567e-08)
    #fit parameters and plot

    popt1, pcov1 = opt.curve_fit(fit_function1, temp_range1, mob_range1)
    popt2, pcov2 = opt.curve_fit(fit_function2, temp_range2, mob_range2)
    popt3, pcov3 = opt.curve_fit(fit_function3, temp_range3, mob_range3)
    popt4, pcov4 = opt.curve_fit(grain_boundary_scattering, temperature, mobility)
    popt5, pcov5 = opt.curve_fit(unknown_scattering, temp_range5, mob_range5)

    # plt.plot(temperature, fit_function1(*popt1, temperature), 'g--', label='$\propto T^{3/2}$ - ionized impurity scattering')
    # plt.plot(temperature, fit_function3(*popt3, temperature), 'y--', label='$\propto T^{-1/2}$ - piezoelectric scattering')
    plt.plot(temperature, fit_function2(*popt2, temperature), 'r--', label='$\propto T^{-3/2}$ - polar optical scattering')
    # plt.plot(temperature, grain_boundary_scattering(*popt4, temperature), 'm--', label='grain boundary scattering')
    plt.plot(temperature, unknown_scattering(*popt5, temperature), 'c--')#, label='$\propto T^{%1.2f}$' %popt5[1])
    #plot details
    plt.xlabel(r'$T(K)$')
    plt.xscale('log')
    plt.xticks([10, 50, 100, 200, 300], [10, 50, 100, 200, 300])
    plt.yscale('log')
    plt.yticks([10e1, 10e2],[r'$10^2$',  r'$10^3$'])
    # plt.xlim(200, 320)
    # plt.ylim(0, 8e3)
    plt.ylabel(r'$\mu(cm^2/Vs)$')
    plt.title('Temperature dependence of mobility in ZnO bulk single crystal')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    
def logistic(x, L, k, b, c):
    return L / (1 + b*np.exp(k/x)) - abs(c)

def plot_concentration_fit(input_file):
    plt.clf()
    temperature, concentration, mobility = get_data(input_file)
    temperature, concentration, mobility = np.array(temperature), np.array(concentration), np.array(mobility)
    norm_conc = model_concentration(mobility, concentration)/max(model_concentration(mobility, concentration))
    
    plt.plot(temperature[1:], norm_conc, 'o', label='corrected concentration $n(T)$')
    popt, pcov = opt.curve_fit(logistic, temperature[1:], norm_conc, method='trf')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(norm_conc, logistic(temperature[1:], *popt))
    # plt.plot(temperature, logistic(temperature, *popt), '-', label='logistic curve fit, $R^2=%1.2f$' %float(math.floor(r_value**2*100)/100.0))
    print(model_concentration(mobility, concentration), popt)
    print(popt[0]*max(model_concentration(mobility, concentration)))
    # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # ax.plot(temperature, model_concentration(mobility, concentration), '-')
    # ax.plot(temperature, logistic(temperature, *popt), 'g-')
    plt.xlabel(r'$T (K)$')
    plt.ylabel(r'normalized $n$')
    plt.title('Carrier Concentration in ZnO Thin Film')
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_data2(input_file):
    temp, conc, mobility = get_data(input_file)
    x_axis = []
    y_axis = []
    temp = np.array(temp[::-1])
    conc = np.array(conc[::-1])
    mobility = np.array(mobility)
    model = list(model_concentration(mobility, conc))
    for i in range(len(temp) - 1):
        x_axis.append(1/temp[i])
        
        y_axis.append(math.log(model[i]*temp[i]**(-3/4)))
        print(y_axis)
    plt.plot(x_axis, y_axis, label = 'Plot of' + r' $\ln(n/T^{3/4})$ vs. 1/T')
    plt.xlabel(r'$1/T (1/K)$')
    plt.ylabel(r'$ln\left(\frac{n}{T^{3/4}}\right)$')
    x_axis = np.array(x_axis[:])
    y_axis = np.array(y_axis[:])
    fit = np.polyfit(x_axis, y_axis, 1)
    predicted = np.polyval(fit, x_axis)
    plt.plot(x_axis, predicted, 'k:', label = 'linear fit with slope ' + str(round(fit[0], 2)) + ' K')
    plt.legend()
    plt.title('Estimating the Donator energy level')
    
def plot_donator_energy(input_file):
    temp, conc, mobility = get_corrected_data(input_file)
    temp, conc = np.array(temp), np.array(conc)
    x_axis = 1/temp
    y_axis = np.log(conc*temp**(-3/4))
    x_ax = []
    y_ax = []
    for i in range(len(temp) - 1):
        x_ax.append(1000/temp[i])
        y_ax.append(math.log())
    plt.plot(x_axis, y_axis, label='Plot of ' + r'$\ln ((n/T^{3/4})$ vs. 1/T')
    plt.xlabel(r'$1/T (1/K)$')
# plot_mobility(file)
# plot_concentration(file)
# plot_mobility_fit(file)
plot_concentration_fit(file)
# plot_data2(file)
# plot_donator_energy(file)
