# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:05:50 2022

@author: User
"""

#A8 (Semiconductor Labs) - Code for plotting

import numpy as np
# import math
import matplotlib.pyplot as plt

file = 'YSZ_measurement_data_plus_model.txt'
color = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#EF4026"]

def get_data(input_file):
    with open(input_file, 'r') as data_file:
        energy = []
        psi1, psi2, psi3 = [], [], []
        delta1, delta2, delta3 = [], [], []
        model_psi1, model_psi2, model_psi3 = [], [], []
        model_delta1, model_delta2, model_delta3 = [], [], []
        data_file.readline()
        data_file.readline()
        for line in data_file:
            a, b, c, d, e, f, g, h, i, j, k, l, m, n = line.split('\t') 
            energy.append(float(a))
            psi1.append(float(b))
            delta1.append(float(c))
            model_psi1.append(float(d)) 
            model_delta1.append(float(e))
            psi2.append(float(f))
            delta2.append(float(g))
            model_psi2.append(float(h))
            model_delta2.append(float(i))
            psi3.append(float(j))
            delta3.append(float(k))
            model_psi3.append(float(l))
            model_delta3.append(float(m))
            
    return (energy, psi1, psi2, psi3, delta1, delta2, delta3, model_psi1, model_psi2, model_psi3, model_delta1, model_delta2, model_delta3)
    
def plot_psi(input_file):
    energy, psi1, psi2, psi3, delta1, delta2, delta3, model_psi1, model_psi2, model_psi3, model_delta1, model_delta2, model_delta3 = get_data(input_file)
    energy = np.array(energy)
    psi1, psi2, psi3 = np.array(psi1), np.array(psi2), np.array(psi3)
    model_psi1, model_psi2, model_psi3 = np.array(model_psi1), np.array(model_psi2), np.array(model_psi3)
    print(psi1, psi2, psi3)
    plt.plot(energy, psi1, color[1], label= r'$\theta = 60^{\circ}$', linewidth=1)
    plt.plot(energy, psi2, color[2], label=r'$\theta = 65^{\circ}$', linewidth=1)
    plt.plot(energy, psi3, color[0], label=r'$\theta = 70^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi1, color[1], linestyle = '--', label=r'$\theta_{model} = 60^{\circ}$')
    plt.plot(energy, model_psi2, color[2], linestyle = '--', label=r'$\theta_{model} = 65^{\circ}$')
    plt.plot(energy, model_psi3, color[0], linestyle = '--', label=r'$\theta_{model} = 70^{\circ}$')
    plt.xlabel(r'$h\nu (eV)$')
    plt.ylabel(r'$\Psi (^{\circ})$')
    plt.title('Single layer - $\Psi$')
    plt.grid()
    lgd = plt.legend(ncol = 2, loc='upper left', handleheight=2.0)
    # plt.xlim(0.0, 4.0)
    # plt.ylim(0.0, 16.0)
    # plt.xticks([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    # plt.yticks([4, 8, 12, 16], [4, 8, 12, 16])
    plt.savefig('single_layer_psi', dpi=200)
    
    plt.show()
    
def plot_delta(input_file):
    energy, psi1, psi2, psi3, delta1, delta2, delta3, model_psi1, model_psi2, model_psi3, model_delta1, model_delta2, model_delta3 = get_data(input_file)
    energy = np.array(energy)
    delta1, delta2, delta3 = np.array(delta1), np.array(delta2), np.array(delta3)
    model_delta1, model_delta2, model_delta3 = np.array(model_delta1), np.array(model_delta2), np.array(model_delta3)
    plt.plot(energy, delta1, color[1], label= r'$\theta = 60^{\circ}$', linewidth=1)
    plt.plot(energy, delta2, color[2], label=r'$\theta = 65^{\circ}$', linewidth=1)
    plt.plot(energy, delta3, color[0], label=r'$\theta = 70^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta1, color[1], linestyle = '--', label=r'$\theta_{model} = 60^{\circ}$')
    plt.plot(energy, model_delta2, color[2], linestyle = '--', label=r'$\theta_{model} = 65^{\circ}$')
    plt.plot(energy, model_delta3, color[0], linestyle = '--', label=r'$\theta_{model} = 70^{\circ}$')
    plt.xlabel(r'$h\nu (eV)$')
    plt.ylabel(r'$\Delta (^{\circ})$')
    plt.title('Single layer - $\Delta$')
    plt.grid()
    plt.legend(ncol=2, loc='center', bbox_to_anchor=(0.5, -0.3))
    plt.savefig('single_layer_delta', dpi=200)
    plt.show()
    
plot_psi(file)
plot_delta(file)    
    
    
    