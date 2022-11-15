# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:05:50 2022

@author: User
"""

#A8 (Semiconductor Labs) - Code for plotting

import numpy as np
# import math
import matplotlib.pyplot as plt

file = 'AlGaAs.txt'
name = file.strip('.txt')

def get_data(input_file):
    with open(input_file, 'r') as data_file:
        energy = []
        psi1, psi2, psi3, psi4 = [], [], [], []
        delta1, delta2, delta3, delta4 = [], [], [], []
        model_psi1, model_psi2, model_psi3, model_psi4 = [], [], [], []
        model_delta1, model_delta2, model_delta3, model_delta4 = [], [], [], []
        data_file.readline()
        data_file.readline()
        data_file.readline()
        data_file.readline()
        for line in data_file:
            a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r = line.split('\t') 
            energy.append(float(a))
            psi1.append(float(b))
            delta1.append(float(c))
            psi2.append(float(d))
            delta2.append(float(e))
            psi3.append(float(f))
            delta3.append(float(g))
            psi4.append(float(h))
            delta4.append(float(i))
            model_psi1.append(float(j)) 
            model_delta1.append(float(k))
            model_psi2.append(float(l))
            model_delta2.append(float(m))
            model_psi3.append(float(n))
            model_delta3.append(float(o))
            model_psi4.append(float(p))
            model_delta4.append(float(q))
            
    return (energy, psi1, psi2, psi3, psi4, delta1, delta2, delta3, delta4, model_psi1, model_psi2, model_psi3, model_psi4, model_delta1, model_delta2, model_delta3, model_delta4)
    
def plot_psi(input_file):
    energy, psi1, psi2, psi3, psi4, delta1, delta2, delta3, delta4, model_psi1, model_psi2, model_psi3, model_psi4, model_delta1, model_delta2, model_delta3, model_delta4 = get_data(input_file)
    energy = np.array(energy)
    psi1, psi2, psi3 = np.array(psi1), np.array(psi2), np.array(psi3)
    model_psi1, model_psi2, model_psi3 = np.array(model_psi1), np.array(model_psi2), np.array(model_psi3)
    color = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#EF4026"]
    plt.plot(energy, psi1, color[1], label= r'$\theta = 70^{\circ}$', linewidth=1)
    plt.plot(energy, psi2, color[2], label=r'$\theta = 72^{\circ}$', linewidth=1)
    plt.plot(energy, psi3, color[0], label=r'$\theta = 74^{\circ}$', linewidth=1)
    plt.plot(energy, psi4, color[3], label=r'$\theta = 76^{\circ}$')
    plt.plot(energy, model_psi1, color[1], linestyle= '--', label=r'$\theta_{model} = 70^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi2, color[2], linestyle='--', label=r'$\theta_{model} = 72^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi3, color[0], linestyle='--', label=r'$\theta_{model} = 74^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi4, color[3], linestyle='--', label=r'$\theta_{model} = 76^{\circ}$', linewidth=1)
    plt.xlabel(r'$h\nu (eV)$')
    plt.ylabel(r'$\Psi (^{\circ})$')
    plt.title('$AlGaAs$ layer - $\Psi$')
    plt.grid()
    lgd = plt.legend(ncol = 2, loc='upper left', bbox_to_anchor=(0.5, -0.5), handleheight= 2.4)
    # plt.xlim(0.0, 4.0)
    # plt.ylim(0.0, 16.0)
    # plt.xticks([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    # plt.yticks([4, 8, 12, 16], [4, 8, 12, 16])
    
    plt.savefig('YSZ_Psi_' + name, dpi=200)
    def export_legend(legend, filename='legend_psi' + name + '.png', expand=[-5,-5,5,5]):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=200, bbox_inches=bbox)
    export_legend(lgd)
    plt.show()
    
def plot_delta(input_file):
    energy, psi1, psi2, psi3, psi4, delta1, delta2, delta3, delta4, model_psi1, model_psi2, model_psi3, model_psi4, model_delta1, model_delta2, model_delta3, model_delta4 = get_data(input_file)
    energy = np.array(energy)
    delta1, delta2, delta3 = np.array(delta1), np.array(delta2), np.array(delta3)
    model_delta1, model_delta2, model_delta3 = np.array(model_delta1), np.array(model_delta2), np.array(model_delta3)
    color = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#EF4026"]
    plt.plot(energy, delta1, color[1], label= r'$\theta = 70^{\circ}$', linewidth=1)
    plt.plot(energy, delta2, color[2], label=r'$\theta = ^{\circ}$', linewidth=1)
    plt.plot(energy, delta3, color[0], label=r'$\theta = 70^{\circ}$', linewidth=1)
    plt.plot(energy, delta4, color[3], label=r'$\theta = 70^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta1, color[1], linestyle='--', label=r'$\theta_{model} = 60^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta2, color[2], linestyle='--', label=r'$\theta_{model} = 65^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta3, color[0], linestyle='--', label=r'$\theta_{model} = 70^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta4, color[3], linestyle='--', label=r'$\theta_{model} = 70^{\circ}$', linewidth=1)
    plt.xlabel(r'$h\nu (eV)$')
    plt.ylabel(r'$\Delta (^{\circ})$')
    plt.title('$AlGaAs$ layer - $\Delta$')
    plt.grid()
    lgd = plt.legend(ncol=2, loc='center', bbox_to_anchor=(0.5, -0.5), handleheight= 2.4)
    plt.savefig('YSZ_Delta_' + name, dpi=200)
    def export_legend(legend, filename='legend_delta' + name + '.png', expand=[-5,-5,5,5]):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=200, bbox_inches=bbox)
    export_legend(lgd)
    plt.show()
    
plot_psi(file)
plot_delta(file)    
    
    
    