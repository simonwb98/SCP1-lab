# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:05:50 2022

@author: User
"""

#A5 (Semiconductor Labs) - Code for plotting

import numpy as np
# import math
import matplotlib.pyplot as plt

file = 'Task1_Al2O3_substrate_measuredPsi&Delta_VASE.txt'

def get_data(input_file):
    with open(input_file, 'r') as data_file:
        energy = []
        psi1, psi2, psi3, psi4, psi5, psi6, psi7, psi8 = [], [], [], [], [], [], [], []
        delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8 = [], [], [], [], [], [], [], []
        model_psi1, model_psi2, model_psi3, model_psi4, model_psi5, model_psi6, model_psi7, model_psi8 = [], [], [], [], [], [], [], []
        model_delta1, model_delta2, model_delta3, model_delta4, model_delta5, model_delta6, model_delta7, model_delta8 = [], [], [], [], [], [], [], []
        data_file.readline()
        data_file.readline()
        data_file.readline()
        data_file.readline()
        for line in data_file:
            a, b, c, f, g, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, A, B, C, D, E, F, G, H, I, J, K, L = line.split('\t') 
            energy.append(float(a))
            
            psi1.append(float(b))
            delta1.append(float(c))
            psi2.append(float(f))
            delta2.append(float(g))
            psi3.append(float(j))
            delta3.append(float(k))
            psi4.append(float(l))
            delta4.append(float(m))
            psi5.append(float(n))
            delta5.append(float(o))
            psi6.append(float(p))
            delta6.append(float(q))
            psi7.append(float(r))
            delta7.append(float(s))
            psi8.append(float(t))
            delta8.append(float(u))
            
            model_psi1.append(float(v)) 
            model_delta1.append(float(w))
            model_psi2.append(float(x))
            model_delta2.append(float(y))
            model_psi3.append(float(z))
            model_delta3.append(float(A))
            model_psi4.append(float(B))
            model_delta4.append(float(C))
            model_psi5.append(float(D))
            model_delta5.append(float(E))
            model_psi6.append(float(F))
            model_delta6.append(float(G))
            model_psi7.append(float(H))
            model_delta7.append(float(I))
            model_psi8.append(float(J))
            model_delta8.append(float(K))
            
        PSI = (psi1, psi2, psi3, psi4, psi5, psi6, psi7, psi8)
        DELTA = (delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8)
        MODEL_PSI = (model_psi1, model_psi2, model_psi3, model_psi4, model_psi5, model_psi6, model_psi7, model_psi8)
        MODEL_DELTA = (model_delta1, model_delta2, model_delta3, model_delta4, model_delta5, model_delta6, model_delta7, model_delta8)
        
    return (energy, PSI, DELTA, MODEL_PSI, MODEL_DELTA)
    
def plot_psi(input_file):
    energy, PSI, DELTA, MODEL_PSI, MODEL_DELTA = get_data(input_file)
    energy = np.array(energy)
    psi1, psi2, psi3, psi4, psi5, psi6, psi7, psi8 = PSI
    model_psi1, model_psi2, model_psi3, model_psi4, model_psi5, model_psi6, model_psi7, model_psi8 = MODEL_PSI
    psi1, psi2, psi3, psi4, psi5, psi6, psi7, psi8 = map(np.array, PSI)
    model_psi1, model_psi2, model_psi3, model_psi4, model_psi5, model_psi6, model_psi7, model_psi8 = map(np.array, MODEL_PSI)
    
    color = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#EF4026"]
    plt.plot(energy, psi1, color[1], label= r'$\theta = 45^{\circ}$', linewidth=1)
    plt.plot(energy, psi2, color[2], label=r'$\theta = 50^{\circ}$', linewidth=1)
    plt.plot(energy, psi3, color[3], label=r'$\theta = 55^{\circ}$', linewidth=1)
    plt.plot(energy, psi4, color[4], label=r'$\theta = 60^{\circ}$', linewidth=1)
    plt.plot(energy, psi5, color[5], label=r'$\theta = 65^{\circ}$', linewidth=1)
    plt.plot(energy, psi6, color[6], label=r'$\theta = 70^{\circ}$', linewidth=1)
    plt.plot(energy, psi7, color[7], label=r'$\theta = 75^{\circ}$', linewidth=1)
    plt.plot(energy, psi8, color[0], label=r'$\theta = 80^{\circ}$', linewidth=1)
    
    plt.plot(energy, model_psi1, color[1], linestyle='--', label=r'$\theta_{model} = 45^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi2, color[2], linestyle='--', label=r'$\theta_{model} = 50^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi3, color[3], linestyle='--', label=r'$\theta_{model} = 55^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi4, color[4], linestyle='--', label=r'$\theta_{model} = 60^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi5, color[5], linestyle='--', label=r'$\theta_{model} = 65^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi6, color[6], linestyle='--', label=r'$\theta_{model} = 70^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi7, color[7], linestyle='--', label=r'$\theta_{model} = 75^{\circ}$', linewidth=1)
    plt.plot(energy, model_psi8, color[0], linestyle='--', label=r'$\theta_{model} = 80^{\circ}$', linewidth=1)

    plt.xlabel(r'$h\nu (eV)$')
    plt.ylabel(r'$\Psi (^{\circ})$')
    plt.title('$Al_2O_3$ - $\Psi$ for different angles of incidence')
    plt.grid()
    legend = plt.legend(ncol = 2, title = 'Measured and modeled $\Psi$ ',loc = 'center', handleheight= 2.4, bbox_to_anchor=(0.5, -0.7))
    # plt.xlim(0.0, 4.0)
    # plt.ylim(0.0, 16.0)
    # plt.xticks([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    # plt.yticks([4, 8, 12, 16], [4, 8, 12, 16])
    # fig = plt.figure(figsize=(3,3))
    # ax  = fig.add_subplot(111)
    # ax.plot(energy, psi1)
    plt.savefig('substrate_psi', dpi=200)
    def export_legend(legend, filename="legend_psi.png", expand=[-5,-5,5,5]):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=200, bbox_inches=bbox)
    export_legend(legend)
    plt.show()
    
def plot_delta(input_file):
    energy, PSI, DELTA, MODEL_PSI, MODEL_DELTA = get_data(input_file)
    energy = np.array(energy)
    delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8 = DELTA
    model_delta1, model_delta2, model_delta3, model_delta4, model_delta5, model_delta6, model_delta7, model_delta8 = MODEL_DELTA
    delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8 = map(np.array, DELTA)
    model_delta1, model_delta2, model_delta3, model_delta4, model_delta5, model_delta6, model_delta7, model_delta8 = map(np.array, MODEL_DELTA)
    
    color = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#EF4026"]
    plt.plot(energy, delta1, color[1], label= r'$\theta = 45^{\circ}$', linewidth=1)
    plt.plot(energy, delta2, color[2], label=r'$\theta = 50^{\circ}$', linewidth=1)
    plt.plot(energy, delta3, color[3], label=r'$\theta = 55^{\circ}$', linewidth=1)
    plt.plot(energy, delta4, color[4], label=r'$\theta = 60^{\circ}$', linewidth=1)
    plt.plot(energy, delta5, color[5], label=r'$\theta = 65^{\circ}$', linewidth=1)
    plt.plot(energy, delta6, color[6], label=r'$\theta = 70^{\circ}$', linewidth=1)
    plt.plot(energy, delta7, color[7], label=r'$\theta = 75^{\circ}$', linewidth=1)
    plt.plot(energy, delta8, color[0], label=r'$\theta = 80^{\circ}$', linewidth=1)
    
    plt.plot(energy, model_delta1, color[1], linestyle='--', label=r'$\theta_{model} = 45^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta2, color[2], linestyle='--', label=r'$\theta_{model} = 50^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta3, color[3], linestyle='--', label=r'$\theta_{model} = 55^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta4, color[4], linestyle='--', label=r'$\theta_{model} = 60^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta5, color[5], linestyle='--', label=r'$\theta_{model} = 65^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta6, color[6], linestyle='--', label=r'$\theta_{model} = 70^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta7, color[7], linestyle='--', label=r'$\theta_{model} = 75^{\circ}$', linewidth=1)
    plt.plot(energy, model_delta8, color[0], linestyle='--', label=r'$\theta_{model} = 80^{\circ}$', linewidth=1)

    plt.xlabel(r'$h\nu (eV)$')
    plt.ylabel(r'$\Delta (^{\circ})$')
    plt.title('$Al_2O_3$ - $\Delta$ for different angles of incidence')
    plt.grid()
    legend = plt.legend(ncol = 2, title = 'Measured and modeled $(\Psi, \Delta)$ ',loc = 'center', handleheight= 2.4, bbox_to_anchor=(0.5, -0.7))
    # plt.xlim(0.0, 4.0)
    # plt.ylim(0.0, 16.0)
    # plt.xticks([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    # plt.yticks([4, 8, 12, 16], [4, 8, 12, 16])
    # fig = plt.figure(figsize=(3,3))
    # ax  = fig.add_subplot(111)
    # ax.plot(energy, psi1)
    plt.savefig('substrate_delta', dpi=200)
    def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=200, bbox_inches=bbox)
    export_legend(legend)
    plt.show()
    
# plot_psi(file)
plot_delta(file)    
    
    
    