# -*- coding: utf-8 -*-
"""
Plotting the dielectric function in dependence of the stochiometry in Al_xGa_1-xAs
"""
import numpy as np
import matplotlib.pyplot as plt

input_file = 'stochiometry.txt'
color = ['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b','#ffffbf','#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2']

def get_data(file):
    with open(file, 'r') as data_file:
        energy = []
        er0, er1, er2, er3, er4, er5, er6, er7, er8, er9, er10 = [], [], [], [], [], [], [], [], [], [], []
        ei0, ei1, ei2, ei3, ei4, ei5, ei6, ei7, ei8, ei9, ei10 = [], [], [], [], [], [], [], [], [], [], []
        for i in range(4):
            data_file.readline()
        for line in data_file:
            data = line.split('\t')
            energy.append(float(data[0]))
            er0.append(float(data[1]))
            ei0.append(float(data[2]))
            er1.append(float(data[3]))
            ei1.append(float(data[4]))
            er2.append(float(data[5]))
            ei2.append(float(data[6]))
            er3.append(float(data[7]))
            ei3.append(float(data[8]))
            er4.append(float(data[9]))
            ei4.append(float(data[10]))
            er5.append(float(data[11]))
            ei5.append(float(data[12]))
            er6.append(float(data[13]))
            ei6.append(float(data[14]))
            er7.append(float(data[15]))
            ei7.append(float(data[16]))
            er8.append(float(data[17]))
            ei8.append(float(data[18]))
            er9.append(float(data[19]))
            ei9.append(float(data[20]))
            er10.append(float(data[21]))
            ei10.append(float(data[22]))
        er = [er0, er1, er2, er3, er4, er5, er6, er7, er8, er9, er10]
        ei = [ei0, ei1, ei2, ei3, ei4, ei5, ei6, ei7, ei8, ei9, ei10]
        return (energy, er, ei)

def plot_data(file):
    energy, er, ei = get_data(file)
    x = 1
    i = 0
    for index in range(10, -1, -1):
        plt.plot(energy, np.array(ei[index]), color[index], label= str(round(x, 2)))
        x -= 0.1
        i += 1
    plt.legend()
    plt.grid()
    plt.xlabel(r'$h\nu$ (eV)')
    plt.ylabel(r'$\epsilon_2$')
    plt.ylim(7.5, 27)
    plt.title(r'$Al_xGa_{1-x}As$ - $\epsilon_2$ for different stochiometries $x$')
    plt.text(1.35, 14, '$E_0$', fontsize = 12)
    plt.text(1.5, 16, '$E_0+\Delta_0$', fontsize=12)
    plt.arrow(1.7, 15.7, 0.01, -1.5, width= 0.005, head_width=0.1, head_length=0.7, length_includes_head=True, color='k')
    plt.text(2.8, 25.1, '$E_1$')
    plt.text(3.1, 25.1, '$E_1 + \Delta_1$')
    plt.arrow(3.25, 24.8, -0.2, -7.5, width= 0.005, head_width=0.1, head_length=0.7, length_includes_head=True, color='k', zorder=10)
    plt.savefig('epsilon_2_stochiometry', dpi=200)
plot_data(input_file)
    