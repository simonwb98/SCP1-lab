# -*- coding: utf-8 -*-
"""
Plotting the reflectivity of the Resonator
"""
import numpy as np
import cmath
import matplotlib.pyplot as plt
from pylab import figure, show, legend, xlabel,  ylabel, title, savefig, suptitle


#constants
hc = 1240 #in units eV*nm
d = 0 #layer thickness in nm

file_name = 'r_45_pol_45to85_deg_by_5.dat'



colors = plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 9))))
#color = ["#E69F00", "#56B4E9", "#009E73", "#0072B2",  "#D55E00", "#CC79A7", "#F0E442", "#EF4026", '#b603fc', '#fcba03']

def get_data(file):
    with open(file, 'r') as input_file:
        E1, E2, E3, E4, E5, E6, E7, E8, E9 = [], [], [], [], [], [], [], [], []
        R1, R2, R3, R4, R5, R6, R7, R8, R9 = [], [], [], [], [], [], [], [], []
        energy = [E1, E2, E3, E4, E5, E6, E7, E8, E9]
        R = [R1, R2, R3, R4, R5, R6, R7, R8, R9]
        input_file.readline()
        input_file.readline()
        input_file.readline()
        input_file.readline()
        i = 0 #list index for E and R
        x = 45.000000 #angle of incidence
        for line in input_file:
            data = line.split('\t')
            if float(data[2]) == x:
                energy[i].append(float(data[1]))
                R[i].append(float(data[3]))
            else:
                i += 1
                x += 5
                energy[i].append(float(data[1]))
                R[i].append(float(data[3]))
        
    return (energy, R)
    
def plot_R(file):
    energy, R = get_data(file)
    x = 45
    for i in range(9):
        plt.plot(energy[i], R[i], label=r'$\theta = $' + str(x) + '$^{\circ}$')
        x += 5
    plt.title('Measured Reflectivity for Bragg Reflector')
    plt.legend(loc = 'upper left')
    plt.xlabel(r'$h\nu$ (eV)')
    plt.ylabel(r'R')
    plt.grid()
    plt.savefig('BR_R_measured.png', dpi=200)
    plt.show()
    
    
    
plot_R(file_name)
    
    