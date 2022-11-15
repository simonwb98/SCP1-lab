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

file_name1 = 'S1127_R_1p5to3eV_0to45deg_by_5deg.txt'
file_name2 = 'S1127_R_1p5to3eV_50to85deg_by_5deg.txt'


colors = plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 19))))

def get_data(file):
    with open(file, 'r') as input_file:
        energy = []
        R1, R2, R3, R4, R5, R6, R7, R8, R9, R10 = [], [], [], [], [], [], [], [], [], []
        input_file.readline()
        input_file.readline()
        for line in input_file:
            data = line.split('\t')
            energy.append(float(data[0]))
            R1.append(float(data[1]))
            R2.append(float(data[2]))
            R3.append(float(data[3]))
            R4.append(float(data[4]))
            R5.append(float(data[5]))
            R6.append(float(data[6]))
            R7.append(float(data[7]))
            R8.append(float(data[8]))
            try:
                R9.append(float(data[9]))
                R10.append(float(data[10]))
            except:
                pass
    R = [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10]
    return (energy, R)
    
def plot_R(file1, file2):
    energy1, R1 = get_data(file1)
    energy2, R2 = get_data(file2)
    x = 0
    for reflectivity in R1:
        plt.plot(energy1, reflectivity, label = r'$\theta =$' + str(x) + '$^{\circ}$')
        x += 5
    for reflectivity in R2:
        try:
            plt.plot(energy2, reflectivity, label = r'$\theta =$' + str(x) + '$^{\circ}$')
            x += 5
        except ValueError:
            pass
    legend = plt.legend(ncol = 2, title = 'incidence angles',loc = 'center', handleheight= 2.4, bbox_to_anchor=(0.5, -0.7))
    def export_legend(legend, filename="BR_legend.png", expand=[-5,-5,5,5]):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=200, bbox_inches=bbox)
    export_legend(legend)
    plt.title('Reflectivity Bragg Reflector')
    plt.xlabel(r'$h\nu$ (eV)')
    plt.ylabel(r'R')
    plt.grid()
    plt.savefig('BR_R.png', dpi=200)
    plt.show()
    
    
    
plot_R(file_name1, file_name2)
    
    