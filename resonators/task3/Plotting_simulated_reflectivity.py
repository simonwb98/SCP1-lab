# -*- coding: utf-8 -*-
"""
Plotting the reflectivity of the Resonator
"""
import numpy as np
import cmath
import matplotlib.pyplot as plt
from pylab import figure, show, legend, xlabel,  ylabel, title, savefig, suptitle
from scipy.signal import find_peaks, peak_widths, peak_prominences


#constants
hc = 1240#in units eV*nm
d = 0 #layer thickness in nm

file_name1 = 'S1128_calculated_R_from_fit_0to45deg.txt'
file_name2 = 'S1128_calculated_R_from_fit_50to85deg.txt'


colors = plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 10))))
color = ["#E69F00", "#56B4E9", "#009E73", "#0072B2",  "#D55E00", "#CC79A7", "#F0E442", "#EF4026", '#b603fc', '#fcba03']

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
    i = 0
    for reflectivity in R1:
        plt.plot(energy1, reflectivity, label = r'$\theta =$' + str(x) + '$^{\circ}$', zorder = 4-i)
        i += 1
        x += 5
        
    for g in range(7):
        plt.plot(energy2, R2[g], label = r'$\theta =$' + str(x) + '$^{\circ}$', zorder = 4-i)
        i += 1
        x += 5

    legend = plt.legend(ncol = 2, title = 'incidence angles',loc = 'center', handleheight= 2.4, bbox_to_anchor=(0.5, -0.7))
    def export_legend(legend, filename="MR_legend.png", expand=[-5,-5,5,5]):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=200, bbox_inches=bbox)
    export_legend(legend)
    plt.title('Reflectivity Microresonator')
    plt.xlabel(r'$h\nu$ (eV)')
    plt.ylabel(r'R')
    plt.grid()
    plt.xlim(1.8, 2.6)
    plt.ylim(0.4, 1)
    plt.savefig('MR_R.png', dpi=200)
    
    plt.show()
    
    # The following is for finding the peak positions of the cavity energies.
    j = 0
    x = [] # saves the peak energies
    y = [] # saves the peak widths in the transmission
    # To convert to peak widths in the energy, assume, that E is linearly distributed with the running index. 
    # Then, E[j + x], where x need not be an integer is approx. E[j] + mx, 
    # Here, m is the slope of E at the index j, calculated with the two neighbouring points of E[j]. 
    # Hence, E[j + width/2] = E[j] + width*(E[j+1] - E[j-1]), s.t. 
    # E_width = 2*width*(E[j+1] - E[j-1])
    
    while j < 10:
        transmission = 1 - np.array(R1[j])
        energy = np.array(energy1)
        peaks, _ = find_peaks(transmission, prominence=0.6)
        
        # widths = peak_widths(transmission, peaks, rel_height=0.5)

        # y.append(widths[0])
        for peak in peaks:
            x.append(energy[peak])
            # E_width.append(float(abs(2*widths[0]*(energy[peak+1] - energy[peak-1]))))
        
        j += 1
    
    k = 0
    while k < 8:
        transmission = 1 - np.array(R2[k])
        energy = np.array(energy2)
        peaks, _ = find_peaks(transmission, threshold=0.05)
        
        # widths = peak_widths(transmission, peaks, rel_height= 0.9)
        
        # y.append(widths[0])
        for peak in peaks:
            x.append(energy[peak])
            # E_width.append(float(abs(2*widths[0]*(energy[peak+1] - energy[peak-1]))))
        k += 1
    
    
    # The commented code unfortunately, doesn't give well FWHM values, instead 
    # they were now obtained using origin
    E_width = np.array([0.0048, 0.0045, 0.0045, 0.0046, 0.0049, 0.0056, 0.0067, 0.0084, 0.0105, 0.013, 0.015, 0.017, 0.02, 0.023, 0.024, 0.032, 0.051, 0.110])
    theta = np.linspace(0, 85, 18)
    plt.title(r'Microresonator - $E_{\mathrm{cav}}$ vs. $\theta$')
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(theta, x, color[0], marker = 'X')
    axs[0].set(ylabel=r'$E_{\mathrm{cav}}$ (eV)', ylim=[1.95, 2.4])
    axs[0].grid()
    # axs[1].plot(theta, E_width, color[0], marker= 'X')
    # axs[1].set(ylabel=r'FWHM (eV)', ylim=[-0.01, 0.13])
    # axs[1].grid()
    # axs[0].plot(theta, x/E_width, color[0], marker='X')
    # axs[0].set(ylabel=r'$Q$', ylim=[-20, 500])
    # axs[0].grid()
    # plt.xlabel(r'$\theta (^{\circ})$')
    print(x)
    
    
    plt.savefig('MR_E_cav.png', dpi=300)
    
def plot_eff_n(x):
    d = 142 #in nm
    x = np.array(x)
    theta = np.pi/180*np.linspace(0, 85, 18)
    k = 2*np.pi*x/hc*np.sin(theta)
    n = hc/(2*np.pi*x)*np.sqrt((np.pi/d)**2 + k**2)
    plt.plot(k*1e3, n, color[0], marker = 'X')
    plt.xlabel(r'$k_{\parallel}(\mu$m$^{-1})$')
    plt.ylabel(r'$n_{{eff}}$')
    plt.grid()
    plt.title('Microresonator - $n_{eff}$ vs. $k_{\parallel}$')
    plt.savefig('MR_n_eff.png', dpi=200)
plot_eff_n([2.004, 2.006, 2.012, 2.022, 2.035, 2.051, 2.07, 2.092, 2.116, 2.142, 2.169, 2.195, 2.221, 2.245, 2.267, 2.285, 2.298, 2.306])
# plot_R(file_name1, file_name2)
    
    