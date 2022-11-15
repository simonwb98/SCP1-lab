# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:11:58 2021

@author: User
"""

#Skript for Task 1: Calculating the Hall coefficient and the resistivity

import numpy as np
import math


file = 'E3266_a-ZTO_2020_1.txt'

def get_data(input_file):
    with open(input_file, 'r') as data_file:
        B = 0.43
        thickness = 1e-06*float(input('Please specify the sample thickness in micrometers: '))
        voltages = []
        currents = []
        lines = data_file.readlines()[5:9] #only read the four lines contating data, switch to 14:18
        for line in lines:
            a, b, c, d, e, f, g = line.split('\t')
            voltages.append(float(c))
            voltages.append(float(d))
            voltages.append(float(e))
            voltages.append(float(f))
            currents.append(float(b))
            temperature = a
    return (voltages, currents, B, thickness, temperature)
    


def calculate_resistivity(input_file):
    voltages, currents, B, thickness, temperature = get_data(input_file)
    A = math.pi*thickness/(2*np.log(2))
    rho1 = -A*(float(voltages[0] + voltages[1]))/currents[0]
    rho2 = -A*(float(voltages[4] + voltages[5]))/currents[1]
    rho3 = A*(float(voltages[8] + voltages[9]))/currents[2]
    rho4 = A*(float(voltages[12] + voltages[13]))/currents[3]
    average = 0.25*(rho1 + rho2 + rho3 + rho4)
    print('The average resistivity at T = ' + str(temperature) + 'K is ' + str(average) + 'Ωm.')
    
def calculate_Hall_coefficient(input_file):
    voltages, currents, B, thickness, temperature = get_data(input_file)
    A = thickness/B
    R1 = -A*(voltages[2] - voltages[3])/currents[0]
    R2 = -A*(voltages[6] - voltages[7])/currents[1]
    R3 = A*(voltages[10] - voltages[11])/currents[2]
    R4 = A*(voltages[14] - voltages[15])/currents[3]
    average = 0.25*(R1 + R2 + R3 + R4)
    print('The average Hall coefficient at T = ' + str(temperature) + 'K is ' + str(average) + 'Ω.')


print(calculate_resistivity(file))
print(calculate_Hall_coefficient(file))