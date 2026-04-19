# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:52:09 2025

@author: risc915d
"""
import numpy as np

def write_schemes(write_scheme, rows, cols, write_voltage, wordline, bitline, Rload, Ropen, R_wordline=None, R_bitline=None):
    
    if write_scheme == 'V/2':
        ## V/2 writing scheme
        V_top       = np.full(cols, write_voltage/2)
        V_left      = np.full(rows, write_voltage/2)       
        V_top[bitline] = 0  # Top side voltages
        V_left[wordline] = write_voltage  # Left side voltages
        R_load_top      = np.full(cols, Rload)   
        R_load_left     = np.full(rows, Rload)  
    elif write_scheme == 'V/3':
        ## V/3 writing scheme
        V_top       = np.full(cols, write_voltage*2/3)
        V_left      = np.full(rows, write_voltage*1/3)       
        V_top[bitline] = 0  # Top side voltages
        V_left[wordline] = write_voltage  # Left side voltages
        R_load_top      = np.full(cols, Rload)   
        R_load_left     = np.full(rows, Rload)  
    elif write_scheme == 'GND':  # grounded scheme
        V_top       = np.full(cols, 0)
        V_left      = np.full(rows, 0)       
        V_top[bitline] = 0  # Top side voltages
        V_left[wordline] = write_voltage  # Left side voltages
        R_load_top      = np.full(cols, Rload)
        R_load_left     = np.full(rows, Rload)  
    else:  # floating scheme
        V_top       = np.full(cols, 0)
        V_left      = np.full(rows, 0)       
        V_top[bitline] = 0  # Top side voltages
        V_left[wordline] = write_voltage  # Left side voltages
        R_load_top      = np.full(cols, Ropen)  
        R_load_left     = np.full(rows, Ropen)  
        R_load_top[bitline]     = Rload 
        R_load_left[wordline]   = Rload 
    
    V_bottom    = np.zeros(cols)  
    V_right     = np.zeros(rows)  
    R_load_right    = np.full(rows, Ropen)  
    
    # type conversion
    R_load_left = R_load_left.astype(np.float64)
    R_load_right = R_load_right.astype(np.float64)
    R_load_top = R_load_top.astype(np.float64)
    
    # add the measured word, bit line resistance
    if R_wordline is not None:
        R_load_left += R_wordline    
    if R_bitline is not None:
        R_load_top += R_bitline
    
    return V_left, V_right, V_top, R_load_left, R_load_right, R_load_top


def read_schemes(config, rows, cols, read_voltage, wordline, bitline, Rload, Ropen, R_wordline=None, R_bitline=None):
    V_left = np.zeros(rows)
    V_right = np.zeros(rows)
    V_top = np.zeros(cols)
    R_load_left = np.full(rows, Ropen)
    R_load_right = np.full(rows, Ropen)
    R_load_top = np.full(cols, Ropen)
    
    # Set selected wordline and bitline
    V_left[wordline] = read_voltage
    R_load_left[wordline] = Rload
    R_load_top[bitline] = Rload

    if config == 1:
        # Unsel. WLs: F, Unsel. BLs: F
        pass  # Already set to Ropen and 0V
    elif config == 2:
        # Unsel. WLs: F, Unsel. BLs: Vdd
        V_top[V_top == 0] = read_voltage
        R_load_top = np.full(cols, Rload)
    elif config == 3:
        # Unsel. WLs: F, Unsel. BLs: 0
        R_load_top = np.full(cols, Rload)
    elif config == 4:
        # Unsel. WLs: Vdd, Unsel. BLs: F
        V_left[V_left == 0] = read_voltage
        R_load_left = np.full(rows, Rload)
    elif config == 5:
        # Unsel. WLs: Vdd, Unsel. BLs: Vdd
        V_left[V_left == 0] = read_voltage
        R_load_left = np.full(rows, Rload)
        V_top[V_top == 0] = read_voltage
        R_load_top = np.full(cols, Rload)
    elif config == 6:
        # Unsel. WLs: Vdd, Unsel. BLs: 0
        V_left[V_left == 0] = read_voltage
        R_load_left = np.full(rows, Rload)
        R_load_top = np.full(cols, Rload)
    elif config == 7:
        # Unsel. WLs: 0, Unsel. BLs: F
        R_load_left = np.full(rows, Rload)
    elif config == 8:
        # Unsel. WLs: 0, Unsel. BLs: Vdd
        R_load_left = np.full(rows, Rload)
        V_top[V_top == 0] = read_voltage
        R_load_top = np.full(cols, Rload)
    elif config == 9:
        # Unsel. WLs: 0, Unsel. BLs: 0
        R_load_left = np.full(rows, Rload)
        R_load_top = np.full(cols, Rload)
    else:
        raise ValueError("Invalid configuration. Please choose a number between 1 and 9.")
    
    R_load_left = R_load_left.astype(np.float64)
    R_load_right = R_load_right.astype(np.float64)
    R_load_top = R_load_top.astype(np.float64)
    
    # add the measured word, bit line resistance
    if R_wordline is not None:
        R_load_left += R_wordline    
    if R_bitline is not None:
        R_load_top += R_bitline
    
    return V_left, V_right, V_top, R_load_left, R_load_right, R_load_top


def run_scheme(rows, cols, time, voltage, Rload, Ropen):
    V_left = np.zeros((rows, len(voltage)))
    V_right = np.zeros(rows)
    V_top = np.zeros(cols)
    R_load_left = np.full(rows, Rload)
    R_load_right = np.full(rows, Ropen)
    R_load_top = np.full(cols, Ropen)
    """
    # define voltage input
    if isinstance(voltage, (list, np.ndarray)):
        # If voltage is an array, create PWL strings for each row
        pwl_string = ' '.join([f'{t} {v}' for t, v in zip(time, voltage)])
        V_left = [f'PWL({pwl_string})' for _ in range(rows)]
    else:
        # If voltage is a single value, fill the array with that value
        V_left[:] = voltage
    """
    
    # define voltage input
    if isinstance(voltage, (list, np.ndarray)):
        # If voltage is an array, create PWL strings for each row
        pwl_strings = []
        for row in range(rows):
            row_time = time[row]
            row_voltage = voltage[row]
            pwl_string = ' '.join([f'{t} {v}' for t, v in zip(row_time, row_voltage)])
            pwl_strings.append(f'PWL({pwl_string})')
        V_left = pwl_strings
    else:
        # If voltage is a single value, fill the array with that value
        V_left[:] = voltage
    
    R_load_left = R_load_left.astype(np.float64)
    R_load_right = R_load_right.astype(np.float64)
    R_load_top = R_load_top.astype(np.float64)
    
    return V_left, V_right, V_top, R_load_left, R_load_right, R_load_top


