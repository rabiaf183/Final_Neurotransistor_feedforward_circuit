"""Fully Connected Layer with MNIST Data - 3 Neurotransistors to 1 Output"""

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from memcap_model import get_memcap_subcircuit
from MNIST import create_MNIST_pulse_train, load_mnist
import warnings

# Suppress warnings to avoid clutter during execution
warnings.filterwarnings('ignore')

mpl.rcParams['axes.formatter.useoffset'] = False

# === PARAMETERS ===
memristor = 'MEMCAP'
memristor_subcircuit = get_memcap_subcircuit()

# Array size for each neurotransistor
N = 6  # rows for MNIST
M = 3  # columns
num_neurons = 3  # 3 input neurons feeding 1 output neuron

# Circuit parameters
C_cross = 10e-12
C_gb = 100e-12
R_wire_wl = 0.33
R_wire_bl = 60
Rload = 1
Ropen = 1e9
Rs = 1
Rout1 = 1e6
Rt = 10

# MOSFET parameters
NMOS_Vto = 0.15
PMOS_Vto = -0.2
VDD = 1.0

# Pulse parameters
pulse_voltage = 1
pulse_width = 5e-6
pulse_slope = 200e-9
rel_tol = 1e-4

# Initial memristor states
xmin = 0.1
xmax = 0.284
seed = 42
np.random.seed(seed)
x0_neurons = [np.random.uniform(low=xmin, high=xmax, size=(N, M)) for _ in range(num_neurons)]
x0_output = np.random.uniform(low=xmin, high=xmax, size=(num_neurons, M))

# === LOAD MNIST ===
print("Loading MNIST...")
imx, imy = load_mnist('raw', kind='train')

# Select 3 different digits for 3 input neurons
digits = [1, 1, 7]
digit_images = []
pulse_trains_list = []

for i, digit in enumerate(digits):
    pulse_trains, digit_image = create_MNIST_pulse_train(
        imx, imy, N, pulse_voltage, pulse_width, pulse_slope,
        selected_digits=[digit], do_plot=None, specific_image_index=i
    )
    pulse_trains_list.append(pulse_trains[digit])
    digit_images.append(digit_image)

times_list = [pt['times'] for pt in pulse_trains_list]
voltages_list = [pt['voltages'] for pt in pulse_trains_list]
total_time = times_list[0][0][-1]

# === BUILD CIRCUIT ===
def build_circuit():
    circuit = Circuit('Fully Connected Layer - MNIST')
    initial_conditions = {}
    
    # MOSFET models
    circuit.model('NM_NMOS', 'nmos', LEVEL=1, L=2e-6, W=94e-6, Vto=NMOS_Vto, Tox=22e-9)
    circuit.model('NM_PMOS', 'pmos', LEVEL=1, L=2e-6, W=94e-6, Vto=PMOS_Vto, Tox=22e-9)
    
    # Power supply
    circuit.V('VDD', 'vdd', circuit.gnd, VDD)
    
    # =====================================================
    # BUILD 3 INPUT NEUROTRANSISTORS
    # =====================================================
    for neuron in range(num_neurons):
        prefix = f'n{neuron}_'
        times = times_list[neuron]
        voltages = voltages_list[neuron]
        x0 = x0_neurons[neuron]
        
        # Wordlines
        for i in range(N):
            for j in range(M - 1):
                circuit.R(f'{prefix}wire_row_{i}_{j}', f'{prefix}row_{i}_{j}', f'{prefix}row_{i}_{j+1}', R_wire_wl)
        
        # Bitlines
        for j in range(M):
            for i in range(N - 1):
                circuit.R(f'{prefix}wire_col_{i}_{j}', f'{prefix}col_{i}_{j}', f'{prefix}col_{i+1}_{j}', R_wire_bl)
        
        # Memristors
        for i in range(N):
            for j in range(M):
                circuit.X(f'{prefix}M_{i}_{j}', memristor, f'{prefix}row_{i}_{j}', f'{prefix}col_{i}_{j}', f'{prefix}sv_{i}_{j}')
                initial_conditions[f'{prefix}sv_{i}_{j}'] = x0[i, j]
                circuit.C(f'{prefix}C_cross_{i}_{j}', f'{prefix}row_{i}_{j}', f'{prefix}col_{i}_{j}', C_cross)
        
        # Input sources from MNIST
        for i in range(N):
            pwl_values = [(t, v) for t, v in zip(times[i], voltages[i])]
            circuit.R(f'{prefix}R_load_left_{i}', f'{prefix}row_{i}_0', f'{prefix}left_{i}', Rload)
            circuit.PieceWiseLinearVoltageSource(f'{prefix}V_left_{i}', f'{prefix}left_{i}', circuit.gnd, values=pwl_values)
        
        # Right side open
        for i in range(N):
            circuit.R(f'{prefix}R_load_right_{i}', f'{prefix}row_{i}_{M-1}', f'{prefix}right_{i}', Ropen)
            circuit.V(f'{prefix}V_right_{i}', f'{prefix}right_{i}', circuit.gnd, 0)
        
        # Top side open
        for j in range(M):
            circuit.R(f'{prefix}R_load_top_{j}', f'{prefix}col_0_{j}', f'{prefix}top_{j}', Ropen)
            circuit.V(f'{prefix}V_top_{j}', f'{prefix}top_{j}', circuit.gnd, 0)
        
        # NMOS transistors
        for j in range(M):
            circuit.MOSFET(f'{prefix}M_bot_{j}', f'{prefix}drain_{j}', f'{prefix}col_{N-1}_{j}', f'{prefix}source_{j}', f'{prefix}bulk', model='NM_NMOS')
            circuit.C(f'{prefix}C_gb_{j}', f'{prefix}col_{N-1}_{j}', f'{prefix}bulk', C_gb)
            circuit.R(f'{prefix}R_gb_{j}', f'{prefix}col_{N-1}_{j}', f'{prefix}bulk', 10e6)
            circuit.R(f'{prefix}R_db_{j}', f'{prefix}drain_{j}', f'{prefix}bulk', 10e6)
            circuit.R(f'{prefix}R_sb_{j}', f'{prefix}source_{j}', f'{prefix}bulk', 10e6)
            if j == 0:
                circuit.R(f'{prefix}R_out_left', f'{prefix}out_left', f'{prefix}source_{j}', 1)
                circuit.V(f'{prefix}V_out_left', f'{prefix}out_left', circuit.gnd, 0)
            else:
                circuit.R(f'{prefix}Rt_{j}', f'{prefix}drain_{j-1}', f'{prefix}source_{j}', Rt)
        
        circuit.R(f'{prefix}R_bulk', f'{prefix}bulk', circuit.gnd, 0.1)
        
        # Sense resistor Rout
        circuit.R(f'{prefix}Rout', f'{prefix}drain_{M-1}', f'{prefix}cm_in', Rs)
        
        # Current mirror for this neuron
        circuit.MOSFET(f'{prefix}M5', f'{prefix}cm_in', f'{prefix}cm_in', 'vdd', 'vdd', model='NM_PMOS')
        circuit.MOSFET(f'{prefix}M4', f'{prefix}v_out', f'{prefix}cm_in', 'vdd', 'vdd', model='NM_PMOS')
        circuit.R(f'{prefix}Rout1', f'{prefix}v_out', circuit.gnd, Rout1)
    
    # =====================================================
    # BUILD OUTPUT NEUROTRANSISTOR (receives from 3 inputs)
    # =====================================================
    prefix = 'out_'
    x0 = x0_output
    N_out = num_neurons
    M_out = M
    
    # Wordlines for output neuron
    for i in range(N_out):
        for j in range(M_out - 1):
            circuit.R(f'{prefix}wire_row_{i}_{j}', f'{prefix}row_{i}_{j}', f'{prefix}row_{i}_{j+1}', R_wire_wl)
    
    # Bitlines for output neuron
    for j in range(M_out):
        for i in range(N_out - 1):
            circuit.R(f'{prefix}wire_col_{i}_{j}', f'{prefix}col_{i}_{j}', f'{prefix}col_{i+1}_{j}', R_wire_bl)
    
    # Memristors for output neuron
    for i in range(N_out):
        for j in range(M_out):
            circuit.X(f'{prefix}M_{i}_{j}', memristor, f'{prefix}row_{i}_{j}', f'{prefix}col_{i}_{j}', f'{prefix}sv_{i}_{j}')
            initial_conditions[f'{prefix}sv_{i}_{j}'] = x0[i, j]
            circuit.C(f'{prefix}C_cross_{i}_{j}', f'{prefix}row_{i}_{j}', f'{prefix}col_{i}_{j}', C_cross)
    
    # Connect outputs of 3 neurons to input rows of output neuron
    for i in range(num_neurons):
        circuit.R(f'{prefix}R_in_{i}', f'n{i}_v_out', f'{prefix}row_{i}_0', Rload)
    
    # Right side open for output neuron
    for i in range(N_out):
        circuit.R(f'{prefix}R_load_right_{i}', f'{prefix}row_{i}_{M_out-1}', f'{prefix}right_{i}', Ropen)
        circuit.V(f'{prefix}V_right_{i}', f'{prefix}right_{i}', circuit.gnd, 0)
    
    # Top side open for output neuron
    for j in range(M_out):
        circuit.R(f'{prefix}R_load_top_{j}', f'{prefix}col_0_{j}', f'{prefix}top_{j}', Ropen)
        circuit.V(f'{prefix}V_top_{j}', f'{prefix}top_{j}', circuit.gnd, 0)
    
    # NMOS transistors for output neuron
    for j in range(M_out):
        circuit.MOSFET(f'{prefix}M_bot_{j}', f'{prefix}drain_{j}', f'{prefix}col_{N_out-1}_{j}', f'{prefix}source_{j}', f'{prefix}bulk', model='NM_NMOS')
        circuit.C(f'{prefix}C_gb_{j}', f'{prefix}col_{N_out-1}_{j}', f'{prefix}bulk', C_gb)
        circuit.R(f'{prefix}R_gb_{j}', f'{prefix}col_{N_out-1}_{j}', f'{prefix}bulk', 10e6)
        circuit.R(f'{prefix}R_db_{j}', f'{prefix}drain_{j}', f'{prefix}bulk', 10e6)
        circuit.R(f'{prefix}R_sb_{j}', f'{prefix}source_{j}', f'{prefix}bulk', 10e6)
        if j == 0:
            circuit.R(f'{prefix}R_out_left', f'{prefix}out_left', f'{prefix}source_{j}', 1)
            circuit.V(f'{prefix}V_out_left', f'{prefix}out_left', circuit.gnd, 0)
        else:
            circuit.R(f'{prefix}Rt_{j}', f'{prefix}drain_{j-1}', f'{prefix}source_{j}', Rt)
    
    circuit.R(f'{prefix}R_bulk', f'{prefix}bulk', circuit.gnd, 0.1)
    
    # Sense resistor Rout2 for output neuron
    circuit.R('Rout2', f'{prefix}drain_{M_out-1}', 'cm_in_out', Rs)
    
    # Current mirror for output neuron
    circuit.MOSFET('M5_out', 'cm_in_out', 'cm_in_out', 'vdd', 'vdd', model='NM_PMOS')
    circuit.MOSFET('M4_out', 'v_out_final', 'cm_in_out', 'vdd', 'vdd', model='NM_PMOS')
    circuit.R('Rout3', 'v_out_final', circuit.gnd, Rout1)
    
    # =====================================================
    # SPICE DIRECTIVES
    # =====================================================
    circuit.raw_spice += memristor_subcircuit
    
    # Save signals from input neurons
    for neuron in range(num_neurons):
        prefix = f'n{neuron}_'
        circuit.raw_spice += f".save v({prefix}v_out)\n"
    
    # Save signals from output neuron
    circuit.raw_spice += ".save v(cm_in_out)\n"
    circuit.raw_spice += ".save v(v_out_final)\n"
    circuit.raw_spice += ".probe I(RRout2)\n"
    circuit.raw_spice += ".probe I(RRout3)\n"
    
    # Save gate voltages of output neuron
    for j in range(M_out):
        circuit.raw_spice += f".save v(out_col_{N_out-1}_{j})\n"
    
    circuit.raw_spice += ".options plotwinsize=0\n"
    circuit.raw_spice += ".options method=gear gmin=1e-12\n"
    circuit.raw_spice += f".options reltol={rel_tol}\n"
    
    return circuit, initial_conditions

# === RUN SIMULATION ===
def run_simulation(circuit, initial_conditions):
    simulator = circuit.simulator(temperature=25, nominal_parameters=[])
    simulator.initial_condition(**initial_conditions)
    return simulator.transient(step_time=pulse_width, end_time=total_time)

# === EXTRACT DATA ===
def extract_data(analysis):
    return {
        't': np.array(analysis.time),
        'V_neuron_out': [np.array(analysis[f'n{i}_v_out']) for i in range(num_neurons)],
        'V_gate_out': [np.array(analysis[f'out_col_{num_neurons-1}_{j}']) for j in range(M)],
        'V_cm_in_out': np.array(analysis['cm_in_out']),
        'V_out_final': np.array(analysis['v_out_final']),
        'I_Rout2': np.array(analysis['rrout2']),
        'I_Rout3': np.array(analysis['rrout3']),
    }

# === PRINT RESULTS ===
def print_results(data):
    print("\n" + "="*60)
    print(f"FULLY CONNECTED LAYER - MNIST Digits {digits}")
    print("="*60)
    for i in range(num_neurons):
        print(f"V(neuron_{i}_out) range: {np.min(data['V_neuron_out'][i])*1e3:.1f} - {np.max(data['V_neuron_out'][i])*1e3:.1f} mV")
    print("-"*60)
    for j in range(M):
        print(f"V(gate_out_{j}) max: {np.max(data['V_gate_out'][j])*1e3:.1f} mV")
    print("-"*60)
    print(f"V(Rout2) range: {np.min(data['V_cm_in_out'])*1e3:.1f} - {np.max(data['V_cm_in_out'])*1e3:.1f} mV")
    print(f"V(Rout3) range: {np.min(data['V_out_final'])*1e3:.1f} - {np.max(data['V_out_final'])*1e3:.1f} mV")
    print(f"I(Rout2) peak: {np.max(np.abs(data['I_Rout2']))*1e6:.2f} µA")
    print(f"I(Rout3) peak: {np.max(np.abs(data['I_Rout3']))*1e9:.2f} nA")
    print("="*60)

# === PLOT RESULTS ===
def plot_results(data, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    t_us = data['t'] * 1e6

    colors = ['#3498db', '#e74c3c', '#8e44ad', '#f39c12', '#9b59b6', '#1abc9c']  # Professional color palette

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f'Fully Connected Layer with Current Mirror - MNIST Digits {digits}', fontsize=16, fontweight='bold', y=0.995)

    # Panel 1: Input V from 3 neurons
    v_neuron_all = np.concatenate([data['V_neuron_out'][0]*1e3, data['V_neuron_out'][1]*1e3, data['V_neuron_out'][2]*1e3])
    v_min, v_max = np.min(v_neuron_all), np.max(v_neuron_all)
    margin = 0.1 * (v_max - v_min) if v_max > v_min else 0.1
    axes[0].plot(t_us, data['V_neuron_out'][0]*1e3, color=colors[0], lw=1.2, label=f'Digit {digits[0]}')
    axes[0].plot(t_us, data['V_neuron_out'][1]*1e3, color=colors[1], lw=1.2, label=f'Digit {digits[1]}')
    axes[0].plot(t_us, data['V_neuron_out'][2]*1e3, color=colors[2], lw=1.2, label=f'Digit {digits[2]}')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_xlim(0, t_us[-1])
    axes[0].set_ylim(v_min - margin, v_max + margin)
    axes[0].legend(loc='upper right')
    axes[0].set_title('Input Voltage from 3 Neurons', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Digit insets
    for i, img in enumerate(digit_images):
        inset = axes[0].inset_axes([0.85 + i*0.05, 0.6, 0.04, 0.35])
        inset.imshow(img, cmap='gray')
        inset.set_title(f'{digits[i]}', fontsize=8)
        inset.axis('off')

    # Panel 2: Gate V at 4th neuron
    v_gate_all = np.concatenate([data['V_gate_out'][0]*1e3, data['V_gate_out'][1]*1e3, data['V_gate_out'][2]*1e3])
    v_min, v_max = np.min(v_gate_all), np.max(v_gate_all)
    margin = 0.1 * (v_max - v_min) if v_max > v_min else 0.1
    axes[1].plot(t_us, data['V_gate_out'][0]*1e3, color=colors[0], lw=1.2, label='V(gate0)')
    axes[1].plot(t_us, data['V_gate_out'][1]*1e3, color=colors[1], lw=1.2, label='V(gate1)')
    axes[1].plot(t_us, data['V_gate_out'][2]*1e3, color=colors[2], lw=1.2, label='V(gate2)')
    axes[1].axhline(NMOS_Vto*1e3, color='k', ls='--', lw=1, label='Vth')
    axes[1].set_ylabel('Voltage (mV)')
    axes[1].set_ylim(v_min - margin, v_max + margin)
    axes[1].legend(loc='upper left')
    axes[1].set_title('Gate Voltages at 4th Neuron', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: I(Rout2)
    i_rout2 = np.abs(data['I_Rout2']) * 1e6
    i_min, i_max = np.min(i_rout2), np.max(i_rout2)
    margin = 0.1 * (i_max - i_min) if i_max > i_min else 0.1
    axes[2].plot(t_us, i_rout2, color=colors[1], lw=1.2)
    axes[2].set_ylabel('Current (µA)')
    axes[2].set_ylim(i_min - margin, i_max + margin)
    axes[2].set_title('I(Rout2)', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: V(Rout2)
    v_cm_in_out = data['V_cm_in_out'] * 1e3
    v_min, v_max = np.min(v_cm_in_out), np.max(v_cm_in_out)
    margin = 0.1 * (v_max - v_min) if v_max > v_min else 0.1
    axes[3].plot(t_us, v_cm_in_out, color=colors[0], lw=1.2)
    axes[3].set_ylabel('Voltage (mV)')
    axes[3].set_ylim(v_min - margin, v_max + margin)
    axes[3].set_title('V(Rout2)', fontsize=11)
    axes[3].grid(True, alpha=0.3)

    # Panel 5: V(Rout3)
    v_out_final = data['V_out_final'] * 1e3
    v_min, v_max = np.min(v_out_final), np.max(v_out_final)
    margin = 0.1 * (v_max - v_min) if v_max > v_min else 0.1
    axes[4].plot(t_us, v_out_final, color=colors[2], lw=1.2)
    axes[4].set_ylabel('Voltage (mV)')
    axes[4].set_ylim(v_min - margin, v_max + margin)
    axes[4].set_title('V(Rout3)', fontsize=11)
    axes[4].grid(True, alpha=0.3)

    # Panel 6: I(Rout3) in nA
    i_rout3 = np.abs(data['I_Rout3']) * 1e9
    i_min, i_max = np.min(i_rout3), np.max(i_rout3)
    margin = 0.1 * (i_max - i_min) if i_max > i_min else 0.1
    axes[5].plot(t_us, i_rout3, color=colors[2], lw=1.2)
    axes[5].set_ylabel('Current (nA)')
    axes[5].set_xlabel('Time (µs)')
    axes[5].set_ylim(i_min - margin, i_max + margin)
    axes[5].set_title('I(Rout3)', fontsize=11)
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'{save_path}/fully_connected_mnist.png', dpi=300)
    print(f"\nSaved: {save_path}/fully_connected_mnist.png")
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Plot display interrupted. Plot saved to file.")
        plt.close('all')

# === MAIN ===
if __name__ == "__main__":
    print("Building fully connected layer circuit with MNIST...")
    circuit, initial_conditions = build_circuit()
    
    print("Running simulation...")
    analysis = run_simulation(circuit, initial_conditions)
    
    print("Extracting data...")
    data = extract_data(analysis)
    
    print_results(data)
    plot_results(data)
