"""Fully Connected Layer - 3 Neurotransistors to 1 Output Neuron"""

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from memcap_model import get_memcap_subcircuit

mpl.rcParams['axes.formatter.useoffset'] = False

# === PARAMETERS ===
memristor = 'MEMCAP'
memristor_subcircuit = get_memcap_subcircuit()

# Array size for each neurotransistor
N = 3  # rows
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
NMOS_Vto = 0.2
PMOS_Vto = -0.2
VDD = 1.0

# Pulse parameters
pulse_voltage = 1
pulse_rise = 100e-9
pulse_fall = 100e-9
pulse_on = 3e-6
pulse_period = 33e-6
Ncycles = 20
total_time = 2e-3

# Delays for 3 neurons (staggered)
delays_neuron = [
    [0, 100e-6, 200e-6],        # Neuron 0: T1=0, T2=100u, T3=200u
    [100e-6, 200e-6, 300e-6],   # Neuron 1: T1=100u, T2=200u, T3=300u
    [200e-6, 300e-6, 400e-6],   # Neuron 2: T1=200u, T2=300u, T3=400u
]

# Initial memristor states for each neuron
x0_neurons = [
    np.array([[0.15, 0.10, 0.20], [0.10, 0.20, 0.20], [0.10, 0.10, 0.20]]),
    np.array([[0.15, 0.10, 0.20], [0.10, 0.20, 0.20], [0.10, 0.10, 0.20]]),
    np.array([[0.15, 0.10, 0.20], [0.10, 0.20, 0.20], [0.10, 0.10, 0.20]]),
]

# Output neuron memristor states (3 inputs, 3 outputs)
x0_output = np.array([[0.15, 0.10, 0.20], [0.10, 0.20, 0.20], [0.10, 0.10, 0.20]])

rel_tol = 1e-4

# === PULSE GENERATOR ===
def generate_pulse_train(delay):
    times = [0]
    voltages = [0]
    if delay > pulse_rise:
        times.append(delay - pulse_rise)
        voltages.append(0)
    for k in range(Ncycles):
        t0 = delay + k * pulse_period
        if t0 >= total_time:
            break
        if t0 > times[-1]:
            times.append(t0)
            voltages.append(0)
        if t0 + pulse_rise > times[-1]:
            times.append(t0 + pulse_rise)
            voltages.append(pulse_voltage)
        t_fall = t0 + pulse_on - pulse_fall
        t_end = t0 + pulse_on
        if t_fall > times[-1]:
            times.append(t_fall)
            voltages.append(pulse_voltage)
        if t_end > times[-1]:
            times.append(t_end)
            voltages.append(0)
    if times[-1] < total_time:
        times.append(total_time)
        voltages.append(0)
    return times, voltages

# === BUILD CIRCUIT ===
def build_circuit():
    circuit = Circuit('Fully Connected Layer - 3 Neurons to 1 Output')
    initial_conditions = {}
    
    # MOSFET models
    circuit.model('NM_NMOS', 'nmos', LEVEL=1, L=26e-6, W=94e-6, Vto=NMOS_Vto, Tox=22e-9)
    circuit.model('NM_PMOS', 'pmos', LEVEL=1, L=26e-6, W=94e-6, Vto=PMOS_Vto, Tox=22e-9)
    
    # Power supply
    circuit.V('VDD', 'vdd', circuit.gnd, VDD)
    
    # =====================================================
    # BUILD 3 INPUT NEUROTRANSISTORS
    # =====================================================
    for neuron in range(num_neurons):
        prefix = f'n{neuron}_'
        delays = delays_neuron[neuron]
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
        
        # Input sources
        for i in range(N):
            times, voltages = generate_pulse_train(delays[i])
            pwl_values = [(t, v) for t, v in zip(times, voltages)]
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
    
    # Output neuron has 3 rows (one from each input neuron)
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
        # Connect v_out of neuron i to row i of output neuron
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
        circuit.raw_spice += f".probe I(R{prefix}Rout)\n"
    
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
    return simulator.transient(step_time=pulse_period, end_time=total_time)

# === EXTRACT DATA ===
def extract_data(analysis):
    data = {
        't': np.array(analysis.time),
        'V_neuron_out': [np.array(analysis[f'n{i}_v_out']) for i in range(num_neurons)],
        'V_gate_out': [np.array(analysis[f'out_col_{num_neurons-1}_{j}']) for j in range(M)],
        'V_cm_in_out': np.array(analysis['cm_in_out']),
        'V_out_final': np.array(analysis['v_out_final']),
        'I_Rout2': np.array(analysis['rrout2']),
        'I_Rout3': np.array(analysis['rrout3']),
    }
    return data

# === PRINT RESULTS ===
def print_results(data):
    print("\n" + "="*60)
    print("FULLY CONNECTED LAYER RESULTS")
    print("="*60)
    for i in range(num_neurons):
        print(f"V(neuron_{i}_out) range: {np.min(data['V_neuron_out'][i])*1e3:.1f} - {np.max(data['V_neuron_out'][i])*1e3:.1f} mV")
    print("-"*60)
    for j in range(M):
        print(f"V(gate_out_{j}) max: {np.max(data['V_gate_out'][j])*1e3:.1f} mV")
    print("-"*60)
    print(f"V(Rout2) range: {np.min(data['V_cm_in_out'])*1e3:.1f} - {np.max(data['V_cm_in_out'])*1e3:.1f} mV")
    print(f"V(out) range: {np.min(data['V_out_final'])*1e3:.1f} - {np.max(data['V_out_final'])*1e3:.1f} mV")
    print(f"I(Rout2) peak: {np.max(np.abs(data['I_Rout2']))*1e6:.2f} µA")
    print(f"I(Rout3) peak: {np.max(np.abs(data['I_Rout3']))*1e9:.2f} nA")
    print("="*60)

# === PLOT RESULTS ===
def plot_results(data, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    t_ms = data['t'] * 1e3

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    fig.suptitle('Fully Connected Layer with Current Mirror', fontsize=16, fontweight='bold', y=0.995)

    # Panel 1: Input V from first 3 neurons
    axes[0].plot(t_ms, data['V_neuron_out'][0]*1e3, 'r', lw=1, label='V(n0_out)')
    axes[0].plot(t_ms, data['V_neuron_out'][1]*1e3, 'g', lw=1, label='V(n1_out)')
    axes[0].plot(t_ms, data['V_neuron_out'][2]*1e3, 'b', lw=1, label='V(n2_out)')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_xlim(0, t_ms[-1])
    axes[0].legend(loc='upper right')
    axes[0].set_title('Input Voltage from 3 Neurons', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Gate V at 4th neuron
    axes[1].plot(t_ms, data['V_gate_out'][0]*1e3, 'r', lw=1, label='V(gate0)')
    axes[1].plot(t_ms, data['V_gate_out'][1]*1e3, 'g', lw=1, label='V(gate1)')
    axes[1].plot(t_ms, data['V_gate_out'][2]*1e3, 'b', lw=1, label='V(gate2)')
    axes[1].axhline(NMOS_Vto*1e3, color='k', ls='--', lw=2, label='Vth')
    axes[1].set_ylabel('Voltage (mV)')
    axes[1].set_ylim(0, 500)
    axes[1].legend(loc='upper left')
    axes[1].set_title('Gate Voltages at 4th Neuron', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: I(Rout2)
    axes[2].plot(t_ms, np.abs(data['I_Rout2'])*1e6, 'g', lw=1)
    axes[2].set_ylabel('Current (µA)')
    axes[2].set_ylim(0, max(5, np.max(np.abs(data['I_Rout2']))*1e6*1.2))
    axes[2].set_title('I(Rout2)', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: V(Rout2)
    axes[3].plot(t_ms, data['V_cm_in_out']*1e3, 'r', lw=1)
    axes[3].set_ylabel('Voltage (mV)')
    axes[3].set_ylim(650, 850)
    axes[3].set_title('V(Rout2)', fontsize=11)
    axes[3].grid(True, alpha=0.3)

    # Panel 5: V(Rout3)
    axes[4].plot(t_ms, data['V_out_final']*1e3, 'b', lw=1)
    axes[4].set_ylabel('Voltage (mV)')
    axes[4].set_ylim(0, 1100)
    axes[4].set_title('V(Rout3)', fontsize=11)
    axes[4].grid(True, alpha=0.3)

    # Panel 6: I(Rout3)
    axes[5].plot(t_ms, np.abs(data['I_Rout3'])*1e6, 'b', lw=1)
    axes[5].set_ylabel('Current (µA)')
    axes[5].set_xlabel('Time (ms)')
    axes[5].set_ylim(0, max(2, np.max(np.abs(data['I_Rout3']))*1e6*1.2))
    axes[5].set_title('I(Rout3)', fontsize=11)
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'{save_path}/fully_connected_layer.png', dpi=300)
    print(f"\nSaved: {save_path}/fully_connected_layer.png")
    plt.show()


# === MAIN ===
if __name__ == "__main__":
    print("Building fully connected layer circuit...")
    circuit, initial_conditions = build_circuit()
    
    print("Running simulation...")
    analysis = run_simulation(circuit, initial_conditions)
    
    print("Extracting data...")
    data = extract_data(analysis)
    
    print_results(data)
    plot_results(data)
