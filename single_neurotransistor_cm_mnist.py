"""Single Neuron with PMOS Current Mirror - MNIST Input"""

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from memcap_model import get_memcap_subcircuit
from MNIST import create_MNIST_pulse_train, load_mnist

mpl.rcParams['axes.formatter.useoffset'] = False

# === PARAMETERS ===
memristor = 'MEMCAP'
memristor_subcircuit = get_memcap_subcircuit()
N = 6                          # 6 rows for MNIST
M = 3                          # 3 columns
C_cross = 10e-12
C_gb = 100e-12
R_wire_wl = 0.33
R_wire_bl = 60
Rload = 1
Ropen = 1e9
Rs = 1
Rout1 = 1e3
Rt = 10
NMOS_Vto = 0.15
PMOS_Vto = -0.2
VDD = 1.0
pulse_voltage = 1
pulse_width = 5e-6
pulse_slope = 200e-9
rel_tol = 1e-4

# Initial memristor states
xmin = 0.1
xmax = 0.284
seed = 42
np.random.seed(seed)
x0 = np.random.uniform(low=xmin, high=xmax, size=(N, M))

# === LOAD MNIST ===
print("Loading MNIST...")
imx, imy = load_mnist('raw', kind='train')

digit = 7

index = 0

pulse_trains, digit_image = create_MNIST_pulse_train(
    imx, imy, N, pulse_voltage, pulse_width, pulse_slope,
    selected_digits=[digit], do_plot=None, specific_image_index=index
)

times = pulse_trains[digit]['times']
voltages = pulse_trains[digit]['voltages']
total_time = times[0][-1]

# === BUILD CIRCUIT ===
def build_circuit():
    circuit = Circuit('Single Neuron with PMOS Current Mirror - MNIST')
    initial_conditions = {}
    
    circuit.model('NM_NMOS', 'nmos', LEVEL=1, L=2e-6, W=94e-6, Vto=NMOS_Vto, Tox=22e-9)
    circuit.model('NM_PMOS', 'pmos', LEVEL=1, L=2e-6, W=94e-6, Vto=PMOS_Vto, Tox=22e-9)
    circuit.V('VDD', 'vdd', circuit.gnd, VDD)
    
    # Wordlines
    for i in range(N):
        for j in range(M - 1):
            circuit.R(f'wire_row_{i}_{j}', f'row_{i}_{j}', f'row_{i}_{j+1}', R_wire_wl)
    
    # Bitlines
    for j in range(M):
        for i in range(N - 1):
            circuit.R(f'wire_col_{i}_{j}', f'col_{i}_{j}', f'col_{i+1}_{j}', R_wire_bl)
    
    # Memristors
    for i in range(N):
        for j in range(M):
            circuit.X(f'M_{i}_{j}', memristor, f'row_{i}_{j}', f'col_{i}_{j}', f'sv_{i}_{j}')
            initial_conditions[f'sv_{i}_{j}'] = x0[i, j]
            circuit.C(f'C_cross_{i}_{j}', f'row_{i}_{j}', f'col_{i}_{j}', C_cross)
    
    # Input sources from MNIST
    for i in range(N):
        pwl_values = [(t, v) for t, v in zip(times[i], voltages[i])]
        circuit.R(f'R_load_left_{i}', f'row_{i}_0', f'left_{i}', Rload)
        circuit.PieceWiseLinearVoltageSource(f'V_left_{i}', f'left_{i}', circuit.gnd, values=pwl_values)
    
    # Right side open
    for i in range(N):
        circuit.R(f'R_load_right_{i}', f'row_{i}_{M-1}', f'right_{i}', Ropen)
        circuit.V(f'V_right_{i}', f'right_{i}', circuit.gnd, 0)
    
    # Top side open
    for j in range(M):
        circuit.R(f'R_load_top_{j}', f'col_0_{j}', f'top_{j}', Ropen)
        circuit.V(f'V_top_{j}', f'top_{j}', circuit.gnd, 0)
    
    # NMOS transistors
    for j in range(M):
        circuit.MOSFET(f'M_bot_{j}', f'drain_{j}', f'col_{N-1}_{j}', f'source_{j}', 'bulk', model='NM_NMOS')
        circuit.C(f'C_gb_{j}', f'col_{N-1}_{j}', 'bulk', C_gb)
        circuit.R(f'R_gb_{j}', f'col_{N-1}_{j}', 'bulk', 10e6)
        circuit.R(f'R_db_{j}', f'drain_{j}', 'bulk', 10e6)
        circuit.R(f'R_sb_{j}', f'source_{j}', 'bulk', 10e6)
        if j == 0:
            circuit.R('R_out_left', 'out_left', f'source_{j}', 1)
            circuit.V('V_out_left', 'out_left', circuit.gnd, 0)
        else:
            circuit.R(f'Rt_{j}', f'drain_{j-1}', f'source_{j}', Rt)
    
    circuit.R('R_bulk', 'bulk', circuit.gnd, 0.1)
    circuit.R('Rout', f'drain_{M-1}', 'cm_in', Rs)
    
    # Current mirror
    circuit.MOSFET('M5', 'cm_in', 'cm_in', 'vdd', 'vdd', model='NM_PMOS')
    circuit.MOSFET('M4', 'v_out', 'cm_in', 'vdd', 'vdd', model='NM_PMOS')
    circuit.R('Rout1', 'v_out', circuit.gnd, Rout1)
    
    # Spice directives
    circuit.raw_spice += memristor_subcircuit
    for i in range(N):
        circuit.raw_spice += f".save v(left_{i})\n"
    for j in range(M):
        circuit.raw_spice += f".save v(col_{N-1}_{j})\n"
        circuit.raw_spice += f".save v(drain_{j})\n"
    circuit.raw_spice += ".save v(cm_in)\n"
    circuit.raw_spice += ".save v(v_out)\n"
    circuit.raw_spice += ".probe I(RRout)\n"
    circuit.raw_spice += ".probe I(RRout1)\n"
    for i in range(N):
        for j in range(M):
            circuit.raw_spice += f".save v(sv_{i}_{j})\n"
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
        'V_input': [np.array(analysis[f'left_{i}']) for i in range(N)],
        'V_gate': [np.array(analysis[f'col_{N-1}_{j}']) for j in range(M)],
        'V_drain': [np.array(analysis[f'drain_{j}']) for j in range(M)],
        'V_cm_in': np.array(analysis['cm_in']),
        'V_out': np.array(analysis['v_out']),
        'I_Rout': np.array(analysis['rrout']),
        'I_Rout1': np.array(analysis['rrout1'])
    }

# === PRINT RESULTS ===
def print_results(data):
    print("\n" + "="*50)
    print(f"RESULTS - MNIST Digit {digit}")
    print("="*50)
    for j in range(M):
        print(f"V(gate_{j}) max: {np.max(data['V_gate'][j])*1e3:.1f} mV")
    print(f"V(Rout) range: {np.min(data['V_cm_in'])*1e3:.1f} - {np.max(data['V_cm_in'])*1e3:.1f} mV")
    print(f"V(Rout1) range: {np.min(data['V_out'])*1e3:.1f} - {np.max(data['V_out'])*1e3:.1f} mV")
    print(f"I(Rout) peak:  {np.max(np.abs(data['I_Rout']))*1e6:.2f} µA")
    print(f"I(Rout1) peak: {np.max(np.abs(data['I_Rout1']))*1e9:.2f} nA")
    print("="*50)

def plot_results(data, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    t_us = data['t'] * 1e6

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    fig.suptitle('Neurotransistor with Current Mirror', fontsize=14, fontweight='bold')

    # Panel 1: Input Pulse
    colors_input = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FF9800', '#F44336']
    for i in range(N):
        axes[0].plot(t_us, data['V_input'][i], color=colors_input[i % len(colors_input)], lw=0.8)
    axes[0].set_ylabel('V(in) [V]')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    # Digit inset
    inset = axes[0].inset_axes([0.92, 0.3, 0.07, 0.6])
    inset.imshow(digit_image, cmap='gray')
    inset.set_title(f'{digit}', fontsize=9)
    inset.axis('off')

    # Panel 2: Gate Voltages
    axes[1].plot(t_us, data['V_gate'][0]*1e3, '#F44336', lw=1)
    axes[1].plot(t_us, data['V_gate'][1]*1e3, '#4CAF50', lw=1)
    axes[1].plot(t_us, data['V_gate'][2]*1e3, '#2196F3', lw=1)
    axes[1].axhline(NMOS_Vto*1e3, color='k', ls='--', lw=1.5)
    axes[1].set_ylabel('V(gate) [mV]')
    axes[1].set_ylim(0, 400)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: V(Rout)
    V_cm = data['V_cm_in'] * 1e3
    v_min = np.min(V_cm)
    v_max = np.max(V_cm)
    margin = (v_max - v_min) * 0.2
    axes[2].plot(t_us, V_cm, '#9C27B0', lw=1)
    axes[2].set_ylabel('V(Rout) [mV]')
    axes[2].set_ylim(v_min - margin, v_max + margin)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: V(Rout1)
    V_out = data['V_out'] * 1e3
    axes[3].plot(t_us, V_out, '#2196F3', lw=1)
    axes[3].set_ylabel('V(Rout1) [mV]')
    axes[3].set_ylim(0, max(100, np.max(V_out) * 1.1))
    axes[3].grid(True, alpha=0.3)

    # Panel 5: I(Rout)
    I_rout_uA = np.abs(data['I_Rout']) * 1e6
    axes[4].plot(t_us, I_rout_uA, '#4CAF50', lw=1)
    axes[4].set_ylabel('I(Rout) [µA]')
    axes[4].set_ylim(0, max(1, np.max(I_rout_uA) * 1.2))
    axes[4].grid(True, alpha=0.3)

    # Panel 6: I(Rout1)
    I_rout1_nA = np.abs(data['I_Rout1']) * 1e9
    axes[5].plot(t_us, I_rout1_nA, '#E91E63', lw=1)
    axes[5].set_ylabel('I(Rout1) [nA]')
    axes[5].set_xlabel('Time [µs]')
    axes[5].set_ylim(0, max(100, np.max(I_rout1_nA) * 1.1))
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/neuron_mnist_digit{digit}.png', dpi=300)
    plt.show()


def print_results(data):
    print("\n" + "="*50)
    print(f"RESULTS - MNIST Digit {digit}")
    print("="*50)
    for j in range(M):
        print(f"V(gate_{j}) max: {np.max(data['V_gate'][j])*1e3:.1f} mV")
    print(f"Vth: {NMOS_Vto*1e3:.1f} mV")
    print(f"V(Rout) range: {np.min(data['V_cm_in'])*1e3:.1f} - {np.max(data['V_cm_in'])*1e3:.1f} mV")
    print(f"V(Rout1) range: {np.min(data['V_out'])*1e3:.1f} - {np.max(data['V_out'])*1e3:.1f} mV")
    print(f"I(Rout) peak: {np.max(np.abs(data['I_Rout']))*1e6:.2f} µA")
    print("="*50)


# === MAIN ===
if __name__ == "__main__":
    print("Building circuit...")
    circuit, initial_conditions = build_circuit()
    
    print("Running simulation...")
    analysis = run_simulation(circuit, initial_conditions)
    
    print("Extracting data...")
    data = extract_data(analysis)
    
    print_results(data)
    plot_results(data)
