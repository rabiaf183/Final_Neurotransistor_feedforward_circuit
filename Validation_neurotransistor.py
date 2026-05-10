"""
Validates the neurotransistor circuit for spiking neural networks.
Equations:
    1. Synaptic Current: I_syn = I_res + I_cap = (Vt - Vm)/Rs + Cp·dVt/dt
    2. Membrane Voltage: C_gb · dVcol/dt = Σ I_syn
    3. Memcapacitor Conductance: G(x, Vm) = Poole-Frenkel model
    4. NMOS Current: EKV model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from PySpice.Spice.Netlist import Circuit


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0: PARAMETER CLASS
# ══════════════════════════════════════════════════════════════════════════════
class P:
    # --- Physical Constants ---
    VT = 0.02585              # Thermal voltage (~25.85mV)
    
    # --- MEMCAP Parameters (NamLab HfOx) ---
    Ar, As = 4.7447e-8, 1.1253e-8
    Br, Bs = 2.6831, 9.3348
    Rs = 278                  # Series resistance (Ω)
    Rp = 8e6                  # Parallel leakage resistance (Ω)
    Cp = 10e-12               # Parallel capacitance (10 pF) ← CRITICAL FOR I_cap!
    
    # --- Capacitances ---
    C_gb = 90e-12             # Explicit SPICE capacitance
    C_gb_math = 100e-12       # Math model (90pF explicit + 10pF BSIM4 intrinsic)
    C_out = 5e-12             # Mirror output capacitance
    
    # ═══════════════════════════════════════════════════════════════
    # OUTPUT NMOS (IHM_NMOS)
    # ═══════════════════════════════════════════════════════════════
    bot_W = 180e-6            # W = 180u
    bot_L = 60e-6             # L = 60u
    bot_Level = 14            # LEVEL = 14
    bot_Vth = 0.3             # Vto = 0.3V
    bot_Tox = 22e-9           # Tox = 22n
    
    # ═══════════════════════════════════════════════════════════════
    # RESET NMOS
    # ═══════════════════════════════════════════════════════════════
    res_W = 100e-6            # W = 100u
    res_L = 10e-6             # L = 10u
    res_Level = 14            # LEVEL = 14
    res_Vth = 0.9             # Vto = 0.9V
    res_Tox = 3e-9            # Tox = 3n
    
    # --- Sweep Parameters ---
    V_sweep_max = 1.5
    V_sweep_step = 0.01
    
    # NMOS sweep ranges
    Vd_arr_bot = [0.4, 0.6, 0.8, 1.0]
    Vg_arr_bot = [0.4, 0.6, 0.8, 1.0]
    
    # Reset NMOS sweep ranges
    Vd_arr_res = [0.6, 0.9, 1.2]
    Vg_arr_res = [0.6, 0.9, 1.2]
    
    # --- Input Pulse Parameters ---
    V_pulse = 1.0
    T_on = 2e-6               # Ton = 2µs
    T_period = 10e-6          # Tperiod = 10µs
    
    # --- Current Mirror ---
    GAIN_MIRROR = 5.0
    R_out_mirr = 100e3


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CORE MATHEMATICAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def safe_log_exp(x):
    """Numerically stable ln(1 + exp(x))"""
    return np.where(x > 50, x, np.log(1 + np.exp(np.clip(x, -100, 50))))


def ekv_model(Vg, Vd, Vs, Is, Vth, kappa, lam):
    """EKV NMOS drain current model."""
    Vg = np.maximum(Vg, 0)
    Vd = np.maximum(Vd, 0)
    Vs = np.maximum(Vs, 0)
    
    term_fwd = (kappa * (Vg - Vth) - Vs) / (2 * P.VT)
    term_rev = (kappa * (Vg - Vth) - Vd) / (2 * P.VT)
    
    I_fwd = safe_log_exp(term_fwd)**2
    I_rev = safe_log_exp(term_rev)**2
    
    Vds = np.maximum(Vd - Vs, 0)
    return Is * (1 + lam * Vds) * (I_fwd - I_rev)


def wrapper_ekv_fit(X, Is, Vth, kappa, lam):
    """Wrapper for scipy curve_fit"""
    Vg, Vd, Vs = X
    return ekv_model(Vg, Vd, Vs, Is, Vth, kappa, lam)


def G_memristor(x, Vm):
    """Memcapacitor conductance (Poole-Frenkel model)"""
    Vm = np.atleast_1d(Vm)
    Vm_safe = np.where(np.abs(Vm) < 1e-12, 1e-12, Vm)
    x_safe = max(x, 1e-12)
    
    return (P.Ar * x * np.exp(P.Br * np.sign(Vm_safe) * np.sqrt(np.abs(Vm_safe) / x_safe)) +
            P.As * x * np.exp(-P.Bs * np.sign(Vm_safe) * np.sqrt(np.abs(Vm_safe))))


def solve_memcap_internal(Vt, x):
    """Solve for internal memcapacitor voltage"""
    if np.abs(Vt) < 1e-12:
        return 0.0
    
    def residual(Vm):
        Gm = G_memristor(x, Vm[0])
        Gm = Gm[0] if isinstance(Gm, np.ndarray) else Gm
        return [(Vt - Vm[0]) / P.Rs - Vm[0] / P.Rp - Vm[0] * Gm]
    
    return float(fsolve(residual, [Vt / 2], xtol=1e-10)[0])


def I_memcap(V_in, V_col, x):
    """Calculate RESISTIVE synaptic current (I_res only)"""
    Vt = V_in - V_col
    if np.abs(Vt) < 1e-12:
        return 0.0
    Vm = solve_memcap_internal(Vt, x)
    return (Vt - Vm) / P.Rs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: NMOS EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_and_fit_nmos(model_name, W, L, Level, Vth0, Tox, Vd_arr, Vg_arr, C_explicit, C_math):
    """Extract NMOS characteristics and fit EKV parameters."""
    
    X_fit, Y_fit = [], []
    data_IdVg, data_IdVd = {}, {}
    
    # ----- Id-Vg Transfer Characteristics -----
    print(f"\nExtracting Id-Vg for {model_name}...")
    for vd in Vd_arr:
        ckt = Circuit(f'{model_name}_IdVg_Vd{vd}'.replace('.', 'p'))
        
        if Level == 14:
            ckt.model('TEST_NMOS', 'nmos', LEVEL=Level, L=L, W=W, VTH0=Vth0, TOXE=Tox)
        else:
            ckt.model('TEST_NMOS', 'nmos', LEVEL=Level, L=L, W=W, Vto=Vth0, Tox=Tox)
        
        ckt.V('d', 'drain', ckt.gnd, vd)
        ckt.V('g', 'gate', ckt.gnd, 0)
        ckt.MOSFET('M1', 'drain', 'gate', ckt.gnd, ckt.gnd, model='TEST_NMOS')
        
        res = ckt.simulator().dc(Vg=slice(0, P.V_sweep_max, P.V_sweep_step))
        Vg_data = np.array(res.sweep)
        Id_data = -np.array(res.Vd)
        
        data_IdVg[vd] = (Vg_data, Id_data)
        
        for vg, id_ in zip(Vg_data, Id_data):
            X_fit.append([vg, vd, 0.0])
            Y_fit.append(id_)
        
        print(f"  Vd={vd}V: {len(Vg_data)} points")
    
    # ----- Id-Vd Output Characteristics -----
    print(f"\nExtracting Id-Vd for {model_name}...")
    for vg in Vg_arr:
        ckt = Circuit(f'{model_name}_IdVd_Vg{vg}'.replace('.', 'p'))
        
        if Level == 14:
            ckt.model('TEST_NMOS', 'nmos', LEVEL=Level, L=L, W=W, VTH0=Vth0, TOXE=Tox)
        else:
            ckt.model('TEST_NMOS', 'nmos', LEVEL=Level, L=L, W=W, Vto=Vth0, Tox=Tox)
        
        ckt.V('d', 'drain', ckt.gnd, 0)
        ckt.V('g', 'gate', ckt.gnd, vg)
        ckt.MOSFET('M1', 'drain', 'gate', ckt.gnd, ckt.gnd, model='TEST_NMOS')
        
        res = ckt.simulator().dc(Vd=slice(0, P.V_sweep_max, P.V_sweep_step))
        Vd_data = np.array(res.sweep)
        Id_data = -np.array(res.Vd)
        
        data_IdVd[vg] = (Vd_data, Id_data)
        
        for vd, id_ in zip(Vd_data, Id_data):
            X_fit.append([vg, vd, 0.0])
            Y_fit.append(id_)
        
        print(f"  Vg={vg}V: {len(Vd_data)} points")
    
    # ----- C-V Characteristics -----
    print(f"\nExtracting C-V for {model_name}...")
    Vgs_cv = np.linspace(-2.0, 2.0, 51)
    f_meas = 1e6
    omega = 2 * np.pi * f_meas
    Cap_array = []
    
    for v_bias in Vgs_cv:
        ckt = Circuit(f'{model_name}_CV_{v_bias}'.replace('.', 'p').replace('-', 'm'))
        
        if Level == 14:
            ckt.model('TEST_NMOS', 'nmos', LEVEL=Level, L=L, W=W, VTH0=Vth0, TOXE=Tox)
        else:
            ckt.model('TEST_NMOS', 'nmos', LEVEL=Level, L=L, W=W, Vto=Vth0, Tox=Tox)
        
        ckt.raw_spice += f"Vgate gate 0 DC {v_bias} AC 1\n"
        ckt.MOSFET('M1', ckt.gnd, 'gate', ckt.gnd, ckt.gnd, model='TEST_NMOS')
        ckt.C('gb', 'gate', ckt.gnd, C_explicit)
        
        res = ckt.simulator().ac(start_frequency=f_meas, stop_frequency=f_meas,
                                  number_of_points=1, variation='dec')
        I_gate = np.array(res.branches['vgate'])[0]
        Cap_array.append(np.abs(np.imag(I_gate)) / omega)
    
    Cap_array = np.array(Cap_array)
    data_CV = (Vgs_cv, Cap_array)
    
    # ----- Curve Fitting -----
    print(f"\nFitting EKV parameters for {model_name}...")
    X_fit = np.array(X_fit).T
    Y_fit = np.array(Y_fit)
    
    p0 = [1e-6, Vth0, 0.7, 0.1]
    bounds = ([1e-9, 0.1, 0.1, 0.0], [1e-3, 1.5, 1.5, 2.0])
    
    fitted_params, _ = curve_fit(wrapper_ekv_fit, X_fit, Y_fit, p0=p0, bounds=bounds)
    
    Is, Vth, kappa, lam = fitted_params
    print(f"\n  Fitted: Is={Is:.2e}A, Vth={Vth:.3f}V, κ={kappa:.3f}, λ={lam:.3f}")
    
    return fitted_params, data_IdVg, data_IdVd, data_CV, C_math


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MEMCAP VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_memcap():
    """Validate memcapacitor current equation"""
    print("\n" + "="*60)
    print("MEMCAP VALIDATION")
    print("="*60)

    V_in_array = np.linspace(0.01, 1.5, 50)
    V_col = 0.3
    x_states = [0.1, 0.15, 0.2, 0.25]
    
    results = {}
    for x in x_states:
        I_syn = np.array([I_memcap(v_in, V_col, x) for v_in in V_in_array])
        results[x] = I_syn
        idx_1v = np.argmin(np.abs(V_in_array - 1.0))
        print(f"  x={x}: I_syn at V_in=1V = {I_syn[idx_1v]*1e6:.2f} µA")
    
    return V_in_array, results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SINGLE COLUMN LIF VALIDATION 
# ══════════════════════════════════════════════════════════════════════════════
def validate_single_column(params_bot):
    """
    Validate single column LIF behavior.
    
  Equation: 
        I_syn = I_res + I_cap = (Vt - Vm)/Rs + Cp·dVt/dt
        C_gb · dVcol/dt = I_syn
    """
    x = 0.2
    V_pulse = P.V_pulse
    Is, Vth, kappa, lam = params_bot
    
    memcap_subckt = f"""
.SUBCKT MEMCAP TE BE SV
.param Ar={P.Ar} As={P.As} Br={P.Br} Bs={P.Bs} Rp={P.Rp} Rs={P.Rs} Cp={P.Cp}
.func Gm(x,Vm) {{Ar*x*exp(Br*sgn(Vm)*sqrt(abs(Vm)/x))+As*x*exp(-Bs*sgn(Vm)*sqrt(abs(Vm)))}}
Bmem IE BE i=V(IE,BE)*Gm(V(SV,0),V(IE,BE))
Rms TE IE {{Rs}}
Rmp IE BE {{Rp}}
Cmp TE BE {{Cp}}
Cx SV 0 1
.IC V(SV)={x}
.ENDS"""
    
    ckt = Circuit('SingleColumnLIF')
    ckt.raw_spice += memcap_subckt
    ckt.V('in', 'in', ckt.gnd, f'PULSE(0 {V_pulse} 10u 1u 1u 80u 200u)')
    ckt.V('dd', 'vdd', ckt.gnd, 1.0)
    ckt.X('mem', 'MEMCAP', 'in', 'col', 'sv')
    ckt.C('gb', 'col', ckt.gnd, P.C_gb)
    
    if P.bot_Level == 14:
        ckt.model('NMOS_BOT', 'nmos', LEVEL=P.bot_Level, L=P.bot_L, W=P.bot_W, 
                  VTH0=P.bot_Vth, TOXE=P.bot_Tox)
    else:
        ckt.model('NMOS_BOT', 'nmos', LEVEL=P.bot_Level, L=P.bot_L, W=P.bot_W, 
                  Vto=P.bot_Vth, Tox=P.bot_Tox)
    
    ckt.R('out', 'vdd', 'drain', 200e3)
    ckt.MOSFET('M1', 'drain', 'col', ckt.gnd, ckt.gnd, model='NMOS_BOT')
    
    res = ckt.simulator().transient(step_time=0.5e-6, end_time=150e-6)
    t_spice = np.array(res.time)
    V_spice = np.array(res['col'])
    
    print(f"  SPICE: max V_col = {np.max(V_spice):.3f} V")
    
    # ----- Math Model (with I_cap) -----
    print("\nRunning Math model (with I_cap)...")
    
    t_math = np.linspace(0, 150e-6, 1000)
    dt = t_math[1] - t_math[0]
    V_math = np.zeros_like(t_math)
    I_out = np.zeros_like(t_math)
    
    # Track previous terminal voltage for I_cap
    Vt_prev = 0.0
    
    for i in range(1, len(t_math)):
        V_in = V_pulse if 10e-6 <= t_math[i-1] <= 90e-6 else 0.0
        
        # Terminal voltage
        Vt = V_in - V_math[i-1]
        
        # Resistive current
        Vm = solve_memcap_internal(Vt, x)
        I_res = (Vt - Vm) / P.Rs
        
        # Capacitive current 
        I_cap = P.Cp * (Vt - Vt_prev) / dt
        
        # Total synaptic current
        I_syn = I_res + I_cap
        
        # Membrane equation: C_gb * dV/dt = I_syn
        dV_dt = I_syn / P.C_gb_math
        V_math[i] = max(0, V_math[i-1] + dV_dt * dt)
        
        # Store for next iteration
        Vt_prev = Vt
        
        # Output current 
        I_out[i] = ekv_model(V_math[i], 1.0, 0, Is, Vth, kappa, lam)
    
    print(f"  Math: max V_col = {np.max(V_math):.3f} V")
    
    return t_spice, V_spice, t_math, V_math, I_out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MULTI-INPUT COLUMN LIF 
# ══════════════════════════════════════════════════════════════════════════════
def validate_multi_column(params_bot):
    """
    Validate 3-input column LIF.
    
    Equation:
        I_syn = I_res + I_cap for EACH row
        C_gb · dVcol/dt = Σ I_syn,i
    """
    
    Is, Vth, kappa, lam = params_bot
    
    x_states = [0.15, 0.2, 0.25]
    delays = [10e-6, 30e-6, 50e-6]
    pulse_width = 40e-6
    V_pulse = P.V_pulse
    
    t = np.linspace(0, 150e-6, 1000)
    dt = t[1] - t[0]
    V_col = np.zeros_like(t)
    I_out = np.zeros_like(t)
    
    # Track previous terminal voltage for EACH row
    Vt_prev = [0.0, 0.0, 0.0]
    
    print(f"  Row 0: x={x_states[0]}, delay={delays[0]*1e6:.0f}µs")
    print(f"  Row 1: x={x_states[1]}, delay={delays[1]*1e6:.0f}µs")
    print(f"  Row 2: x={x_states[2]}, delay={delays[2]*1e6:.0f}µs")
    
    for i in range(1, len(t)):
        I_syn_total = 0.0
        
        for row in range(3):
            t_current = t[i-1]
            pulse_on = delays[row] <= t_current <= (delays[row] + pulse_width)
            V_in = V_pulse if pulse_on else 0.0
            
            # Terminal voltage
            Vt = V_in - V_col[i-1]
            
            # Resistive current
            Vm = solve_memcap_internal(Vt, x_states[row])
            I_res = (Vt - Vm) / P.Rs
            
            # Capacitive current
            I_cap = P.Cp * (Vt - Vt_prev[row]) / dt
            
            # Total for this row
            I_syn_total += (I_res + I_cap)
            
            # Store for next iteration
            Vt_prev[row] = Vt
        
        # Membrane equation: C_gb * dV/dt = Σ I_syn
        dV_dt = I_syn_total / P.C_gb_math
        V_col[i] = max(0, V_col[i-1] + dV_dt * dt)
        I_out[i] = ekv_model(V_col[i], 1.0, 0, Is, Vth, kappa, lam)
    
    print(f"\n  Max V_col = {np.max(V_col):.3f} V")
    
    return t, V_col, I_out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: PULSE WIDTH ANALYSIS 
# ══════════════════════════════════════════════════════════════════════════════
def analyze_pulse_width(params_bot, x=0.2):
    """
    Find required pulse width for MNIST encoding.
    """

    Is, Vth_fitted, kappa, lam = params_bot
    Vth_target = P.bot_Vth

    
    t_cont = np.linspace(0, 100e-6, 5000)
    dt = t_cont[1] - t_cont[0]
    V_cont = np.zeros_like(t_cont)
    
    # Track Vt_prev
    Vt_prev = 0.0
    
    for i in range(1, len(t_cont)):
        Vt = P.V_pulse - V_cont[i-1]
        
        Vm = solve_memcap_internal(Vt, x)
        I_res = (Vt - Vm) / P.Rs
        I_cap = P.Cp * (Vt - Vt_prev) / dt
        I_syn = I_res + I_cap
        
        V_cont[i] = max(0, V_cont[i-1] + I_syn / P.C_gb_math * dt)
        Vt_prev = Vt
    
    V_steady = V_cont[-1]
    
    idx_vth = np.argmax(V_cont >= Vth_target)
    t_to_vth = t_cont[idx_vth] if idx_vth > 0 else float('inf')
    
    print(f"  V_steady = {V_steady:.3f}V")
    print(f"  Time to reach Vth={Vth_target}V: {t_to_vth*1e6:.1f} µs")
    
    if V_steady < Vth_target:
        print(f"\n   V_steady < Vth, neuron can not fire!")
    
    # ----- Pulsed Input Analysis -----
    print(f"\n2. PULSED INPUT (Ton={P.T_on*1e6:.0f}µs, Period={P.T_period*1e6:.0f}µs):")
    
    t_pulsed = np.linspace(0, 500e-6, 25000)
    dt_p = t_pulsed[1] - t_pulsed[0]
    V_pulsed = np.zeros_like(t_pulsed)
    
    # Track Vt_prev
    Vt_prev_p = 0.0
    
    pulses_to_vth = 0
    reached_vth = False
    pulse_count = 0
    last_pulse_state = False
    
    for i in range(1, len(t_pulsed)):
        t_in_period = t_pulsed[i-1] % P.T_period
        pulse_on = t_in_period < P.T_on
        V_in = P.V_pulse if pulse_on else 0.0
        
        if pulse_on and not last_pulse_state:
            pulse_count += 1
        last_pulse_state = pulse_on
        
        Vt = V_in - V_pulsed[i-1]
        
        Vm = solve_memcap_internal(Vt, x)
        I_res = (Vt - Vm) / P.Rs
        I_cap = P.Cp * (Vt - Vt_prev_p) / dt_p
        I_syn = I_res + I_cap
        
        V_pulsed[i] = max(0, V_pulsed[i-1] + I_syn / P.C_gb_math * dt_p)
        Vt_prev_p = Vt
        
        if not reached_vth and V_pulsed[i] >= Vth_target:
            pulses_to_vth = pulse_count
            reached_vth = True
            print(f"\n   Reached Vth after {pulses_to_vth} pulses!")
            print(f"    Total time = {t_pulsed[i]*1e6:.1f} µs")
    
    if not reached_vth:
        V_max = np.max(V_pulsed)
        print(f"\n  Did NOT reach Vth within {pulse_count} pulses")
        print(f"    Max V_col = {V_max:.3f}V, Gap = {Vth_target - V_max:.3f}V")
        
        if V_max > 0:
            V_per_pulse = V_max / pulse_count
            pulses_estimated = int(np.ceil(Vth_target / V_per_pulse))
            print(f"    Estimated pulses needed ≈ {pulses_estimated}")
            pulses_to_vth = pulses_estimated
    
    return {
        't_continuous': t_cont,
        'V_continuous': V_cont,
        't_to_vth': t_to_vth,
        'V_steady': V_steady,
        'Vth_target': Vth_target,
        't_pulsed': t_pulsed,
        'V_pulsed': V_pulsed,
        'pulses_to_vth': pulses_to_vth,
        'T_on': P.T_on,
        'T_period': P.T_period
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def plot_nmos_comparison(params_bot, data_bot, params_res, data_res):
    """Plot BOTH NMOS types for comparison"""
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    colors = ['#8E44AD', '#E91E8C', '#27AE60', '#3498DB']
    
    # ----- Row 0: Output NMOS -----
    data_IdVg_bot, data_IdVd_bot, data_CV_bot, C_math_bot = data_bot
    Is_b, Vth_b, kappa_b, lam_b = params_bot
    
    for i, vd in enumerate(P.Vd_arr_bot):
        Vg, Id = data_IdVg_bot[vd]
        axs[0,0].semilogy(Vg, Id*1e6, '-', color=colors[i], lw=1.5, label=f'SPICE Vd={vd}V')
        axs[0,0].semilogy(Vg, ekv_model(Vg, vd, 0, *params_bot)*1e6, '--', color=colors[i], lw=2, alpha=0.7)
    axs[0,0].set_xlabel('Vg (V)')
    axs[0,0].set_ylabel('Id (µA)')
    axs[0,0].set_title(f'Output NMOS Id-Vg ')
    axs[0,0].legend(fontsize=7)
    axs[0,0].grid(True, alpha=0.3)
    
    for i, vg in enumerate(P.Vg_arr_bot):
        Vd, Id = data_IdVd_bot[vg]
        axs[0,1].plot(Vd, Id*1e6, '-', color=colors[i], lw=1.5, label=f'SPICE Vg={vg}V')
        axs[0,1].plot(Vd, ekv_model(vg, Vd, 0, *params_bot)*1e6, '--', color=colors[i], lw=2, alpha=0.7)
    axs[0,1].set_xlabel('Vd (V)')
    axs[0,1].set_ylabel('Id (µA)')
    axs[0,1].set_title('Output NMOS Id-Vd')
    axs[0,1].legend(fontsize=7)
    axs[0,1].grid(True, alpha=0.3)
    
    Vgs_cv, Cap_cv = data_CV_bot
    axs[0,2].plot(Vgs_cv, Cap_cv*1e12, 'b-', lw=2, label='SPICE')
    axs[0,2].axhline(C_math_bot*1e12, color='k', ls='--', lw=1.5, label=f'C_math={C_math_bot*1e12:.0f}pF')
    axs[0,2].axvline(Vth_b, color='gray', ls=':', label=f'Vth={Vth_b:.2f}V')
    axs[0,2].set_xlabel('Vgs (V)')
    axs[0,2].set_ylabel('C (pF)')
    axs[0,2].set_title('Output NMOS C-V')
    axs[0,2].legend(fontsize=7)
    axs[0,2].grid(True, alpha=0.3)
    
    # ----- Row 1: Reset NMOS -----
    data_IdVg_res, data_IdVd_res, data_CV_res, C_math_res = data_res
    Is_r, Vth_r, kappa_r, lam_r = params_res
    
    for i, vd in enumerate(P.Vd_arr_res):
        Vg, Id = data_IdVg_res[vd]
        axs[1,0].semilogy(Vg, Id*1e6, '-', color=colors[i], lw=1.5, label=f'SPICE Vd={vd}V')
        axs[1,0].semilogy(Vg, ekv_model(Vg, vd, 0, *params_res)*1e6, '--', color=colors[i], lw=2, alpha=0.7)
    axs[1,0].set_xlabel('Vg (V)')
    axs[1,0].set_ylabel('Id (µA)')
    axs[1,0].set_title(f'Reset NMOS Id-Vg ')
    axs[1,0].legend(fontsize=7)
    axs[1,0].grid(True, alpha=0.3)
    
    for i, vg in enumerate(P.Vg_arr_res):
        Vd, Id = data_IdVd_res[vg]
        axs[1,1].plot(Vd, Id*1e6, '-', color=colors[i], lw=1.5, label=f'SPICE Vg={vg}V')
        axs[1,1].plot(Vd, ekv_model(vg, Vd, 0, *params_res)*1e6, '--', color=colors[i], lw=2, alpha=0.7)
    axs[1,1].set_xlabel('Vd (V)')
    axs[1,1].set_ylabel('Id (µA)')
    axs[1,1].set_title('Reset NMOS Id-Vd')
    axs[1,1].legend(fontsize=7)
    axs[1,1].grid(True, alpha=0.3)
    
    Vgs_cv_r, Cap_cv_r = data_CV_res
    axs[1,2].plot(Vgs_cv_r, Cap_cv_r*1e12, 'r-', lw=2, label='SPICE')
    axs[1,2].axvline(Vth_r, color='gray', ls=':', label=f'Vth={Vth_r:.2f}V')
    axs[1,2].set_xlabel('Vgs (V)')
    axs[1,2].set_ylabel('C (pF)')
    axs[1,2].set_title('Reset NMOS C-V ')
    axs[1,2].legend(fontsize=7)
    axs[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure1_nmos_both.png', dpi=200)
    plt.show()


def plot_validation(params_bot, memcap_data, column_data, multi_col_data, pulse_data):
    """Plot validation results"""
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    colors = ['#8E44AD', '#E91E8C', '#27AE60', '#3498DB']
    
    Is, Vth, kappa, lam = params_bot
    Vth_target = pulse_data['Vth_target']
    
    # 1. MEMCAP Current
    V_in, mem_results = memcap_data
    for i, (x, I_syn) in enumerate(mem_results.items()):
        axs[0,0].plot(V_in, I_syn*1e6, color=colors[i], lw=2, label=f'x={x}')
    axs[0,0].axvline(0.3, color='gray', ls=':', lw=1)
    axs[0,0].set_xlabel('V_in (V)')
    axs[0,0].set_ylabel('I_syn (µA)')
    axs[0,0].set_title('MEMCAP Synaptic Current')
    axs[0,0].legend(fontsize=8)
    axs[0,0].grid(True, alpha=0.3)
    
    # 2. Single Column LIF
    t_sp, V_sp, t_m, V_m, I_out = column_data
    axs[0,1].plot(t_sp*1e6, V_sp, 'b-', lw=1.5, label='SPICE')
    axs[0,1].plot(t_m*1e6, V_m, 'r--', lw=2, label='Math (I_res + I_cap)')
    axs[0,1].axhline(Vth_target, color='gray', ls=':', label=f'Vth={Vth_target}V')
    axs[0,1].set_xlabel('Time (µs)')
    axs[0,1].set_ylabel('V_col (V)')
    axs[0,1].set_title('Single Column LIF ')
    axs[0,1].legend()
    axs[0,1].grid(True, alpha=0.3)
    
    # 3. Multi-Input Column
    t_multi, V_multi, I_multi = multi_col_data
    axs[0,2].plot(t_multi*1e6, V_multi, 'g-', lw=2)
    axs[0,2].axhline(Vth_target, color='gray', ls=':', label=f'Vth={Vth_target}V')
    axs[0,2].set_xlabel('Time (µs)')
    axs[0,2].set_ylabel('V_col (V)')
    axs[0,2].set_title('3-Input Column LIF')
    axs[0,2].legend()
    axs[0,2].grid(True, alpha=0.3)
    
    # 4. Continuous Pulse
    t_cont = pulse_data['t_continuous']
    V_cont = pulse_data['V_continuous']
    t_to_vth = pulse_data['t_to_vth']
    V_steady = pulse_data['V_steady']
    
    axs[1,0].plot(t_cont*1e6, V_cont, 'g-', lw=2)
    axs[1,0].axhline(Vth_target, color='r', ls='--', lw=1.5, label=f'Vth={Vth_target}V')
    if t_to_vth < float('inf'):
        axs[1,0].axvline(t_to_vth*1e6, color='r', ls=':', lw=1.5, label=f't={t_to_vth*1e6:.1f}µs')
    axs[1,0].axhline(V_steady, color='orange', ls='--', lw=1, label=f'V_ss={V_steady:.3f}V')
    axs[1,0].set_xlabel('Time (µs)')
    axs[1,0].set_ylabel('V_col (V)')
    axs[1,0].set_title('Continuous Pulse Input ')
    axs[1,0].legend(fontsize=8)
    axs[1,0].grid(True, alpha=0.3)
    axs[1,0].set_xlim([0, min(t_to_vth*1e6*3 if t_to_vth < float('inf') else 100, 100)])
    
    # 5. Pulsed Input
    t_pulsed = pulse_data['t_pulsed'][:5000]
    V_pulsed = pulse_data['V_pulsed'][:5000]
    pulses = pulse_data['pulses_to_vth']
    T_on = pulse_data['T_on']
    T_period = pulse_data['T_period']
    
    axs[1,1].plot(t_pulsed*1e6, V_pulsed, 'b-', lw=1.5)
    axs[1,1].axhline(Vth_target, color='r', ls='--', lw=1.5, label=f'Vth={Vth_target}V')
    
    for p in range(min(pulses + 2, 20)):
        t_start = p * T_period
        axs[1,1].axvspan(t_start*1e6, (t_start + T_on)*1e6, alpha=0.2, color='green')
    
    axs[1,1].set_xlabel('Time (µs)')
    axs[1,1].set_ylabel('V_col (V)')
    axs[1,1].set_title('Pulsed Input')
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)
    
    # 6. Summary
    axs[1,2].axis('off')
    summary = f"""
══════════════════════════════════════
     VALIDATION SUMMARY (CORRECTED)
══════════════════════════════════════

CORRECTED EQUATIONS:
  I_syn = I_res + I_cap
        = (Vt-Vm)/Rs + Cp·dVt/dt
  
  C_gb · dVcol/dt = Σ I_syn

TWO NMOS TYPES:
   Output: Vth = {P.bot_Vth}V
   Reset:  Vth = {P.res_Vth}V

FITTED Output NMOS:
  Is  = {params_bot[0]:.2e} A
  Vth = {params_bot[1]:.3f} V
  κ   = {params_bot[2]:.3f}
  λ   = {params_bot[3]:.3f}

PULSE WIDTH (using Vth={Vth_target}V):
──────────────────────────────────
  Continuous: {t_to_vth*1e6:.1f} µs
  Pulsed: {pulses} pulses ({pulses*T_period*1e6:.0f} µs)
  V_steady: {V_steady:.3f} V
──────────────────────────────────

For MNIST: ≥{pulses} pulses/pixel
══════════════════════════════════════
"""
    axs[1,2].text(0.05, 0.5, summary, transform=axs[1,2].transAxes, fontsize=9,
                  verticalalignment='center', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figure2_validation_corrected.png', dpi=200)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    
    # ----- Extract BOTH NMOS types -----
    params_bot, *data_bot_tuple = extract_and_fit_nmos(
        'Output_NMOS', P.bot_W, P.bot_L, P.bot_Level, P.bot_Vth, P.bot_Tox,
        P.Vd_arr_bot, P.Vg_arr_bot, P.C_gb, P.C_gb_math
    )
    data_bot = data_bot_tuple
    
    params_res, *data_res_tuple = extract_and_fit_nmos(
        'Reset_NMOS', P.res_W, P.res_L, P.res_Level, P.res_Vth, P.res_Tox,
        P.Vd_arr_res, P.Vg_arr_res, 0, 0
    )
    data_res = data_res_tuple
    
    # ----- Run validations -----
    memcap_data = validate_memcap()
    column_data = validate_single_column(params_bot)
    multi_col_data = validate_multi_column(params_bot)
    pulse_data = analyze_pulse_width(params_bot)
    
    # ----- Generate plots -----
    plot_nmos_comparison(params_bot, data_bot, params_res, data_res)
    plot_validation(params_bot, memcap_data, column_data, multi_col_data, pulse_data)
 
