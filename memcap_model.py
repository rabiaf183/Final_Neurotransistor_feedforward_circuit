"""
Memristor/Memcapacitor Models for ngspice
Contains: MEMCAP, MEM_NAMLAB, MEM_NAMLAB_DYN
Usage: from memcap_model import get_subcircuit
       subcircuit = get_subcircuit('MEMCAP')
"""

def get_subcircuit(model='MEMCAP'):
    """
    Return memristor subcircuit string for ngspice.
    
    Parameters:
        model: str - 'MEMCAP', 'MEM_NAMLAB', or 'MEM_NAMLAB_DYN'
    
    Returns:
        str - SPICE subcircuit definition
    """
    
    if model == 'MEMCAP':
        # MEMCAP with parallel capacitor Cp=10p
        subcircuit = """
.SUBCKT MEMCAP TE BE SV
.param Ar=4.7447e-8 As=1.1253e-8 Br=2.6831 Bs=9.3348
.param c1=2.9457e-4 c2=57414 c3=11103 Rp=8e6 Rs=278 Cp=10p
.param wc=1000 xon=0.1 xoff=0.284
.func Gm(x,Vm) {Ar*x*exp(Br*sgn(Vm)*sqrt(abs(Vm)/x))+As*x*exp(-Bs*sgn(Vm)*sqrt(abs(Vm)))}
.func win_off(x) {-exp((x-xoff)*wc)}
.func win_on(x) {-exp(-(x-xon)*wc)}
.func dxdt(Vm,x) {c1*(exp(c2*x*Vm*Gm(x,Vm)+win_off(x))-exp(-c3*x*Vm*Gm(x,Vm)+win_on(x)))}
Cx SV 0 1
Bx 0 SV i=dxdt(V(IE,BE),V(SV,0))
Bmem IE BE i=V(IE,BE)*Gm(V(SV,0),V(IE,BE))
Rms TE IE {Rs}
Rmp IE BE {Rp}
Cmp TE BE {Cp}
.ENDS MEMCAP
"""
    
    elif model == 'MEM_NAMLAB':
        # Original Namlab model (no parallel capacitor)
        subcircuit = """
.SUBCKT MEM_NAMLAB TE BE SV
.param Ar=4.7447e-8
.param As=1.1253e-8
.param Br=2.6831
.param Bs=9.3348
.param c1=2.9457e-4
.param c2=57414
.param c3=11103
.param wc=1000
.param xon=0.1
.param xoff=0.284
.func Gm(x,Vm) {Ar*x*exp(Br*sgn(Vm)*sqrt(abs(Vm)/x))+As*x*exp(-Bs*sgn(Vm)*sqrt(abs(Vm)))}
.func win_off(x) {-exp((x-xoff)*wc)}
.func win_on(x) {-exp(-(x-xon)*wc)}
.func dxdt(Vm,x) {c1*(exp(c2*x*Vm*Gm(x,Vm)+win_off(x))-exp(-c3*x*Vm*Gm(x,Vm)+win_on(x)))}
Cx SV 0 1 
Bx 0 SV i=dxdt(V(ME,BE),V(SV,0))
Bmem ME BE i=V(ME,BE)*Gm(V(SV,0),V(ME,BE))
Rs TE ME 278
Rp ME BE 8e6
.ENDS MEM_NAMLAB
"""
    
    elif model == 'MEM_NAMLAB_DYN':
        # Dynamic Namlab model with current-dependent switching
        subcircuit = """
.SUBCKT MEM_NAMLAB_DYN TE BE SV
.param PF=5.072849397365829e-09
.param C=6.151058751629694e+00
.param n=1000
.param Ar_LRS=8.2531e-01
.param Ar_HRS=4.4837e-01
.param As_LRS=4.3505e-01
.param As_HRS=2.8518e-01
.param Br_LRS=6.4610e-01
.param Br_HRS=1.1572e+00
.param Bs_LRS=5.5787e-01
.param Bs_HRS=4.9976e-01
.param Rs=2.5874e+02
.param Rp=2.7226e+07
.param Cr_0=2.6724e-04
.param Cs_2=5.3305e-03
.param Cr_LRS=1.1715e+01
.param Cr_HRS=1.7021e+01
.param Cs_LRS=2.0450e+01
.param Cs_HRS=4.5518e+01
.param Cr_exp=5.1744e-01
.param Cs_exp=5.2448e-01
.param xoff=0.99
.param xon=0.01
.param x0=0.01
.param Cs_par=11.127e-12
.func plin(x,lower,upper) {lower + (upper-lower)*x}
.func Gm_model(x,Vm) {PF*plin(x,Ar_LRS,Ar_HRS)*exp(C*sgn(Vm)*sqrt(abs(Vm/plin(x,Br_LRS,Br_HRS)))) + PF*plin(x,As_LRS,As_HRS)*exp(-C*sgn(Vm)*sqrt(abs(Vm/plin(x,Bs_LRS,Bs_HRS))))}
.func sigmn(x) {1/(1+n*exp(-n*x))}
.func dxdt_model(x,Im,Vm) {Cr_0*pow(x,0)*exp(plin(x,Cr_LRS,Cr_HRS)*sgn(Im)*pow(abs(Im*1e3),Cr_exp))*sigmn(xoff-x) - Cs_2*pow(x,2)*exp(-plin(x,Cs_LRS,Cs_HRS)*sgn(Im)*pow(abs(Im*1e3),Cs_exp))*sigmn(x-xon)}
Cx SV 0 1
Bx 0 SV i=dxdt_model(V(SV,0),I(Bmem),V(ME,BE))
Rx SV 0 1G
Bmem ME BE i=V(ME,BE)*Gm_model(V(SV,0),V(ME,BE))
Rs TE ME {Rs}
Rp ME BE {Rp}
Cs TE ME {Cs_par}
.ENDS MEM_NAMLAB_DYN
"""
    
    else:
        raise ValueError(f"Unknown model: {model}. Use 'MEMCAP', 'MEM_NAMLAB', or 'MEM_NAMLAB_DYN'")
    
    return subcircuit


def list_models():
    """Print available models."""
    print("Available models:")
    print("  MEMCAP        - With parallel capacitor Cp=10p (LTspice compatible)")
    print("  MEM_NAMLAB    - Original Namlab model (no Cp)")
    print("  MEM_NAMLAB_DYN - Dynamic model with current-dependent switching")


# For backward compatibility
def get_memcap_subcircuit():
    """Return MEMCAP subcircuit (backward compatible)."""
    return get_subcircuit('MEMCAP')
