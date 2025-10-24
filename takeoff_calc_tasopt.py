"""
This script performs the balanced field length calculation
outlined in \PhD\Software\TASOPT\takeoff.pdf and coded up
in src/mission/takeoff.jl. It is used to determine the
thrust and velocity profile throughout take-off, which
is then prepended to the pyBADA mission analysis.
"""

__all__ = [
    "solve_takeoff",
    "analyse_takeoff",
    "plot_takeoff",
]

import math
import numpy as np
import matplotlib.pyplot as plt


def solve_takeoff(
    # required inputs (scalars)
    W_MTO, W_fuel, W_pay_gross, parm_Wpay, parm_WTO, S,
    dfan, HTRf, neng, muroll, mubrake, hobst,
    CD_wing_roll, CD_climb, CDgear, cdefan, CDspoil,
    Fmax_static, Fref_rotate,
    rho_rotate, rho_static, rho_climb1,
    Vref, V2,
    # optional
    CDivert=0.002,
    mdot_core_static=0.0, eff_static=1.0,
    mdot_core_rotate=0.0, eff_rotate=1.0,
    gee=9.80665, toler=1e-7, printTO=True,
):
    """Compute takeoff numbers and plot thrust/velocity profiles.

    Returns a dict containing key outputs (V1, V2, lTO, l1, lBF, tTO, FTO, gamVTO, gamVBF, singTO, singBF, WfTO).
    If make_plot=True a figure with two panels is shown (thrust vs V, and V & thrust vs distance).
    """
    # ----- housekeeping / derived props
    Wzero = W_MTO - W_fuel - W_pay_gross + parm_Wpay
    WfTO = parm_WTO - Wzero   # mission TO fuel (initial estimate)
    Afan = 0.25 * math.pi * dfan**2 * (1.0 - HTRf**2)
    CDeng = cdefan * (0.25 * math.pi * dfan**2) / S

    # per-engine thrust fit constants (use Vstall as reference speed as in Julia)
    F01 = 0.5 * (Fmax_static + Fref_rotate)
    KV1 = (Fmax_static - Fref_rotate) / Vref**2

    # total full-power constants (segment A: all engines)
    F0_A = F01 * neng
    KV_A = KV1 * neng
    FTO = Fmax_static * neng

    # drag during the roll
    CDroll = CD_wing_roll + CDgear

    # kA and VAlim^2 (segment A)
    rho0 = rho_rotate
    denomA = KV_A + rho0 * S * CDroll
    kA = denomA / (parm_WTO / gee)
    VAlimsq = 2.0 * (F0_A - parm_WTO * muroll) / denomA

    if VAlimsq <= 0.0:
        # normal takeoff impossible (no positive limiting speed)
        raise Exception("Normal takeoff impossible: VAlimsq <= 0.")

    VAlim = math.sqrt(VAlimsq)
    # normal takeoff distance & time
    if VAlim <= V2:
        raise Exception(f"Normal takeoff impossible. VAlim < V2: VAlim={VAlim:.3f}, V2={V2:.3f}")
    else:
        Vrat = V2 / VAlim
        one_minus = 1.0 - Vrat**2
        lTO = -math.log(max(1e-16, one_minus)) / kA
        tTO = math.log((1.0 + Vrat) / (1.0 - Vrat)) / (kA * VAlim)

    # climb (all engines) — climb acceleration sin(gamma)
    F2_A = F0_A - KV_A * 0.5 * V2**2
    singTO = (F2_A - 0.5 * V2**2 * rho0 * S * CD_climb) / parm_WTO
    singTO = max(0.01, min(0.99, singTO))

    # ---------------- Balanced-field (engine-out and braking)
    # effective drag terms
    CDb = CDroll + CDeng + CDivert                # engine-out (B)
    CDc = CDroll + CDeng * neng + CDspoil        # braking (C)

    # thrust constants for engine-out (neng-1 engines)
    F0_B = F01 * (neng - 1.0)
    KV_B = KV1 * (neng - 1.0)

    # kB and kC (use W=parm_WTO)
    kB = (KV_B + rho0 * S * CDb) / (parm_WTO / gee)
    kC = (rho0 * S * CDc) / (parm_WTO / gee)

    denomB = KV_B + rho0 * S * CDb
    VBlimsq = 2.0 * (F0_B - parm_WTO * muroll) / denomB
    denomC = rho0 * S * CDc
    VClimsq = 2.0 * (-parm_WTO * mubrake) / denomC  # "Note that V_c_lim^2 is negative."

    VBlim = math.sqrt(VBlimsq)
    
    if VBlim <= V2:
        # engine-out impossible → follow Julia behavior: set defaults and compute fuel burn
        raise Exception("Engine-out takeoff impossible. VBlim < V2 or invalid limits.")

    # initial guesses for Newton (Julia: 0.8*lTO and 1.3*lTO)
    l1 = 0.8 * lTO
    lBF = 1.3 * lTO
    V2sq = V2**2
    
    if printTO:
        print("\nTakeoff:\n#   lTO(m)     l1(m)      lBF(m)     dmax(m)")
        
    dl1 = dlBF = 0.0
    singBF = 0.0

    # Newton iteration for l1 and lBF (same algebra as Julia)
    for it in range(1, 16):
        exA = math.exp(-kA * l1)
        exB = math.exp(kB * (lBF - l1))
        exC = math.exp(kC * (lBF - l1))

        exA_l1 = -kA * exA
        exB_l1 = -kB * exB
        exC_l1 = -kC * exC
        exA_lBF = 0.0
        exB_lBF = kB * exB
        exC_lBF = kC * exC

        r1 = VAlimsq * (1.0 - exA) + (VBlimsq - V2sq) * exB - VBlimsq
        a11 = VAlimsq * (-exA_l1) + (VBlimsq - V2sq) * exB_l1
        a12 = VAlimsq * (-exA_lBF) + (VBlimsq - V2sq) * exB_lBF

        r2 = VAlimsq * (1.0 - exA) - VClimsq * (1.0 - exC)
        a21 = VAlimsq * (-exA_l1) - VClimsq * (-exC_l1)
        a22 = VAlimsq * (-exA_lBF) - VClimsq * (-exC_lBF)

        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-18:
            raise Exception("Singular Jacobian during BFL Newton iteration; aborting.")

        dl1 = -(r1 * a22 - a12 * r2) / det
        dlBF = -(a11 * r2 - r1 * a21) / det
        dmax = max(abs(dl1), abs(dlBF))

        if printTO:
            print(f"{it:2d} {lTO:10.3f} {l1:10.3f} {lBF:10.3f} {dmax:10.6f}")

        l1 += dl1
        lBF += dlBF

        if dmax < toler * lTO:
            # converged: compute V1 and engine-out climb sin(gamma)
            V1 = VAlim * math.sqrt(max(0.0, 1.0 - math.exp(-kA * l1)))
            F2_B = F0_B - KV_B * 0.5 * V2**2
            CD_total_climb_down = CD_climb + CDgear + CDeng
            singBF = (F2_B - 0.5 * V2**2 * rho0 * S * CD_total_climb_down) / parm_WTO
            singBF = max(0.01, min(0.99, singBF))

            mdotf1 = mdot_core_static * eff_static * neng
            mdotf2 = mdot_core_rotate * eff_rotate * neng
            WfTO = 0.5 * (mdotf1 + mdotf2) * tTO * gee
            
            outputs = {
                'V1': V1, 'V2': V2, 'lTO': lTO, 'l1': l1, 'lBF': lBF,
                'tTO': tTO, 'FTO': FTO, 'gamVTO': math.asin(singTO),
                'gamVBF': math.asin(singBF), 'singTO': singTO, 'singBF': singBF,
                'WfTO': WfTO
            }
            return (
                outputs, lTO, l1, lBF, kA, kB, kC, VAlimsq, VBlimsq, VClimsq,
                F0_A, KV_A, F0_B, KV_B, FTO,
            )

    raise Exception('Newton did not converge.')
    
    
def analyse_takeoff(
    lTO, l1, lBF,
    kA, kB, kC,
    VAlimsq, VBlimsq, VClimsq,
    F0_A, KV_A, F0_B, KV_B, FTO
):
    
    # x-grid from 0 to a bit past the largest endpoint (cover both lTO and lBF)
    maxL = max(lTO, lBF, 1e-6) * 1.05
    x = np.linspace(0.0, maxL, 500)

    # --- Helper functions for analytic V^2 solutions ---
    # Segment A (all engines): V^2_A(l) = VAlimsq * (1 - exp(-kA * l))
    def V2_A_of_l(l_arr):
        return VAlimsq * (1.0 - np.exp(-kA * np.maximum(l_arr, 0.0)))

    # V1^2 (value at l1) -- used as continuity condition
    V1sq = VAlimsq * (1.0 - math.exp(-kA * l1))

    # Segment B (engine-out, continuity at l1):
    # V^2_B(l) = VBlimsq - (VBlimsq - V1^2) * exp(-kB * (l - l1)), for l >= l1
    def V2_B_of_l(l_arr):
        l_rel = np.maximum(l_arr - l1, 0.0)
        return VBlimsq - (VBlimsq - V1sq) * np.exp(-kB * l_rel)

    # Segment C (braking, continuity at l1):
    # V^2_C(l) = VClimsq + (V1^2 - VClimsq) * exp(-kC * (l - l1)), for l >= l1
    def V2_C_of_l(l_arr):
        l_rel = np.maximum(l_arr - l1, 0.0)
        return VClimsq + (V1sq - VClimsq) * np.exp(-kC * l_rel)

    # --- Build per-scenario arrays (only defined up to their domain endpoints) ---
    Vg = np.full_like(x, np.nan)   # normal (green) up to lTO
    Vr = np.full_like(x, np.nan)   # continued (red) up to lBF
    Vb = np.full_like(x, np.nan)   # aborted  (blue) up to lBF

    # normal: 0 <= x <= lTO
    mask_g = x <= lTO
    V2_g = V2_A_of_l(x[mask_g])
    Vg[mask_g] = np.sqrt(V2_g)

    # continued (engine-out): 0 <= x <= lBF
    mask_before_l1 = x <= l1
    mask_after_l1 = (x > l1) & (x <= lBF)

    # up to l1 same as A
    V2_before = V2_A_of_l(x[mask_before_l1])
    Vr[mask_before_l1] = np.sqrt(V2_before)

    # after l1 use B-formula
    if np.any(mask_after_l1):
        V2_afterB = V2_B_of_l(x[mask_after_l1])
        Vr[mask_after_l1] = np.sqrt(V2_afterB)

    # aborted (braking): 0 <= x <= lBF
    # up to l1 same as A
    Vb[mask_before_l1] = np.sqrt(V2_before)
    # after l1 use C-formula
    if np.any(mask_after_l1):
        V2_afterC = V2_C_of_l(x[mask_after_l1])
        Vb[mask_after_l1] = np.sqrt(V2_afterC)

    # --- Thrust histories (dashed curves) ---
    # green thrust (normal) defined where Vg is defined
    Fg = np.full_like(x, np.nan)
    mask_fg = ~np.isnan(Vg)
    Fg[mask_fg] = F0_A - 0.5 * KV_A * (Vg[mask_fg] ** 2)

    # red thrust (continued): before l1 use all-engines, after l1 use engine-out model
    Fr = np.full_like(x, np.nan)
    # before l1
    Fr[mask_before_l1] = F0_A - 0.5 * KV_A * (np.sqrt(V2_before) ** 2)
    # after l1
    if np.any(mask_after_l1):
        # use Vr values (engine-out)
        mask_r_after = mask_after_l1
        Fr[mask_r_after] = F0_B - 0.5 * KV_B * (Vr[mask_r_after] ** 2)

    # blue thrust (aborted): before l1 use all-engines, after l1 assume thrust reduced to zero (idle/brakes)
    Fb = np.full_like(x, np.nan)
    Fb[mask_before_l1] = F0_A - 0.5 * KV_A * (np.sqrt(V2_before) ** 2)
    if np.any(mask_after_l1):
        Fb[mask_after_l1] = 0.0
    
    return (
        x, l1, lTO, lBF,
        Vg, mask_g, Fg,
        Vr, Fr,
        Vb, Fb,
    )


def plot_takeoff(
    x, l1, lTO, lBF,
    Vg, mask_g, Fg,
    Vr, Fr,
    Vb, Fb,
    ax1=None, ax2=None,
):
    """
    Updated plotting function:
      - Left subplot: dashed thrust vs velocity for the three scenarios (green/red/blue).
      - Right subplot: solid velocity and dashed thrust vs distance for the three scenarios.
    Scenarios:
      green = normal takeoff (all engines) up to lTO
      red   = continued takeoff after engine-out (engine out at l1) up to lBF
      blue  = rejected/aborted takeoff (braking after l1) up to lBF

    Inputs are the same quantities used in the solver:
      lTO, l1, lBF : distances (m)
      kA, kB, kC   : distance-constants for the three segments
      VAlimsq, VBlimsq, VClimsq : limiting V^2 values for A, B, C
      F0_A, KV_A   : thrust model params for all-engines (F = F0 - 0.5*KV*V^2)
      F0_B, KV_B   : thrust model params for (neng-1)-engines after failure
      FTO          : total static takeoff thrust (for reference / annotation)
    """
    
    if ax1 == None and ax2 == None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # # Left: thrust vs velocity (dashed lines)
    # # Extract usable (non-nan) pairs for each scenario
    # def plot_F_vs_V(ax):
    #     for V_arr, F_arr, color, label in (
    #         (Vg[mask_g], Fg[mask_g], 'green', 'Normal (all engines)'),
    #         (Vr[~np.isnan(Vr)], Fr[~np.isnan(Vr)], 'red',   'Engine-out (continue)'),
    #         (Vb[~np.isnan(Vb)], Fb[~np.isnan(Vb)], 'blue',  'Rejected (brake)')
    #     ):
    #         if V_arr.size > 0 and F_arr.size > 0:
    #             ax.plot(V_arr, F_arr, linestyle='--', color=color, label=label)
    #     ax.set_xlabel('Velocity (m/s)')
    #     ax.set_ylabel('Thrust (N)')
    #     ax.set_title('Thrust vs Velocity (three scenarios)')
    #     ax.grid(True)
    #     ax.legend()

    # plot_F_vs_V(ax1)

    # Right: velocity (solid) and thrust (dashed) vs distance for each scenario
    ax1.plot(x, Vg, color='blue',  linestyle='-', label='Normal', clip_on=False)
    ax1.plot(x, Vr, color='blue',    linestyle='--', label='One-engine-out', clip_on=False)
    ax1.plot(x, Vb, color='blue',   linestyle='-.', label='Rejected', clip_on=False)
    # ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity and Thrust vs Distance')
    ax1.grid(True)
    ax1.set_xticks([l1, lTO, lBF])
    ax1.set_xticks([l1, lTO, lBF])
    ax1.set_xticklabels([r'$l_1$', r'$l_\mathrm{TO}$', r'$l_\mathrm{BFL}$'])

    # plot thrust on twin axis (dashed, matching color of scenario)
    # ax2b = ax2.twinx()
    ax2.plot(x, Fg, linestyle='-', color='red', label='Thrust - Normal', clip_on=False)
    ax2.plot(x, Fr, linestyle='--', color='red',   label='Thrust - Engine-out', clip_on=False)
    ax2.plot(x, Fb, linestyle='-.', color='red',  label='Thrust - Rejected', clip_on=False)
    ax2.set_ylabel('Net thrust (N)')
    ax2.set_xticks([l1, lTO, lBF])
    ax2.set_xticklabels([r'$l_1$', r'$l_\mathrm{TO}$', r'$l_\mathrm{BFL}$'])

    # # vertical markers for key distances
    # ax1.axvline(l1,  color='k', linestyle=':', linewidth=1.0, label='l1')
    # ax1.axvline(lTO, color='k', linestyle='--', linewidth=1.0, label='lTO')
    # ax1.axvline(lBF, color='k', linestyle='-.', linewidth=1.0, label='lBF')

    # Combined legend (velocity + thrust)
    lines_v, labels_v = ax1.get_legend_handles_labels()
    lines_f, labels_f = ax2.get_legend_handles_labels()
    # ensure unique labels (avoid repeating l1/lTO/lBF)
    combined_lines = lines_v + lines_f
    combined_labels = labels_v + labels_f
    ax1.legend(frameon=False, loc='lower center')

    # # Annotate endpoints
    # ax1.text(lTO, 0.95 * np.nanmax(Vg[~np.isnan(Vg)]) if np.any(~np.isnan(Vg)) else 0.0,
    #          ' lTO', color='black', verticalalignment='top')
    # ax1.text(lBF, 0.95 * np.nanmax(np.concatenate([Vr[~np.isnan(Vr)], Vb[~np.isnan(Vb)]])) if (np.any(~np.isnan(Vr)) or np.any(~np.isnan(Vb))) else 0.0,
    #          ' lBF', color='black', verticalalignment='top')

    if ax1 == None and ax2 == None:
        plt.tight_layout()
        plt.show()
    
    ###

    # gamma = 1.4
    # R = 287
    # T_f = 288.15
    # M_f = Vg / np.sqrt(gamma * R * T_f)
    # pi_fan = 1.3
    # M_j = np.sqrt(((1 + (gamma - 1) / 2 * M_f**2) * pi_fan**((gamma - 1) / gamma) - 1) * 2 / (gamma - 1))
    # Pg = Fg * np.sqrt(gamma * R * T_f) / 2 * (M_j + M_f)
    
    # fig, ax = plt.subplots()
    # ax.plot(x, Pg, color='green',  linestyle='-', label='V - Normal (to lTO)')
    # plt.show()

#%%

if __name__ == "__main__":
    
    # Fmax_static = (
    #     A_8 * p_amb * 2 * gamma / (gamma - 1) * ((p_t_8 / p_amb)**((gamma - 1) / gamma) - 1) +
    #     A_18 * p_amb * 2 * gamma / (gamma - 1) * ((p_t_18 / p_amb)**((gamma - 1) / gamma) - 1)
    # )
    Fmax_static = 150e3
    Fref_rotate = 90000.0
    rho_sl = 1.225
    V_stall = 80.0 / 1.2
    Vref = 60.0
    
    
    (outputs, lTO, l1, lBF, kA, kB, kC, VAlimsq, VBlimsq, VClimsq,
         F0_A, KV_A, F0_B, KV_B, FTO,
    ) = solve_takeoff(
        W_MTO=49e3*9.81,  # BADA
        W_fuel=49e3*9.81/4,
        W_pay_gross=49e3*9.81/4,
        parm_Wpay=49e3*9.81/4,
        parm_WTO=49e3*9.81,
        S=103,  # BADA
        dfan=2.0,  # https://en.wikipedia.org/wiki/Pratt_%26_Whitney_PW1000G
        HTRf=0.3,
        neng=2,
        muroll=0.02,
        mubrake=0.5,
        hobst=10.0,
        CD_wing_roll=0.03,
        CD_climb=0.032,
        CDgear=0.02,
        cdefan=0.02,
        CDspoil=0.1,
        Fmax_static=Fmax_static,
        Fref_rotate=Fref_rotate,  # first point of BADA mission profile
        rho_rotate=rho_sl,
        rho_static=rho_sl,
        rho_climb1=rho_sl,
        Vref=Vref,  # first point of BADA mission profile
        V2=1.2*V_stall,  # V_2 = 1.2 * V_stall; (10) in takeoff.pdf
        mdot_core_static=0.8,
        eff_static=1.0,
        mdot_core_rotate=1.0,
        eff_rotate=1.0,
        printTO=True
    )
    print("\nOutputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
        
    (x, l1, lTO, lBF,
    Vg, mask_g, Fg,
    Vr, Fr,
    Vb, Fb) = analyse_takeoff(
        lTO, l1, lBF, kA, kB, kC, VAlimsq, VBlimsq, VClimsq,
        F0_A, KV_A, F0_B, KV_B, FTO
    )
    plot_takeoff(
        x, l1, lTO, lBF,
        Vg, mask_g, Fg,
        Vr, Fr,
        Vb, Fb,
        ax1=None, ax2=None,
    )
    
