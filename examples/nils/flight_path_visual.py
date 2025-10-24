r"""
This script reads in results of the BADA User Interface -> Calculation tools
-> Start simple session with nominal A/C Gross Mass, default climb options (Climb at
given CAS/Mach), Hmo final pressure altitude and a step size of 500 ft,
for the Airbus A320neo (ICAO: A20N),

and of the BADA User Interface -> Calculation tools
-> Start simple session with nominal A/C Gross Mass, default climb options (Climb at
given CAS/Mach), Hmo final pressure altitude and a step size of 500 ft,
for the ATR 72-600 (ICAO: AT76),

as found under C:\Users\nmb48\Documents\GitHub\pybada\examples\nils\nilsbarner_V_316_All.
"""

__all__ = []

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pyBADA import atmosphere as atm
from pyBADA import conversions as conv
from pyBADA.bada3 import Bada3Aircraft
from pyBADA.bada3 import Parser as Bada3Parser

from matplotlib_custom_settings import *
from examples.nils.ptf_results_reader import read_ptf_results
from examples.nils.flight_envelope import get_flight_envelope
from takeoff_calc_tasopt import (
    solve_takeoff, analyse_takeoff, plot_takeoff,
)

# %% Read

# Example usage
directory = os.path.join(
    # "C:\\",
    # "Users",
    # "nmb48",
    # "Documents",
    # "GitHub",
    # "pybada",
    os.getcwd(),
    "examples",
    "nils",
    "nilsbarner_V_316_All",
)
df_e290_climb = read_ptf_results(
    os.path.join(directory, "V_316_E290_CLIMB_20251013_123219.PTF")
)
df_e290_cruise = read_ptf_results(
    os.path.join(directory, "V_316_E290_CRUISE_20251014_125509.PTF")
)
df_e290_descent = read_ptf_results(
    os.path.join(directory, "V_316_E290_DESCENT_20251014_125311.PTF")
)
dfs_e290 = [df_e290_climb, df_e290_cruise, df_e290_descent]

h_traj = []
M_traj = []
Tnet_traj = []
V_traj = []
for df_e290 in dfs_e290:

    h_traj.append(df_e290['Hp'].to_numpy() * 0.3048)
    M_traj.append(df_e290['Mach'].to_numpy())
    
    if 'Thrust' in df_e290.columns.to_list():
        Tnet_traj.append(df_e290['Thrust'].to_numpy())
    else:
        Tnet_traj.append(df_e290['Drag'].to_numpy())
        
    V_traj.append(df_e290['TAS'].to_numpy() * 0.514444)

h_traj = np.concatenate(h_traj)
M_traj = np.concatenate(M_traj)
Tnet_traj = np.concatenate(Tnet_traj)
V_traj = np.concatenate(V_traj)

#%%

gamma = 1.4
R = 287
# EPR_core = EDF.FPR
# EPR_bp = EDF.FPR
# Fmax_static = (
#     EDFBYPASSMODEL_des.A_9 * p_0 * 2 * gamma / (gamma - 1) * (EPR_core**((gamma - 1) / gamma) - 1) +
#     EDFBYPASSMODEL_des.A_19 * p_0 * 2 * gamma / (gamma - 1) * (EPR_bp**((gamma - 1) / gamma) - 1)
# ) * 2
Fref_rotate = Tnet_traj[0] / 2
rho_sl = 1.225
V_stall = 105 * 0.514444  # BADA
Vref = V_traj[0]

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
    Fmax_static=Fref_rotate * 1.3,  # NILS: ASSUMPTION TO GET PLOTTING TO WORK!
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
    
(x, l1, lTO, lBF,
Vg, mask_g, Fg,
Vr, Fr,
Vb, Fb) = analyse_takeoff(
    lTO, l1, lBF, kA, kB, kC, VAlimsq, VBlimsq, VClimsq,
    F0_A, KV_A, F0_B, KV_B, FTO
)
    
Tnet_takeoff = Fg
Tnet_takeoff = Tnet_takeoff[~np.isnan(Tnet_takeoff)]
V_takeoff = Vg
V_takeoff = V_takeoff[~np.isnan(V_takeoff)]
M_takeoff = V_takeoff / (gamma * R * 288.15)
h_takeoff = np.zeros_like(Tnet_takeoff)

h_traj = np.hstack((h_takeoff, h_traj))
M_traj = np.hstack((M_takeoff, M_traj))
Tnet_traj = np.hstack((Tnet_takeoff, Tnet_traj))
V_traj = np.hstack((V_takeoff, V_traj))

# %% Process

badaVersion = "bada_316"
allData = Bada3Parser.parseAll(badaVersion=badaVersion)
AC_e290 = Bada3Aircraft(
    badaVersion=badaVersion, acName="E290", allData=allData
)

### The following airline procedure default speeds are given in CAS [kt] (page 29 of EIH-Technical-Report-220512-45)

# BADA speed schedule
[Vcl1_e290, Vcl2_e290, Mcl_e290] = AC_e290.flightEnvelope.getSpeedSchedule(
    phase="Climb"
)  # BADA Climb speed schedule
[Vcr1_e290, Vcr2_e290, Mcr_e290] = AC_e290.flightEnvelope.getSpeedSchedule(
    phase="Cruise"
)  # BADA Cruise speed schedule
[Vdes1_e290, Vdes2_e290, Mdes_e290] = AC_e290.flightEnvelope.getSpeedSchedule(
    phase="Descent"
)  # BADA Descent speed schedule

# calculate the crosover altitude for climb phase
crossoverAltitude_climb_e290 = conv.m2ft(atm.crossOver(Vcl2_e290, Mcl_e290))

# calculate the crosover altitude for descent phase
crossoverAltitude_descent_e290 = conv.m2ft(
    atm.crossOver(Vdes2_e290, Mdes_e290)
)

# NILS: the BADA speed schedule velocities are given in CAS
# and would have to be transformed to TAS to plot on figure below,
# but not clear what reference altitude used (see page 29 in
# EIH-Technical-Report-220512-45) so not used for now.

# CAS-to-TAS conversion code from C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py
# H_m = conv.ft2m(1e4)  # altitude [m]
# [theta, delta, sigma] = atm.atmosphereProperties(
#     h=H_m, deltaTemp=0
# )
# tas = atm.cas2Tas(cas=Vcl1_at76, delta=delta, sigma=sigma)

# %% Plot

from matplotlib.patches import Rectangle

from takeoff_calc_tasopt import (
    solve_takeoff, analyse_takeoff, plot_takeoff,
)

jet_climb_schedule_altitudes = [
    1499,
    2999,
    3999,
    4999,
    5999,
    9999,
    crossoverAltitude_climb_e290,
]
descent_schedule_altitudes = [
    999,
    1499,
    1999,
    2999,
    5999,
    9999,
    crossoverAltitude_descent_e290,
]

fig = plt.figure(figsize=(12,8), constrained_layout=True)
gs = gridspec.GridSpec(
    2, 2, width_ratios=[1,1], hspace=0.3, height_ratios=[1,1], wspace=0.15,
)
ax_1 = fig.add_subplot(gs[0, 0])
ax_2 = fig.add_subplot(gs[0, 1])
ax_3 = fig.add_subplot(gs[1, 0])
ax_4 = fig.add_subplot(gs[1, 1])

ax_1.plot(
    V_takeoff / 0.514444,
    h_takeoff / 0.3048,
    color="blue",
    linestyle="solid",
    label="A20N climb",
    # marker=".",
    clip_on=False,
    markersize=5,
)

ax_1.plot(
    df_e290_climb["TAS"],
    df_e290_climb["Hp"],
    color="blue",
    linestyle="solid",
    label="A20N climb",
    marker=".",
    markersize=5,
)

ax_1.plot(
    [df_e290_climb["TAS"].to_numpy()[-1], df_e290_cruise["TAS"].to_numpy()[0]],
    [df_e290_climb["Hp"].to_numpy()[-1], df_e290_cruise["Hp"].to_numpy()[0]],
    color="blue",
    linestyle="solid",
    label="A20N climb",
    marker=".",
    markersize=5,
)

ax_1.plot(
    df_e290_cruise["TAS"],
    df_e290_cruise["Hp"],
    color="blue",
    linestyle="solid",
    label="A20N cruise",
    marker=".",
    markersize=5,
)

ax_1.plot(
    [df_e290_cruise["TAS"].to_numpy()[-1], df_e290_descent["TAS"].to_numpy()[0]],
    [df_e290_cruise["Hp"].to_numpy()[-1], df_e290_descent["Hp"].to_numpy()[0]],
    color="blue",
    linestyle="solid",
    label="A20N climb",
    marker=".",
    markersize=5,
)

ax_1.plot(
    df_e290_descent["TAS"],
    df_e290_descent["Hp"],
    color="blue",
    linestyle="dashed",
    label="A20N descent",
    marker=".",
    markersize=5,
)

envelope_coordinates_e290 = get_flight_envelope(plot_bool=False, ax=None, ac_name='E290')[0]
ax_1.fill(
    envelope_coordinates_e290[:,0] / 0.514444, envelope_coordinates_e290[:,1] / 0.3048, facecolor='blue', edgecolor='none', alpha=0.2, label='Flight envelope GasTurb'
)

rect_1 = Rectangle((0, -1e3), 120, 2e3, facecolor='None', edgecolor='blue', linestyle='--', clip_on=False)
ax_1.add_patch(rect_1)

# '''
rect_2 = Rectangle((1.25e5, -1e3), 0.2e5, 2e3, facecolor='None', edgecolor='red', linestyle='--', clip_on=False)
ax_2.add_patch(rect_2)
# '''

ax_2.fill_betweenx(
    x1=np.zeros_like(np.unique(envelope_coordinates_e290[:,2])), x2=np.unique(envelope_coordinates_e290[:,2]), y=np.flip(np.unique(envelope_coordinates_e290[:,1])) / 0.3048, facecolor='red', edgecolor='none', alpha=0.2, label='Flight envelope GasTurb'
)

#####

ax_2.plot(
    Tnet_takeoff,
    h_takeoff / 0.3048,
    color="red",
    linestyle="solid",
    label="A20N climb",
    # marker=".",
    clip_on=False,
    markersize=5,
)

ax_2.plot(
    df_e290_climb["Thrust"],
    df_e290_climb["Hp"],
    color="red",
    linestyle="solid",
    label="A20N climb",
    marker=".",
    markersize=5,
)

ax_2.plot(
    [df_e290_climb["Thrust"].to_numpy()[-1], df_e290_cruise["Drag"].to_numpy()[0]],
    [df_e290_climb["Hp"].to_numpy()[-1], df_e290_cruise["Hp"].to_numpy()[0]],
    color="red",
    linestyle="solid",
    label="A20N climb",
    marker=".",
    markersize=5,
)

ax_2.plot(
    df_e290_cruise["Drag"],
    df_e290_cruise["Hp"],
    color="red",
    linestyle="solid",
    label="A20N cruise",
    marker=".",
    markersize=5,
)

ax_2.plot(
    [df_e290_cruise["Drag"].to_numpy()[-1], df_e290_descent["Thrust"].to_numpy()[0]],
    [df_e290_cruise["Hp"].to_numpy()[-1], df_e290_descent["Hp"].to_numpy()[0]],
    color="red",
    linestyle="solid",
    label="A20N climb",
    marker=".",
    markersize=5,
)

ax_2.plot(
    df_e290_descent["Thrust"],
    df_e290_descent["Hp"],
    color="red",
    linestyle="dashed",
    label="A20N descent",
    marker=".",
    markersize=5,
)

#####

# Plot lines of constant CAS

TAS_range = np.linspace(0, 500, 10)
H_ft_range = np.linspace(0, np.max(df_e290_cruise["Hp"].to_numpy()), 10)
TAS_grid, H_ft_grid = np.meshgrid(TAS_range, H_ft_range)
CAS_grid = np.zeros_like(TAS_grid)
M_grid = np.zeros_like(TAS_grid)

# jet_climb_schedule_tass = []
# for H_m in jet_climb_schedule_altitudes:
#     idx = np.argmin(abs(df_e290_climb["Hp"].to_numpy() * 0.3048 - H_m))
#     TAS = df_e290_climb["TAS"].to_numpy()[idx]
#     jet_climb_schedule_tass.append(TAS)
    
# # sys.exit()

# jet_climb_schedule_cass = []
# jet_climb_schedule_ms = []
# for (H_m, TAS) in zip(jet_climb_schedule_altitudes, jet_climb_schedule_tass):
#     H_m = H_ft * 0.3048
#     [theta, delta, sigma] = atm.atmosphereProperties(
#         h=H_m,
#         deltaTemp=0,
#     )
#     CAS = atm.tas2Cas(tas=TAS, delta=delta, sigma=sigma)
#     M = atm.tas2Mach(v=TAS, theta=theta)
#     jet_climb_schedule_cass.append(CAS)
#     jet_climb_schedule_ms.append(M)

# for i, j in np.ndindex(TAS_grid.shape):
#     TAS = TAS_grid[i, j]
#     H_ft = H_ft_grid[i, j]
#     H_m = H_ft * 0.3048
#     [theta, delta, sigma] = atm.atmosphereProperties(
#         h=H_m,
#         deltaTemp=0,
#     )
#     CAS = atm.tas2Cas(tas=TAS, delta=delta, sigma=sigma)
#     M = atm.tas2Mach(v=TAS, theta=theta)
#     CAS_grid[i, j] = CAS
#     M_grid[i, j] = M

# =============================================================================
# 1) get TAS samples at the requested altitudes (altitudes are in ft)
jet_climb_schedule_tass = []
for H_ft in jet_climb_schedule_altitudes:            # H_ft is in feet
    # df_e290_climb['Hp'] is in ft, so compare feet->feet (no 0.3048 here)
    idx = np.argmin(np.abs(df_e290_climb["Hp"].to_numpy() - H_ft))
    TAS_kn = df_e290_climb["TAS"].to_numpy()[idx]   # TAS in knots
    jet_climb_schedule_tass.append(TAS_kn)

# 2) convert those TAS (knots->m/s) and altitudes (ft->m) when calling atm functions
jet_climb_schedule_cass = []
jet_climb_schedule_ms = []
for (H_ft, TAS_kn) in zip(jet_climb_schedule_altitudes, jet_climb_schedule_tass):
    H_m = H_ft * 0.3048             # feet -> metres
    TAS_ms = TAS_kn * 0.514444      # knots -> m/s
    theta, delta, sigma = atm.atmosphereProperties(h=H_m, deltaTemp=0)
    CAS = atm.tas2Cas(tas=TAS_ms, delta=delta, sigma=sigma)
    M = atm.tas2Mach(v=TAS_ms, theta=theta)
    jet_climb_schedule_cass.append(CAS)
    jet_climb_schedule_ms.append(M)

# 3) when filling the grid, convert TAS grid values to m/s before passing to atm.tas2Cas
for i, j in np.ndindex(TAS_grid.shape):
    TAS_kn = TAS_grid[i, j]             # grid is in knots (for plotting)
    H_ft = H_ft_grid[i, j]              # grid is in ft
    H_m = H_ft * 0.3048                 # ft -> m
    TAS_ms = TAS_kn * 0.514444          # knots -> m/s
    theta, delta, sigma = atm.atmosphereProperties(h=H_m, deltaTemp=0)
    CAS = atm.tas2Cas(tas=TAS_ms, delta=delta, sigma=sigma)
    M = atm.tas2Mach(v=TAS_ms, theta=theta)
    CAS_grid[i, j] = CAS
    M_grid[i, j] = M

# =============================================================================

ctas = ax_1.contour(TAS_grid, H_ft_grid, CAS_grid, levels=jet_climb_schedule_cass, colors='grey', alpha=0.3, zorder=-10)
# ax_1.clabel(ctas)
cmach = ax_1.contour(TAS_grid, H_ft_grid, M_grid, levels=[jet_climb_schedule_ms[-1]], colors='blue', alpha=0.3, zorder=-20)
# ax_1.clabel(cmach)

ax_1.set_xlabel("True airspeed (kt)", labelpad=10)
ax_1.set_ylabel("Geopotential altitude (ft)", labelpad=10)
# ax_1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
ax_1.spines[['right', 'top']].set_visible(False)
ax_1.tick_params(axis='y', which='both', right=False, length=0)
ax_1.tick_params(axis='x', which='both', length=0)
ax_1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# ax_2.set_xlabel("True airspeed (kt)", labelpad=10)
ax_2.set_xlabel("Net thrust (N)", labelpad=10)
# ax_2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
ax_2.spines[['right', 'top']].set_visible(False)
ax_2.tick_params(axis='y', which='both', right=False, length=0)
ax_2.tick_params(axis='x', which='both', length=0)
# ax_2.ticklabel_format(useMathText=True, axis='y')
ax_2.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
ax_2.set_yticks([])

# fig.text(0.5, 0.05, "True airspeed (kt)", ha='center', va='center')
# fig.text(0.05, 0.5, "Geopotential altitude (ft)", ha='center', va='center', rotation='vertical')

add_margin(ax_1, m=0.05)
add_margin(ax_2, m=0.05)
ax_1.set_ylim(bottom=0)
ax_2.set_ylim(bottom=0)
# ax_1.set_xlim(-25,500)
# ax_2.set_xlim(-25,500)
ax_1.set_xlim(left=0)
ax_2.set_xlim(left=0)

ax1_ylim = ax_1.get_ylim()
ax_2.set_ylim(ax1_ylim)

ax_1.set_axisbelow(False)
ax_2.set_axisbelow(False)
for ax in (ax_1, ax_2, ax_3, ax_4):
    for sp in ax.spines.values():
        sp.set_zorder(1)   # keep low so data (zorder>=10) is on top
        
# # =============================================================================
# # '''

# x1, x2, y1, y2 = 1, 200, 2e4, 3e4  # subregion of the original image
# axins = ax_1.inset_axes(
#     [0, 0, 0.1, 0.1],
#     xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
# # axins.imshow(Z2, extent=extent, origin="lower")

# ax_1.indicate_inset_zoom(axins, edgecolor="black")

plot_takeoff(
    x, l1, lTO, lBF,
    Vg, mask_g, Fg,
    Vr, Fr,
    Vb, Fb,
    ax_3, ax_4,
)

ax_3.spines[['right', 'top']].set_visible(False)
ax_3.tick_params(axis='y', which='both', right=False, length=0)
ax_3.tick_params(axis='x', which='both', top=False)
ax_3.grid(False)
ax_3.title.set_visible(False)
# ax_3.get_legend().set_visible(False)
ax_3.set_xlim(left=0)
ax_3.set_ylim(bottom=0)

ax_4.spines[['right', 'top']].set_visible(False)
ax_4.tick_params(axis='y', which='both', right=False, length=0)
ax_4.tick_params(axis='x', which='both', top=False)
ax_4.grid(False)
ax_4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax_4.title.set_visible(False)
ax_4.set_xlim(left=0)
ax_4.set_ylim(bottom=0)

# # =============================================================================

plt.savefig('mission.svg', format='svg')
plt.show()
