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

from pyBADA import atmosphere as atm
from pyBADA import conversions as conv
from pyBADA.bada3 import Bada3Aircraft
from pyBADA.bada3 import Parser as Bada3Parser


from examples.nils.ptf_results_reader import read_ptf_results

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
df_a20n_climb = read_ptf_results(
    os.path.join(directory, "V_316_A20N_CLIMB_20251004_133315.PTF")
)
df_at76_climb = read_ptf_results(
    os.path.join(directory, "V_316_AT76_CLIMB_20251004_133316.PTF")
)
df_a20n_descent = read_ptf_results(
    os.path.join(directory, "V_316_A20N_DESCENT_20251004_133315.PTF")
)
df_at76_descent = read_ptf_results(
    os.path.join(directory, "V_316_AT76_DESCENT_20251004_133316.PTF")
)

# %% Process

badaVersion = "bada_316"
allData = Bada3Parser.parseAll(badaVersion=badaVersion)
AC_at76 = Bada3Aircraft(
    badaVersion=badaVersion, acName="AT76", allData=allData
)
AC_a20n = Bada3Aircraft(
    badaVersion=badaVersion, acName="A20N", allData=allData
)

### The following airline procedure default speeds are given in CAS [kt] (page 29 of EIH-Technical-Report-220512-45)

# BADA speed schedule
[Vcl1_at76, Vcl2_at76, Mcl_at76] = AC_at76.flightEnvelope.getSpeedSchedule(
    phase="Climb"
)  # BADA Climb speed schedule
[Vcr1_at76, Vcr2_at76, Mcr_at76] = AC_at76.flightEnvelope.getSpeedSchedule(
    phase="Cruise"
)  # BADA Cruise speed schedule
[Vdes1_at76, Vdes2_at76, Mdes_at76] = AC_at76.flightEnvelope.getSpeedSchedule(
    phase="Descent"
)  # BADA Descent speed schedule
# BADA speed schedule
[Vcl1_a20n, Vcl2_a20n, Mcl_a20n] = AC_a20n.flightEnvelope.getSpeedSchedule(
    phase="Climb"
)  # BADA Climb speed schedule
[Vcr1_a20n, Vcr2_a20n, Mcr_a20n] = AC_a20n.flightEnvelope.getSpeedSchedule(
    phase="Cruise"
)  # BADA Cruise speed schedule
[Vdes1_a20n, Vdes2_a20n, Mdes_a20n] = AC_a20n.flightEnvelope.getSpeedSchedule(
    phase="Descent"
)  # BADA Descent speed schedule

# calculate the crosover altitude for climb phase
crossoverAltitude_climb_at76 = conv.m2ft(atm.crossOver(Vcl2_at76, Mcl_at76))
crossoverAltitude_climb_a20n = conv.m2ft(atm.crossOver(Vcl2_a20n, Mcl_a20n))

# calculate the crosover altitude for descent phase
crossoverAltitude_descent_at76 = conv.m2ft(
    atm.crossOver(Vdes2_at76, Mdes_at76)
)
crossoverAltitude_descent_a20n = conv.m2ft(
    atm.crossOver(Vdes2_a20n, Mdes_a20n)
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

jet_climb_schedule_altitudes = [
    1499,
    2999,
    3999,
    4999,
    5999,
    9999,
    crossoverAltitude_climb_a20n,
]
tp_climb_schedule_altitudes = [
    499,
    999,
    1499,
    9999,
    crossoverAltitude_climb_at76,
]
descent_schedule_altitudes = [
    999,
    1499,
    1999,
    2999,
    5999,
    9999,
    crossoverAltitude_descent_a20n,
    crossoverAltitude_descent_at76,
]

fig, ax = plt.subplots()

# ax.plot(df_a20n_climb['CAS'][:10], df_a20n_climb['Hp'][:10], color='blue', linestyle='solid', label='A20N climb', marker='.')
ax.plot(
    df_a20n_climb["TAS"],
    df_a20n_climb["Hp"],
    color="blue",
    linestyle="solid",
    label="A20N climb",
    marker=".",
)
ax.plot(
    df_at76_climb["TAS"],
    df_at76_climb["Hp"],
    color="red",
    linestyle="solid",
    label="AT76 climb",
    marker=".",
)

# ax.plot(df_a20n_descent['CAS'], df_a20n_descent['Hp'], color='blue', linestyle='dashed', label='A20N descent', marker='.')
ax.plot(
    df_a20n_descent["TAS"],
    df_a20n_descent["Hp"],
    color="blue",
    linestyle="dashed",
    label="A20N descent",
    marker=".",
)
ax.plot(
    df_at76_descent["TAS"],
    df_at76_descent["Hp"],
    color="red",
    linestyle="dashed",
    label="AT76 descent",
    marker=".",
)

ax.set_xscale("log")
ax.set_yscale("log")

for h in jet_climb_schedule_altitudes:
    ax.axhline(
        h,
        color="blue",
        linestyle="dotted",
        lw=1,
        alpha=0.5,
        label=f"A20N climb {h}",
    )

for h in tp_climb_schedule_altitudes:
    ax.axhline(
        h,
        color="red",
        linestyle="dotted",
        lw=1,
        alpha=0.5,
        label=f"AT76 climb {h}",
    )

for h in descent_schedule_altitudes:
    ax.axhline(
        h,
        color="black",
        linestyle="dotted",
        lw=1,
        alpha=0.5,
        label=f"Descent {h}",
    )

ax.axvline(
    Vcl1_a20n,
    color="blue",
    linestyle="dashdot",
    lw=1,
    alpha=0.5,
    label="Vcl1_a20n",
)
ax.axvline(
    Vcl2_a20n,
    color="blue",
    linestyle="dashdot",
    lw=1,
    alpha=0.5,
    label="Vcl2_a20n",
)
ax.axvline(
    Vdes1_a20n,
    color="blue",
    linestyle="dashdot",
    lw=1,
    alpha=0.5,
    label="Vdes1_a20n",
)
ax.axvline(
    Vdes2_a20n,
    color="blue",
    linestyle="dashdot",
    lw=1,
    alpha=0.5,
    label="Vdes2_a20n",
)

ax.axvline(
    Vcl1_at76,
    color="red",
    linestyle="dashdot",
    lw=1,
    alpha=0.5,
    label="Vcl1_at76",
)
ax.axvline(
    Vcl2_at76,
    color="red",
    linestyle="dashdot",
    lw=1,
    alpha=0.5,
    label="Vcl2_at76",
)
ax.axvline(
    Vdes1_at76,
    color="red",
    linestyle="dashdot",
    lw=1,
    alpha=0.5,
    label="Vdes1_at76",
)
ax.axvline(
    Vdes2_at76,
    color="red",
    linestyle="dashdot",
    lw=1,
    alpha=0.5,
    label="Vdes2_at76",
)

# Plot lines of constant CAS
for CAS in np.linspace(149.3, 310, 5):
    H_ft_range = np.linspace(0, 4e4, 100)
    iso_CAS_line = []
    for H_ft in H_ft_range:
        H_m = H_ft * 0.3048
        [theta, delta, sigma] = atm.atmosphereProperties(
            h=H_m,
            deltaTemp=0,
        )
        tas = atm.cas2Tas(cas=CAS, delta=delta, sigma=sigma)
        iso_CAS_line.append(tas)

    ax.plot(iso_CAS_line, H_ft_range)

ax.set_xlabel("TAS [kt]")
ax.set_ylabel("Hp [ft]")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

plt.show()
