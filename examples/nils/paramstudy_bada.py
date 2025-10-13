r"""
This script is adapted from C:\Users\nmb48\Documents\GitHub\pybada\examples\trajectoryAC.py
and reproduces the results of the BADA User Interface -> Calculation tools
-> Start simple session with nominal A/C Gross Mass, default climb options (Climb at
given CAS/Mach), Hmo final pressure altitude and a step size of 500 ft,
for the Airbus A320neo (ICAO: A20N).
"""

__all__ = []

import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from ambiance import Atmosphere

from pyBADA import TCL as TCL
from pyBADA import atmosphere as atm
from pyBADA import conversions as conv
from pyBADA.bada3 import Bada3Aircraft
from pyBADA.bada3 import Parser as Bada3Parser
from pyBADA.bada4 import Bada4Aircraft
from pyBADA.bada4 import Parser as Bada4Parser
from pyBADA.flightTrajectory import FlightTrajectory as FT

engine_type = 'turbofan'

@dataclass
class target:
    ROCDtarget: float = None
    slopetarget: float = None
    acctarget: float = None
    ESFtarget: float = None


# initialization of BADA3/4
# uncomment for testing different BADA family if available

badaVersion = "bada_316"

allData = Bada3Parser.parseAll(badaVersion=badaVersion)

AC = Bada3Aircraft(
    badaVersion=badaVersion,
    # acName="A20N",
    acName="A321" if engine_type == 'turbofan' else "AT76",
    # acName="AT76",
    allData=allData
)

# create a Flight Trajectory object to store the output from TCL segment calculations
ft = FT()

# default parameters
deltaTemp = 0  # [K] delta temperature from ISA

#%%

# rating_list = ["MCMB", "MTKF", "MCRZ"]
rating_list = ["MCMB"]
# rating_list = ["MTKF"]
if engine_type == 'turbofan':
    h_array = np.linspace(0, 12e3, 13)
    M_array = np.linspace(0, 0.9, 10)
elif engine_type == 'turboprop':
    h_array = np.linspace(0, 7620, 10)
    M_array = np.linspace(0, 0.42, 10)#[1:]
h_grid, M_grid = np.meshgrid(h_array, M_array)
F_net_grid = np.zeros_like(h_grid)
TSFC_grid = np.zeros_like(h_grid)
ff_grid = np.zeros_like(h_grid)

# fig, ax = plt.subplots()

for rating in rating_list:

    for (i, j), h in np.ndenumerate(h_grid):
        M = M_grid[i, j]
        amb = Atmosphere(h)
        a = amb.speed_of_sound[0]
        
        TAS = M * a
        T_net = AC.Thrust(
            h=h, deltaTemp=0, rating=rating, v=TAS, config="IC",
        )
        FF = AC.ff(
            h=h, v=TAS, T=T_net, config='IC', flightPhase='Climb',
        )
        TSFC = FF / T_net
        
        F_net_grid[i, j] = T_net / 2
        TSFC_grid[i, j] = TSFC
        ff_grid[i, j] = FF / 2


    # ax.scatter(F_net_grid / 1e3, TSFC_grid * 1e6)

# plt.show()

if engine_type == 'turbofan':
    np.save('h_grid_tfan.npy', h_grid)
    np.save('M_grid_tfan.npy', M_grid)
    np.save('Tnet_grid_tfan.npy', F_net_grid)
    np.save('TSFC_grid_tfan.npy', TSFC_grid)
    np.save('ff_grid_tfan.npy', ff_grid)
elif engine_type == 'turboprop':
    np.save('h_grid_tprop.npy', h_grid)
    np.save('M_grid_tprop.npy', M_grid)
    np.save('Tnet_grid_tprop.npy', F_net_grid)
    np.save('TSFC_grid_tprop.npy', TSFC_grid)
    np.save('ff_grid_tprop.npy', ff_grid)

#%%

fig, ax = plt.subplots(figsize=(8, 6))

# Overlay contour of Mach number (M_array)
M_levels = np.unique(M_array)
M_levels[-1] -= 1e-6
# cs = ax.contour(F_net_grid/1000, TSFC_grid, M_grid, levels=M_levels, colors='blue', linewidths=0.8, zorder=10, extend='max')
# ax.clabel(cs, fmt='%0.3f', fontsize=8)
ax.scatter(F_net_grid/1000, TSFC_grid, color='blue')

# Overlay contour of altitude (h_array)
h_levels = np.unique(h_array)/1000
h_levels[-1] -= 1e-6
# cs2 = ax.contour(F_net_grid/1000, TSFC_grid, h_grid/1000, levels=h_levels, colors='red', linestyles='dashed', linewidths=0.8, zorder=10, extend='max')
# ax.clabel(cs2, fmt='%0.3f', fontsize=8)
ax.scatter(F_net_grid/1000, TSFC_grid, color='red')

# Labels and style
ax.set_xlabel("Net Thrust [kN]")
ax.set_ylabel("Sp. Fuel Consumption [g/(kN·s)]")
# ax.set_xlim(0, 120)
# ax.set_ylim(10, 24)

# plt.tight_layout()
plt.show()


