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

from pyBADA import TCL as TCL
from pyBADA import atmosphere as atm
from pyBADA import conversions as conv
from pyBADA.bada3 import Bada3Aircraft
from pyBADA.bada3 import FlightEnvelope
from pyBADA.bada3 import Parser as Bada3Parser
from pyBADA.bada4 import Bada4Aircraft
from pyBADA.bada4 import Parser as Bada4Parser
from pyBADA.flightTrajectory import FlightTrajectory as FT
from pyBADA import utils

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
    allData=allData
)

flightEnvelope = FlightEnvelope(AC)

# create a Flight Trajectory object to store the output from TCL segment calculations
ft = FT()

# default parameters
speedType = "CAS"  # {M, CAS, TAS}
deltaTemp = 0  # [K] delta temperature from ISA

#%% FUEL FLOW

# define aircraft mass - here as reference mass
mass = AC.MREF

h_max_m = flightEnvelope.maxAltitude(mass, deltaTemp)
h_max_ft = h_max_m / 0.3048

colors = ['blue', 'red']
# envelopeTypes = ['OPERATIONAL', 'CERTIFIED']
envelopeTypes = ['OPERATIONAL']
phases = ['TO', 'IC', 'CR', 'AP', 'LD']
# phase = 'IC'

fig, ax = plt.subplots()

for i, envelopeType in enumerate(envelopeTypes):
    
    color = colors[i]

    VMin_lists = []
    VMax_lists = []
    VStall_lists = []
    
    MMin_lists = []
    MMax_lists = []
    MStall_lists = []
    
    for phase in phases:
    
        dh = 500
        h_range_ft = np.append(np.arange(0, h_max_ft, dh), h_max_ft)
        h_range_m = h_range_ft * 0.3048
        
        VMin_list = []
        VMax_list = []
        VStall_list = []
        
        MMin_list = []
        MMax_list = []
        MStall_list = []
        
        for j, h_m in enumerate(h_range_m):
            h_ft = h_m / 0.3048
            
            config = phase
            
            # CAS
            VMin = flightEnvelope.VMin(
                h_m, mass, config, deltaTemp, nz=1.2, envelopeType=envelopeType,
            )
            VMax = flightEnvelope.VMax(h_m, deltaTemp)
            VStall = flightEnvelope.VStall(mass, config)
            VMin_list.append(VMin)
            VMax_list.append(VMax)
            VStall_list.append(VStall)
            
            # Mach number
            [theta, delta, sigma] = atm.atmosphereProperties(
                h=h_m, deltaTemp=deltaTemp
            )
            MMin = atm.cas2Mach(
                cas=VMin, theta=theta, delta=delta, sigma=sigma
            )
            MMax = atm.cas2Mach(
                cas=VMax, theta=theta, delta=delta, sigma=sigma
            )
            MStall = atm.cas2Mach(
                cas=VStall, theta=theta, delta=delta, sigma=sigma
            )
            MMin_list.append(MMin)
            MMax_list.append(MMax)
            MStall_list.append(MStall)
            
        VMin_lists.append(VMin_list)
        VMax_lists.append(VMax_list)
        VStall_lists.append(VStall_list)
        
        MMin_lists.append(MMin_list)
        MMax_lists.append(MMax_list)
        MStall_lists.append(MStall_list)
        
        # ax.plot(np.array(VMin_list) * 1.94384, h_range_m / 0.3048, color=color)
        # ax.plot(np.array(VMax_list) * 1.94384, h_range_m / 0.3048, color=color)
        # # ax.plot(np.array(VStall_list) * 1.94384, h_range_m / 0.3048, color=color)
        
        # ax.plot(MMin_list, h_range_m / 0.3048, color=color)
        # ax.plot(MMax_list, h_range_m / 0.3048, color=color)
        # # ax.plot(MStall_list, h_range_m / 0.3048, color=color)
    
    VMin_arrays = np.array(VMin_lists)
    VMax_arrays = np.array(VMax_lists)
    
    MMin_arrays = np.array(MMin_lists)
    MMax_arrays = np.array(MMax_lists)
    
    VMin_list_max = np.max(VMin_arrays, axis=0)
    VMax_list_min = np.min(VMax_arrays, axis=0)
    
    MMin_list_max = np.max(MMin_arrays, axis=0)
    MMax_list_min = np.min(MMax_arrays, axis=0)
    
    if speedType == 'CAS':
        ax.plot(VMin_list_max * 1.94384, h_range_m / 0.3048, color=color, marker='.')
        ax.plot(VMax_list_min * 1.94384, h_range_m / 0.3048, color=color, marker='.')
        ax.plot(
            [VMin_list_max[-1] * 1.94384, VMax_list_min[-1] * 1.94384],
            [h_range_m[-1] / 0.3048, h_range_m[-1] / 0.3048],
            color=color, marker='.',
        )
        ax.plot(
            [VMin_list_max[0] * 1.94384, VMax_list_min[0] * 1.94384],
            [h_range_m[0] / 0.3048, h_range_m[0] / 0.3048],
            color=color, marker='.',
        )
        ax.set_xlim(100, 350)
        
    if speedType == 'M':
        ax.plot(MMin_list_max, h_range_m / 0.3048, color=color, marker='.')
        ax.plot(MMax_list_min, h_range_m / 0.3048, color=color, marker='.')
        ax.plot(
            [MMin_list_max[-1], MMax_list_min[-1]],
            [h_range_m[-1] / 0.3048, h_range_m[-1] / 0.3048],
            color=color, marker='.',
        )
        ax.plot(
            [MMin_list_max[0], MMax_list_min[0]],
            [h_range_m[0] / 0.3048, h_range_m[0] / 0.3048],
            color=color, marker='.',
        )
        ax.set_xlim(0.2, 1)
        
ax.set_ylim(-1000, 40e3)

plt.show()

#%%

envelope_x_coordinates = np.hstack((
    MMin_list_max, np.flip(MMax_list_min)
))
envelope_y_coordinates = np.hstack((
    h_range_m, np.flip(h_range_m)
))
envelope_coordinates = np.vstack((
    envelope_x_coordinates, envelope_y_coordinates
)).T

if engine_type == 'turbofan':
    np.save('envelope_coordinates_tfan.npy', envelope_coordinates)
elif engine_type == 'turboprop':
    np.save('envelope_coordinates_tprop.npy', envelope_coordinates)
    
    
