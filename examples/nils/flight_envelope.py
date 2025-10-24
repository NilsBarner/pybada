r"""This script is adapted from C:\Users\nmb48\Documents\GitHub\pybada\examples\trajectoryAC.py
and reproduces the results of the BADA User Interface -> Calculation tools
-> Start simple session with nominal A/C Gross Mass, default climb options (Climb at
given CAS/Mach), Hmo final pressure altitude and a step size of 500 ft,
for the Airbus A320neo (ICAO: A20N)."""

__all__ = ["get_flight_envelope"]

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
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


@dataclass
class target:
    ROCDtarget: float = None
    slopetarget: float = None
    acctarget: float = None
    ESFtarget: float = None
    
    
def get_flight_envelope(
    plot_bool:bool=True, ax:Any=None, ac_name:str='A20N', speed_type:str='TAS',
) -> tuple([np.ndarray, Any]):
    
    # initialization of BADA3/4
    # uncomment for testing different BADA family if available
    
    badaVersion = "bada_316"
    
    allData = Bada3Parser.parseAll(badaVersion=badaVersion)
    
    AC = Bada3Aircraft(
        badaVersion=badaVersion,
        acName=ac_name,
        allData=allData,
    )
    
    flightEnvelope = FlightEnvelope(AC)
    
    # create a Flight Trajectory object to store the output from TCL segment calculations
    ft = FT()
    
    # default parameters
    speedType = speed_type  # {M, CAS, TAS}
    deltaTemp = 0  # [K] delta temperature from ISA
    
    # %% FUEL FLOW
    
    # define aircraft mass - here as reference mass
    mass = AC.MREF
    
    h_max_m = flightEnvelope.maxAltitude(mass, deltaTemp)
    # h_max_ft = h_max_m / 0.3048
    h_max_ft = 41e3
    
    colors = ["blue", "red"]
    # envelopeTypes = ['OPERATIONAL', 'CERTIFIED']
    envelopeTypes = ["OPERATIONAL"]
    phases = ["TO", "IC", "CR", "AP", "LD"]
    # phase = 'IC'
    
    if plot_bool == True:
        if ax == None:
            fig, ax = plt.subplots()
    
    for i, envelopeType in enumerate(envelopeTypes):
        color = colors[i]
    
        VMin_lists = []
        VMax_lists = []
        VStall_lists = []
    
        MMin_lists = []
        MMax_lists = []
        MStall_lists = []
        
        VMin_tas_lists = []
        VMax_tas_lists = []
        VStall_tas_lists = []
        
        T_lists = []
    
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
            
            VMin_tas_list = []
            VMax_tas_list = []
            VStall_tas_list = []
            
            T_list = []
    
            for j, h_m in enumerate(h_range_m):
                h_ft = h_m / 0.3048
    
                config = phase
                
                # print(vars(flightEnvelope))
                # sys.exit()
    
                # CAS
                VMin = flightEnvelope.VMin(
                    h_m,
                    mass,
                    config,
                    deltaTemp,
                    nz=1.2,
                    envelopeType=envelopeType,
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
                
                # TAS
                VMin_tas = atm.cas2Tas(cas=VMin, delta=delta, sigma=sigma)
                VMax_tas = atm.cas2Tas(cas=VMax, delta=delta, sigma=sigma)
                VStall_tas = atm.cas2Tas(cas=VStall, delta=delta, sigma=sigma)
                VMin_tas_list.append(VMin_tas)
                VMax_tas_list.append(VMax_tas)
                VStall_tas_list.append(VStall_tas)
                
                # Thrust
                T = flightEnvelope.Thrust(
                    h=h_m, deltaTemp=deltaTemp, rating='MTKF', v=VMin_tas, config=config  # rating ('MTKF', 'MCMB', 'MCRZ')
                )
                T_list.append(T)
                # print('T =', T)
                # sys.exit('Stop.')
    
            VMin_lists.append(VMin_list)
            VMax_lists.append(VMax_list)
            VStall_lists.append(VStall_list)
    
            MMin_lists.append(MMin_list)
            MMax_lists.append(MMax_list)
            MStall_lists.append(MStall_list)
            
            VMin_tas_lists.append(VMin_tas_list)
            VMax_tas_lists.append(VMax_tas_list)
            VStall_tas_lists.append(VStall_tas_list)
            
            T_lists.append(T_list)
    
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
        
        VMin_tas_arrays = np.array(VMin_tas_lists)
        VMax_tas_arrays = np.array(VMax_tas_lists)
        
        T_arrays = np.array(T_lists)
    
        VMin_list_max = np.max(VMin_arrays, axis=0)
        VMax_list_min = np.min(VMax_arrays, axis=0)
    
        MMin_list_max = np.max(MMin_arrays, axis=0)
        MMax_list_min = np.min(MMax_arrays, axis=0)
        
        VMin_tas_list_max = np.max(VMin_tas_arrays, axis=0)
        VMax_tas_list_min = np.min(VMax_tas_arrays, axis=0)
        
        T_list_max = np.max(T_arrays, axis=0)
    
        if speedType == "CAS":
            
            if plot_bool == True:
                ax.plot(
                    VMin_list_max * 1.94384,
                    h_range_m / 0.3048,
                    color=color,
                    marker=".",
                )
                ax.plot(
                    VMax_list_min * 1.94384,
                    h_range_m / 0.3048,
                    color=color,
                    marker=".",
                )
                ax.plot(
                    [VMin_list_max[-1] * 1.94384, VMax_list_min[-1] * 1.94384],
                    [h_range_m[-1] / 0.3048, h_range_m[-1] / 0.3048],
                    color=color,
                    marker=".",
                )
                ax.plot(
                    [VMin_list_max[0] * 1.94384, VMax_list_min[0] * 1.94384],
                    [h_range_m[0] / 0.3048, h_range_m[0] / 0.3048],
                    color=color,
                    marker=".",
                )
                ax.set_xlim(100, 350)
            
            # Collate coordinates
            envelope_x_coordinates = np.hstack((VMin_list_max, np.flip(VMax_list_min)))
            envelope_y_coordinates = np.hstack((h_range_m, np.flip(h_range_m)))
            envelope_coordinates = np.vstack(
                (envelope_x_coordinates, envelope_y_coordinates)
            ).T
    
    
        elif speedType == "M":
            
            if plot_bool == True:
                ax.plot(MMin_list_max, h_range_m / 0.3048, color=color, marker=".")
                ax.plot(MMax_list_min, h_range_m / 0.3048, color=color, marker=".")
                ax.plot(
                    [MMin_list_max[-1], MMax_list_min[-1]],
                    [h_range_m[-1] / 0.3048, h_range_m[-1] / 0.3048],
                    color=color,
                    marker=".",
                )
                ax.plot(
                    [MMin_list_max[0], MMax_list_min[0]],
                    [h_range_m[0] / 0.3048, h_range_m[0] / 0.3048],
                    color=color,
                    marker=".",
                )
                ax.set_xlim(0.2, 1)
                
                # Collate coordinates
                envelope_x_coordinates = np.hstack((MMin_list_max, np.flip(MMax_list_min)))
                envelope_y_coordinates = np.hstack((h_range_m, np.flip(h_range_m)))
                envelope_coordinates = np.vstack(
                    (envelope_x_coordinates, envelope_y_coordinates)
                ).T
            
            
        elif speedType == "TAS":
            
            if plot_bool == True:
                ax.plot(
                    VMin_tas_list_max * 1.94384,
                    h_range_m / 0.3048,
                    color=color,
                    marker=".",
                )
                ax.plot(
                    VMax_tas_list_min * 1.94384,
                    h_range_m / 0.3048,
                    color=color,
                    marker=".",
                )
                ax.plot(
                    [VMin_tas_list_max[-1] * 1.94384, VMax_tas_list_min[-1] * 1.94384],
                    [h_range_m[-1] / 0.3048, h_range_m[-1] / 0.3048],
                    color=color,
                    marker=".",
                )
                ax.plot(
                    [VMin_tas_list_max[0] * 1.94384, VMax_tas_list_min[0] * 1.94384],
                    [h_range_m[0] / 0.3048, h_range_m[0] / 0.3048],
                    color=color,
                    marker=".",
                )
                ax.set_xlim(0, 500)
            
            # Collate coordinates
            envelope_x_coordinates = np.hstack((VMin_tas_list_max, np.flip(VMax_tas_list_min)))
            envelope_y_coordinates = np.hstack((h_range_m, np.flip(h_range_m)))
            envelope_z_coordinates = np.hstack((T_list_max, np.flip(T_list_max)))
            envelope_coordinates = np.vstack(
                (envelope_x_coordinates, envelope_y_coordinates, envelope_z_coordinates)
            ).T
    
    
    if plot_bool == True:
        ax.set_ylim(-1000, 40e3)
        plt.show()
    
    return envelope_coordinates, ax

# %%

if __name__ == '__main__':
    
    envelope_coordinates_a20N = get_flight_envelope(plot_bool=False, ax=None, ac_name='A321')
    envelope_coordinates_a20N = get_flight_envelope(plot_bool=False, ax=None, ac_name='A20N')
    envelope_coordinates_at76 = get_flight_envelope(plot_bool=False, ax=None, ac_name='AT76')
    
    # np.save("envelope_coordinates_tfan.npy", envelope_coordinates_a20N)
    # np.save("envelope_coordinates_tprop.npy", envelope_coordinates_at76)
