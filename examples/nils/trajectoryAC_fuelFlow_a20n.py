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


# initialization of BADA3/4
# uncomment for testing different BADA family if available

badaVersion = "bada_316"

allData = Bada3Parser.parseAll(badaVersion=badaVersion)

AC = Bada3Aircraft(
    badaVersion=badaVersion,
    # acName="A20N",
    acName="A321",
    allData=allData,
)

# create a Flight Trajectory object to store the output from TCL segment calculations
ft = FT()

# default parameters
speedType = "CAS"  # {M, CAS, TAS}
wS = 0  # [kt] wind speed
ba = 0  # [deg] bank angle
deltaTemp = 0  # [K] delta temperature from ISA

# Initial conditions
# m_init = 68e3  # [kg] initial mass
m_init = 72e3  # [kg] initial mass
CAS_init = 310  # [kt] Initial CAS
Hp_RWY = 0  # [ft] CDG RWY26R elevation

# take-off conditions
[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(Hp_RWY), deltaTemp=deltaTemp
)  # atmosphere properties at RWY altitude
[cas_cl1, speedUpdated] = AC.ARPM.climbSpeed(
    h=conv.ft2m(Hp_RWY),
    mass=m_init,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
)  # [m/s] take-off CAS

Hp_CR = 40000  # [ft] CRUISing level

# BADA speed schedule
[Vcl1, Vcl2, Mcl] = AC.flightEnvelope.getSpeedSchedule(
    phase="Climb"
)  # BADA Climb speed schedule
[Vcr1, Vcr2, Mcr] = AC.flightEnvelope.getSpeedSchedule(
    phase="Cruise"
)  # BADA Cruise speed schedule
[Vdes1, Vdes2, Mdes] = AC.flightEnvelope.getSpeedSchedule(
    phase="Descent"
)  # BADA Descent speed schedule

# %% CLIMB
# %% constantSpeedROCD from 0 to 1499

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=conv.ms2kt(cas_cl1),
    Hp_init=Hp_RWY,
    Hp_final=1499,
    m_init=m_init,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(1500), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.climbSpeed(
    h=conv.ft2m(1500),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (search for below string in C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py)
)  # 'check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed'

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Climb",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% constantSpeedROCD from 1500 to 2999

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=CAS_final,
    Hp_init=Hp,
    Hp_final=2999,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(3000), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.climbSpeed(
    h=conv.ft2m(3000),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (search for below string in C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py)
)  # 'check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed'

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Climb",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% constantSpeedROCD from 3000 to 3999

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=CAS_final,
    Hp_init=Hp,
    Hp_final=3999,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(4000), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.climbSpeed(
    h=conv.ft2m(4000),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (search for below string in C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py)
)  # 'check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed'

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Climb",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% constantSpeedROCD from 4000 to 4999

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=CAS_final,
    Hp_init=Hp,
    Hp_final=4999,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(5000), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.climbSpeed(
    h=conv.ft2m(5000),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (search for below string in C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py)
)  # 'check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed'

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Climb",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% constantSpeedROCD from 5000 to 5999

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=CAS_final,
    Hp_init=Hp,
    Hp_final=5999,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(6000), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.climbSpeed(
    h=conv.ft2m(6000),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (search for below string in C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py)
)  # 'check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed'

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Climb",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% constantSpeedROCD from 6000 to 9999

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=min(conv.ms2kt(Vcl1), 250),
    Hp_init=Hp,
    Hp_final=9999,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(10000), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.climbSpeed(
    h=conv.ft2m(10000),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (search for below string in C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py)
)  # 'check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed'

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Climb",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% constantSpeedROCD from 10000 ft to Mach transition altitude (also called "crossover altitude")

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

# calculate the crosover altitude for climb phase
crossoverAltitude = conv.m2ft(atm.crossOver(Vcl2, Mcl))

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=conv.ms2kt(Vcl2),
    Hp_init=Hp,
    Hp_final=crossoverAltitude,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

# %% constantSpeedROCD above Mach transition altitude

# current values
Hp, m_final = ft.getFinalValue(AC, ["Hp", "mass"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="M",
    v=Mcl,
    Hp_init=Hp,
    Hp_final=Hp_CR,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)
r"""
#%% DESCENT
#%% constantSpeedROCD above Mach transition altitude

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

# calculate the crosover altitude for climb phase
crossoverAltitude = conv.m2ft(atm.crossOver(Vdes2, Mdes))

if crossoverAltitude < Hp_CR:

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="M",
        v=Mdes,
        Hp_init=Hp,
        Hp_final=crossoverAltitude,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        deltaTemp=deltaTemp,
    )
    ft.append(AC, flightTrajectory)
    
    ##% constantSpeedROCD from Mach transition altitude (also called "crossover altitude") to 10000 ft
    
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])
    
    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=conv.ms2kt(Vdes2),
        Hp_init=Hp,
        Hp_final=10000,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        deltaTemp=deltaTemp,
    )
    ft.append(AC, flightTrajectory)
    
else:

    ##% constantSpeedROCD from Mach transition altitude (also called "crossover altitude") to 10000 ft
    
    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=conv.ms2kt(Vdes2),
        Hp_init=Hp,
        Hp_final=10000,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        deltaTemp=deltaTemp,
    )
    ft.append(AC, flightTrajectory)

#%% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(9999), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.descentSpeed(
    h=conv.ft2m(9999),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (search for below string in C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py)
)  # 'check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed'

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Descent",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% constantSpeedROCD from 9999 to 6000

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=min(conv.ms2kt(Vdes1), 250),
    Hp_init=9999,
    Hp_final=6000,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(5999), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.descentSpeed(
    h=conv.ft2m(5999),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (search for below string in C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py)
)  # 'check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed'

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=5999,
    control=None,
    # phase="Descent",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% constantSpeedROCD from 5999 to 3000

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    # v=min(conv.ms2kt(Vdes1), 250),  # NILS: I believe this is a typo on page 32 ofEIH-Technical-Report-220512-45, as this does not match the figure produced using the BADA User Interface
    v=CAS_final,
    Hp_init=5999,
    Hp_final=3000,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(2999), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.descentSpeed(
    h=conv.ft2m(2999),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (search for below string in C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py)
)  # 'check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed'

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Descent",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% constantSpeedROCD from 2999 to 2000

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=CAS_final,
    Hp_init=2999,
    Hp_final=2000,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(1999), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.descentSpeed(
    h=conv.ft2m(1999),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (search for below string in C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\bada3.py)
)  # 'check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed'

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Descent",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% constantSpeedROCD from 1999 to 1500

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=CAS_final,
    Hp_init=1999,
    Hp_final=1500,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(1499), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.descentSpeed(
    h=conv.ft2m(1499),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (see '# check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed')
)

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Descent",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% constantSpeedROCD from 1499 to 1000

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=CAS_final,
    Hp_init=1499,
    Hp_final=1000,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(999), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.descentSpeed(
    h=conv.ft2m(999),
    mass=m_final,
    theta=theta,
    delta=delta,
    deltaTemp=deltaTemp,
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (see '# check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed')
)

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    # phase="Descent",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% constantSpeedROCD from 999 to 0

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=CAS_final,
    Hp_init=999,
    Hp_final=0,
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)
"""
# %%

# print and plot final trajectory
df = ft.getFT(AC=AC)

fig, ax = plt.subplots(figsize=(8, 6))
for _, seg in df.groupby((df["comment"] != df["comment"].shift()).cumsum()):
    ax.plot(seg["TAS"], seg["Hp"], "-", label=seg["comment"].iloc[0])
    # ax.plot(seg["CAS"], seg["Hp"], "-", label=seg["comment"].iloc[0])
    # ax.plot(seg["time"], seg["Hp"], "-", label=seg["comment"].iloc[0])
ax.set_xlabel("TAS [kt]")
ax.set_ylabel("Hp [ft]")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
plt.show()

# %% FUEL FLOW

crossAlt = conv.ft2m(crossoverAltitude)

# define aircraft mass - here as reference mass
mass = AC.MREF

Hp_ft_vals = []
ff_vals = []

for Hp, CAS, TAS, M, ROCD in zip(
    df["Hp"].to_numpy(),
    df["CAS"].to_numpy(),
    df["TAS"].to_numpy(),
    df["M"].to_numpy(),
    df["ROCD"].to_numpy(),
):
    Hp = conv.ft2m(Hp)
    TAS = conv.kt2ms(TAS)
    CAS = conv.kt2ms(CAS)
    ROCD = conv.ft2m(ROCD / 60)  # NILS: ROCD in `df` in ft/min, convert to m/s

    # atmosphere properties
    theta, delta, sigma = atm.atmosphereProperties(h=Hp, deltaTemp=deltaTemp)

    # determine the aerodynamic configuration if necesary
    config = AC.flightEnvelope.getConfig(
        h=Hp, phase="Climb", v=CAS, mass=mass, deltaTemp=deltaTemp
    )

    # calculate Energy Share Factor depending if aircraft is flying constant M or CAS (based on crossover altitude)
    if Hp < crossAlt:
        ESF = AC.esf(
            h=Hp, flightEvolution="constCAS", M=M, deltaTemp=deltaTemp
        )
    else:
        ESF = AC.esf(h=Hp, flightEvolution="constM", M=M, deltaTemp=deltaTemp)

    adaptedThrust = False

    n = 1.0
    CL = AC.CL(sigma=sigma, mass=mass, tas=TAS, nz=n)
    CD = AC.CD(CL=CL, config=config)
    Drag = AC.D(sigma=sigma, tas=TAS, CD=CD)
    Thrust = AC.Thrust(
        rating="ADAPTED",
        v=TAS,
        config=config,
        h=Hp,
        ROCD=ROCD,
        mass=mass,
        acc=0,
        deltaTemp=deltaTemp,
        Drag=Drag,
    )
    ff = AC.ff(
        flightPhase="Climb",
        v=TAS,
        h=Hp,
        T=Thrust,
        config=config,
        adapted=adaptedThrust,
    )

    fl = int(utils.proper_round(conv.m2ft(Hp) / 100))
    Hp_ft = conv.m2ft(Hp)
    print(f"{fl:>4d}  {Hp_ft:>8.0f}  {config:>3}  {M:>6.3f}  {ff:>10.6f}")

    Hp_ft_vals.append(Hp_ft)
    ff_vals.append(float(ff))


# %% Fuel flow vs altitude

fig, ax = plt.subplots()

ax.plot(
    df["Hp"],
    np.diff(df["FUELCONSUMED"], prepend=df["FUELCONSUMED"][0])
    / np.diff(df["time"], prepend=df["time"][0]),
    label="np.diff(FUELCONSUMED)",
)
ax.plot(
    df["Hp"],
    np.gradient(df["FUELCONSUMED"]) / np.gradient(df["time"]),
    label="np.gradient(FUELCONSUMED)",
)
ax.plot(df["Hp"], df["FUEL"], label="FUEL")
ax.plot(Hp_ft_vals, ff_vals, label="Calculation (constant n=1)")

ax.set_xlabel("Altitude (ft)")
ax.set_ylabel("Fuel Flow (kg/s)")
ax.legend()

plt.show()

# %% TSFC vs altitude

TSFC = df["FUEL"] / df["THR"] * 1e6

fig, ax = plt.subplots()

ax.plot(df["Hp"], TSFC, label="TSFC")

ax.set_xlabel("Altitude (ft)")
ax.set_ylabel("TSFC (g/kN/s)")
ax.legend()

plt.show()

# %%


def mdot_f_ndim_func(mdot_f, P_02, T_02, A, LCV, C_p):
    """(3) in medium_paper_2010_perfemissPESO_veramorales"""
    return mdot_f * LCV / (P_02 * A * np.sqrt(C_p * T_02))


def F_G_ndim_func(F_G, P_amb, A_N, P_02):
    """(5) in medium_paper_2010_perfemissPESO_veramorales"""
    return (F_G + P_amb * A_N) / (A_N * P_02)


# %%

import pyromat as pm
from ambiance import Atmosphere

air = pm.get("ig.air")

A_nozzle_core = 0.34308
A_nozzle_bp = 1.05524
A_nozzle_tot = A_nozzle_core + A_nozzle_bp
LCV = 43e6

mdotf_array = df["FUEL"].to_numpy() / 60
Tgross_array = df["THR"].to_numpy()

Cp2_array = np.zeros(len(df["Hp"]))
p02_array = np.zeros_like(Cp2_array)
T02_array = np.zeros_like(Cp2_array)
pamb_array = np.zeros_like(Cp2_array)

for i, (h_ft, M) in enumerate(zip(df["Hp"], df["M"])):
    h_m = h_ft * 0.3048

    amb = Atmosphere(h_m)
    p0, T0 = amb.pressure[0], amb.temperature[0]

    Cp = air.cp(T=T0, p=p0)
    gamma = air.gam(T=T0, p=p0)
    p02 = p0 * (1 + (gamma - 1) / 2 * M**2) ** (gamma / (gamma - 1))
    T02 = T0 * (1 + (gamma - 1) / 2 * M**2)

    Cp2_array[i] = Cp
    p02_array[i] = p02
    T02_array[i] = T02
    pamb_array[i] = p0


mdot_f_ndim_array = mdot_f_ndim_func(
    mdotf_array, p02_array, T02_array, A_nozzle_tot, LCV, Cp2_array
)
F_G_ndim_array = F_G_ndim_func(
    Tgross_array, pamb_array, A_nozzle_tot, p02_array
)

# %%

fig, ax = plt.subplots()

ax.scatter(F_G_ndim_array, mdot_f_ndim_array)

plt.show()
