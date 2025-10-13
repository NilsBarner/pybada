r"""
This script is adapted from C:\Users\nmb48\Documents\GitHub\pybada\examples\trajectoryAC.py
and reproduces the results of the BADA User Interface -> Calculation tools
-> Start simple session with nominal A/C Gross Mass, default climb options (Climb at
given CAS/Mach), Hmo final pressure altitude and a step size of 500 ft,
for the ATR 72-600 (ICAO: AT76).
"""

__all__ = []

import sys
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
    acName="AT76",
    allData=allData
)

# create a Flight Trajectory object to store the output from TCL segment calculations
ft = FT()

# default parameters
speedType = "CAS"  # {M, CAS, TAS}
wS = 0  # [kt] wind speed
ba = 0  # [deg] bank angle
deltaTemp = 0  # [K] delta temperature from ISA

# Initial conditions
m_init = 20e3  # [kg] initial mass
CAS_init = 170.0  # [kt] Initial CAS
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

Hp_CR = 25000  # [ft] CRUISing level

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

#%% CLIMB
#%% constantSpeedROCD from 0 to 499

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=conv.ms2kt(cas_cl1),
    Hp_init=Hp_RWY,
    Hp_final=499,
    m_init=m_init,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% accDec at constant altitude to next ARPM speed

# current values
Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

[theta, delta, sigma] = atm.atmosphereProperties(
    h=conv.ft2m(500), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.climbSpeed(
    h=conv.ft2m(500),
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
    Hp_init=500,
    control=None,
    # phase="Climb",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% constantSpeedROCD from 500 to 999

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=CAS_final,
    Hp_init=Hp,
    Hp_final=999,
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
    h=conv.ft2m(1000), deltaTemp=deltaTemp
)
[cas_cl2, speedUpdated] = AC.ARPM.climbSpeed(
    h=conv.ft2m(1000),
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
    Hp_init=1000,
    control=None,
    # phase="Climb",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% constantSpeedROCD from 1000 to 1499

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

flightTrajectory = TCL.constantSpeedRating(
    AC=AC,
    speedType="CAS",
    v=CAS_final,
    Hp_init=Hp,
    Hp_final=1499,
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
    Hp_init=1500,
    control=None,
    # phase="Climb",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% constantSpeedROCD from 1500 to 9999

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

#%% accDec at constant altitude to next ARPM speed
# NILS: this segment is superfluous when Vcl1 = Vcl2 (length 0)

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
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (see '# check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed')
)

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=10000,
    control=None,
    # phase="Climb",
    phase="Cruise",  # NILS: otherwise get non-zero climb rate
    m_init=m_final,
    wS=wS,
    bankAngle=ba,
    deltaTemp=deltaTemp,
)
ft.append(AC, flightTrajectory)

#%% constantSpeedROCD from 10000 ft to Mach transition altitude (also called "crossover altitude")

Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

# calculate the crosover altitude for climb phase
crossoverAltitude = conv.m2ft(atm.crossOver(Vcl2, Mcl))

if crossoverAltitude < Hp_CR:

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
    
    ##% constantSpeedROCD above Mach transition altitude

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
    
else:
    
    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=conv.ms2kt(Vcl2),
        Hp_init=Hp,
        Hp_final=Hp_CR,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        deltaTemp=deltaTemp,
    )
    ft.append(AC, flightTrajectory)
r'''
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
        # Hp_init=Hp,
        Hp_init=Hp_CR,  # NILS: with above line get Hp_init = 24000 ft due to C:\Users\nmb48\Documents\GitHub\pybada\src\pyBADA\TCL.py:5925: UserWarning:
        Hp_final=crossoverAltitude,  # Value ROCD = 258.9409699262891 [ft/min] exceeds the service ceiling limit defined by minimum ROCD = 300 [ft/min] at the altitude 24000.0 [ft].
        m_init=m_final,  # resulting in code failure as now Hp_init < Hp_final
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
    applyLimits=False,  # NILS: otherwise CAS gets corrected if not within flight envelope (see '# check if the speed is within the limits of minimum and maximum speed from the flight envelope, if not, overwrite calculated speed with flight envelope min/max speed')
)

flightTrajectory = TCL.accDec(
    AC=AC,
    speedType="CAS",
    v_init=CAS_final,
    v_final=conv.ms2kt(cas_cl2),
    Hp_init=Hp,
    control=None,
    phase="Descent",
    # phase="Cruise",  # NILS: otherwise get non-zero climb rate
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
'''
#%%

# print and plot final trajectory
df = ft.getFT(AC=AC)

fig, ax = plt.subplots(figsize=(8, 6))
for _, seg in df.groupby((df["comment"] != df["comment"].shift()).cumsum()):
    print('seg["ROCD"][0] =', seg["ROCD"].to_numpy()[0])
    ax.plot(seg["TAS"], seg["Hp"], "-", label=seg["comment"].iloc[0])
    # ax.plot(seg["CAS"], seg["Hp"], "-", label=seg["comment"].iloc[0])    
    # ax.plot(seg["time"], seg["Hp"], "-", label=seg["comment"].iloc[0])
ax.set_xlabel("TAS [kt]")
ax.set_ylabel("Hp [ft]")
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
plt.show()

