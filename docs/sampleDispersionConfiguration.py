from rocketpyBeta import *
from numpy.random import normal, uniform, choice
from datetime import datetime

IN = {
"impulse": (1240, 100),
"burnOut": (2.84, 0.15),
"nozzleRadius": (30/1000, 0.5/1000),
"throatRadius": (8/1000, 0.5/1000),
"grainSeparation": (1/1000, 0.5/1000),
"grainDensity": (1707, 24),
"grainOuterRadius": (21.05/1000, 0.137/1000),
"grainInitialInnerRadius": (9.63/1000, 0.076/1000),
"grainInitialHeight": (118.38/1000, 0.415/1000),
"m_prop": (1.664, 0.05),
"m_aero": (0.696, 0.02),
"inertiaI": (0.3437,0.01*0.3437),
"inertiaZ": (0.00288,0.01*0.00288),
"radius": (0.0378,0.0001),
"distanceRocketNozzle": (0.467,0.003),
"distanceRocketPropellant": (0.091,0.003),
"powerOffDrag": (1,0.03),
"powerOnDrag": (1,0.03),
"noseLength": (0.151, 0.001),
"noseDistanceToCM": (0.539, 0.003),
"tail1TopRadius": (0.0378, 0.0001),
"tail1BottomRadius": (0.0602/2, 0.0001),
"tail1Length": (0.00765, 0.0001),
"tail1DistanceToCM": (0.168, 0.003),
"tail2Length": (0.00580, 0.0001),
"tail2TopRadius": (0.0602/2,0.0001),
"tail2BottomRadius": (0.0723/2,0.0001),
"tail2DistanceToCM": (-0.3374,0.003),
"tail3Length": (0.005, 0.0005),
"tail3TopRadius": (0.0723/2, 0.0001),
"tail3BottomRadius": (0.0411/2, 0.0001),
"tail3DistanceToCM": (-0.4624, 0.0001),
"finSpan": (0.070, 0.001),
"finRootChord": (0.08, 0.001),
"finTipChord": (0.04, 0.001),
"finDistanceToCM": (-0.344, 0.003),
"inclination": (85, 1),
"heading": (90, 1),
"m_rec": (0.160, 0.024),
"CdS": (0.43, 0.086),
"lag_rec": (1 , 0.5),
"m_se": (0.300, 0.02),
"lag_se": (0.73, 0.16)}

while True:
    # Number of simulations
    s = 500

    print('Initializing new dispersion analysis sequence.')
    print('Euporia I - Plan A - Balistic')
    print('Number of simulations: '+str(s))
    print('Estimated time: ' + str(1.5*s/60) + ' mins')
    print(datetime.now())
    init = datetime.now()

    # Initialize output
    inputs = []
    output = []

    # Enviroment Variabels
    envRailLength = normal(2, 0.01, s)
    envYear = choice(np.arange(2013, 2017), s)
    envDay = choice(np.arange(1, 10), s)
    # envHour = choice([18, 12], s, p=[0.5, 0.5])

    # Motor Variables
    motorBurnOut = normal(*IN['burnOut'], s)
    motorTotalImpulse = normal(*IN['impulse'], s)
    motornozzleRadius = normal(*IN['nozzleRadius'], s)
    motorthroatRadius = normal(*IN['throatRadius'], s)
    motorgrainSeparation = normal(*IN['grainSeparation'], s)
    motorgrainDensity = normal(*IN['grainDensity'], s)
    motorgrainOuterRadius = normal(*IN['grainOuterRadius'], s)
    motorgrainInitialInnerRadius = normal(*IN['grainInitialInnerRadius'], s)
    motorgrainInitialHeight = normal(*IN['grainInitialHeight'], s)

    # Rocket Variables
    rMassSE = normal(*IN['m_se'], s)
    rMassRec = normal(*IN['m_rec'], s)
    rMassProp = normal(*IN['m_prop'], s)
    rMassAero = normal(*IN['m_aero'], s)
    rInertiaI = normal(*IN['inertiaI'], s)
    rInertiaZ = normal(*IN['inertiaZ'], s)
    rRadius = normal(*IN['radius'], s)
    rDistanceRocketNozzle = normal(*IN['distanceRocketNozzle'], s)
    rDistanceRocketPropellant = normal(*IN['distanceRocketPropellant'], s)
    rpowerOnDrag = normal(*IN['powerOnDrag'], s)
    rpowerOffDrag = normal(*IN['powerOffDrag'], s)

    # Nose
    rNoseLength = normal(*IN['noseLength'], s)
    rNoseDistanceToCM = normal(*IN['noseDistanceToCM'], s)
    # Fins
    rFinsSpan = normal(*IN['finSpan'], s)
    rFinsRootChord = normal(*IN['finRootChord'], s)
    rFinsTipChord = normal(*IN['finTipChord'], s)
    rFinsDistanceToCM = normal(*IN['finDistanceToCM'], s)
    # Tail 1
    rTail1TopRadius = normal(*IN['tail1TopRadius'], s)
    rTail1BottomRadius = normal(*IN['tail1BottomRadius'], s)
    rTail1Length = normal(*IN['tail1Length'], s)
    rTail1DistanceToCM = normal(*IN['tail1DistanceToCM'], s)
    # Tail 2
    rTail2TopRadius = normal(*IN['tail2TopRadius'], s)
    rTail2BottomRadius = normal(*IN['tail2BottomRadius'], s)
    rTail2Length = normal(*IN['tail2Length'], s)
    rTail2DistanceToCM = normal(*IN['tail2DistanceToCM'], s)
    # Tail 3
    rTail3TopRadius = normal(*IN['tail3TopRadius'], s)
    rTail3BottomRadius = normal(*IN['tail3BottomRadius'], s)
    rTail3Length = normal(*IN['tail3Length'], s)
    rTail3DistanceToCM = normal(*IN['tail3DistanceToCM'], s)

    # Parachute
    pDrogueCdS = normal(*IN['CdS'], s)
    pDrogueLag = normal(*IN['lag_rec'], s)
    dSeLag = normal(*IN['lag_se'], s)

    # Flight variables
    fInclination = normal(*IN['inclination'], s)
    fHeading = normal(*IN['heading'], s)

    # Initialize enviroment and motor
    E = Environment(railLength=2,
                    gravity=9.8,
                    windData='../data/weather/RioSaoPaulo.nc',
                    location=(-21.961526, -47.480908),
                    date=(2016, 2, 4, 12))
    for i in range(s):
        print('Iteration: ', i, end='\r')
        # Enviroment Variabels
        railLength = envRailLength[i]
        year = envYear[i]
        day = envDay[i]
        hour = 12

        # Motor Variables
        burnOut = motorBurnOut[i]
        totalImpulse = motorTotalImpulse[i]
        nozzleRadius = motornozzleRadius[i]   
        throatRadius = motorthroatRadius[i]
        grainSeparation = motorgrainSeparation[i]
        grainDensity = motorgrainDensity[i]
        grainOuterRadius = motorgrainOuterRadius[i]
        grainInitialInnerRadius = motorgrainInitialInnerRadius[i]
        grainInitialHeight = motorgrainInitialHeight[i]

        # Rocket Variables
        m_aeroI = rMassAero[i]
        m_recI = rMassRec[i]
        m_seI = rMassSE[i]
        m_propI = rMassProp[i]
        mass = m_aeroI + m_recI + m_seI + m_propI
        inertiaI = rInertiaI[i]
        inertiaZ = rInertiaZ[i]
        radius = rRadius[i]
        distanceRocketNozzle = rDistanceRocketNozzle[i]
        distanceRocketPropellant = rDistanceRocketPropellant[i]
        powerOnDrag = rpowerOnDrag[i]
        powerOffDrag = rpowerOffDrag[i]
        # Nose
        noseLength = rNoseLength[i]
        noseDistanceToCM = rNoseDistanceToCM[i]
        # Fins
        finSpan = rFinsSpan[i]
        finRootChord = rFinsRootChord[i]
        finTipChord = rFinsTipChord[i]
        finDistanceToCM = rFinsDistanceToCM[i]
        # Tail 1
        tail1TopRadius = rTail1TopRadius[i]
        tail1BottomRadius = rTail1BottomRadius[i]
        tail1Length = rTail1Length[i]
        tail1DistanceToCM = rTail1DistanceToCM[i]
        # Tail 2
        tail2TopRadius = rTail2TopRadius[i]
        tail2BottomRadius = rTail2BottomRadius[i]
        tail2Length = rTail2Length[i]
        tail2DistanceToCM = rTail2DistanceToCM[i]
        # Tail 3
        tail3TopRadius = rTail3TopRadius[i]
        tail3BottomRadius = rTail3BottomRadius[i]
        tail3Length = rTail3Length[i]
        tail3DistanceToCM = rTail3DistanceToCM[i]

        # Parachute
        drogueCdS = pDrogueCdS[i]
        drogueLag = pDrogueLag[i] + dSeLag[i]

        # Flight variables
        inclination = fInclination[i]
        heading = fHeading[i]

        inputs.append([year, day, hour, railLength, burnOut, totalImpulse, mass, inertiaI, inertiaZ, radius, inclination, heading])

        E.setDate((year, 2, day, hour))
        E.railLength = railLength
        Jiboia58 = Motor(thrustSource='../data/jiboia/thrustCurve.csv',
                  burnOut=2.84,
                  reshapeThrustCurve=(burnOut, totalImpulse),
                  interpolationMethod='spline',
                  nozzleRadius=nozzleRadius,
                  throatRadius=throatRadius,
                  grainNumber=5,
                  grainSeparation=grainSeparation,
                  grainDensity=grainDensity,
                  grainOuterRadius=grainOuterRadius,
                  grainInitialInnerRadius=grainInitialInnerRadius,
                  grainInitialHeight=grainInitialHeight)
        EuporiaI = Rocket(motor=Jiboia58,
                   mass=m_aeroI+m_propI+m_recI+m_seI,
                   inertiaI=inertiaI,
                   inertiaZ=inertiaZ,
                   radius=radius,
                   distanceRocketNozzle=distanceRocketNozzle,
                   distanceRocketPropellant=distanceRocketPropellant,
                   offCenter=0,
                   powerOffDrag="../data/euporia/euporiaIDragOff.csv",
                   powerOnDrag="../data/euporia/euporiaIDragOn.csv",
                   drogueArea=False,
                   drogueCd=False,
                   drogueLag=drogueLag,
                   mainArea=False,
                   mainCd=False,
                   mainAlt=50)
        EuporiaI.powerOffDrag = powerOffDrag*EuporiaI.powerOffDrag
        EuporiaI.powerOnDrag = powerOnDrag*EuporiaI.powerOnDrag
        EuporiaI.addNose(length=noseLength, kind='parabolic', distanceToCM=noseDistanceToCM)
        EuporiaI.addTail(topRadius=tail1TopRadius, bottomRadius=tail1BottomRadius, length=tail1Length, distanceToCM=tail1DistanceToCM)
        EuporiaI.addTail(topRadius=tail2TopRadius, bottomRadius=tail2BottomRadius, length=tail2Length, distanceToCM=tail2DistanceToCM)
        EuporiaI.addFins(n=4, rootChord=finRootChord, tipChord=finTipChord, span=finSpan, distanceToCM=finDistanceToCM)
        EuporiaI.addTail(topRadius=tail3TopRadius, bottomRadius=tail3BottomRadius, length=tail3Length, distanceToCM=tail3DistanceToCM)
        F = Flight(EuporiaI, E, inclination=inclination, heading=heading, flightPhases=-1, timeStep=[0.01, 0.1])
        # Calculate Max Vel
        sol = np.array(F.solution)
        F.vx = Function(sol[:, [0, 4]], 'Time (s)', 'Vx (m/s)', 'spline', extrapolation="natural")
        F.vy = Function(sol[:, [0, 5]], 'Time (s)', 'Vy (m/s)', 'spline', extrapolation="natural")
        F.vz = Function(sol[:, [0, 6]], 'Time (s)', 'Vz (m/s)', 'spline', extrapolation="natural")
        F.v = (F.vx**2 + F.vy**2 + F.vz**2)**0.5
        F.v.setDiscrete(0, burnOut, 100)
        F.maxVel = np.amax(F.v.source[:, 1])
        # Output
        output.append([F.outOfRailTime, F.outOfRailVelocity, F.maxVel, F.apogeeTime, F.apogee, F.apogeeX, F.apogeeY,
                       F.drogueOpeningTime, F.drogueOpeningVelocity, F.drogueX, F.drogueY, F.drogueZ,
                       F.tFinal, F.xImpact, F.yImpact, F.impactVelocity, F.rocket.staticMargin])
    # Write to file
    print('Sequence completed!')
    id = str(choice(200000))
    np.savetxt('InpDispersion' + id + '.euporia_I_AB', inputs, delimiter=',')
    np.savetxt('OutDispersion' + id + '.euporia_I_AB', output, delimiter=',')
    print('Results written to file!')
    print('End Time:', datetime.now())
    print('Total Elapsed Time (min):', (datetime.now() - init).seconds/60)
    print('Avarage Time (s):', (datetime.now() - init).seconds/s)
    print()