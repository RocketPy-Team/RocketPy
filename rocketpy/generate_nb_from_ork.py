import xmltodict
import nbformat
from bs4 import BeautifulSoup
import numpy as np
import orhelper
from orhelper import FlightDataType, FlightEvent

def generate(ork_file, nb_file, eng_file):
    nb = nbformat.read(nb_file, nbformat.NO_CONVERT)
    bs = BeautifulSoup(open(ork_file).read())
    team_name = ork_file.split("/")[2]
    with orhelper.OpenRocketInstance() as instance:
        orh = orhelper.Helper(instance)
        ork = orh.load_doc('./Trajectory Simulations/Team12_OpenRocketProject_v3.04/rocket.ork')
        
        time_vect = [float(datapoint.text.split(',')[0]) for datapoint in bs.find_all('datapoint')]
        thrust_vect = [float(datapoint.text.split(',')[28]) for datapoint in bs.find_all('datapoint')]
        cg_location_vect = [float(datapoint.text.split(',')[24]) for datapoint in bs.find_all('datapoint')]
        cp_location_vect = [float(datapoint.text.split(',')[23]) for datapoint in bs.find_all('datapoint')]
        altitude_vect = [float(datapoint.text.split(',')[1]) for datapoint in bs.find_all('datapoint')]
        propelant_mass_vect = [float(datapoint.text.split(',')[20]) for datapoint in bs.find_all('datapoint')]
        mass = [float(datapoint.text.split(',')[19]) for datapoint in bs.find_all('datapoint')]
        drag_coefficient = [float(datapoint.text.split(',')[30]) for datapoint in bs.find_all('datapoint')]

        empty_rocket_cm = calculate_rocket_cg(ork.getRocket())
        mass = ork.getRocket().getSectionMass()
        elements = process_elements_position(ork.getRocket(), {}, empty_rocket_cm, mass, top_position=0)

        normalize_propelant_mass_vect = np.array(propelant_mass_vect)
        normalize_propelant_mass_vect = normalize_propelant_mass_vect - normalize_propelant_mass_vect[np.argmin(normalize_propelant_mass_vect)]
        propelant_mass_vect = list(normalize_propelant_mass_vect)

        #TODO Resolver isso
        longitudinal_moment_of_inertia = [float(datapoint.text.split(',')[21]) for datapoint in bs.find_all('datapoint')]
        rotational_moment_of_inertia = [float(datapoint.text.split(',')[21]) for datapoint in bs.find_all('datapoint')]

        motor_cm = elements['Booser Tube']['DistanceToCG']
        nozzle_cm = motor_cm - elements['Booser Tube']['length']/2
        # proess burnout from eng_file
        # what to do about the propellent data
        # nozzle radius
        # throuat radius
        burnout_position = np.argwhere(np.array(propelant_mass_vect)==0)[0][0]
        burnout_time = time_vect[burnout_position]

        # rocket_mass = mass[0] - propelant_mass_vect[0]
        nb['cells'][15]['source'] = generate_motor_code(team_name, burnout_position, burnout_time, thrust_vect, time_vect)
        

        drag_coefficient = np.array([time_vect, drag_coefficient]).T
        
        rocket_radius = ork.getRocket().getChild(0).getChild(1).getAftRadius()
        drag_coefficient_source_path = f'{ork_file.split("/")[2]}_drag_coefficient.csv'
        np.savetxt(drag_coefficient_source_path, drag_coefficient, delimiter=",")
        nb['cells'][20]['source'] = f'Calisto = Rocket(\n    motor=Pro75M1670,\n    radius={rocket_radius},\n    mass={mass},\n    inertiaI={longitudinal_moment_of_inertia[0]},\n    inertiaZ=0,\n    distanceRocketNozzle={nozzle_cm},\n    distanceRocketPropellant={motor_cm},\n    powerOffDrag="{drag_coefficient_source_path}",\n    powerOnDrag="{drag_coefficient_source_path}",\n)\n\nCalisto.setRailButtons([0.2, -0.5])'

        nosecone = bs.find('nosecone')
        nosecone = generate_nosecone_code(bs, empty_rocket_cm)        

        fin = bs.find('trapezoidfinset')
        n_fin = int(fin.find('fincount').text)
        root_chord = float(fin.find('rootchord').text)
        tip_chord = float(fin.find('tipchord').text)
        span = float(fin.find('sweeplength').text)
        fin_distance_to_cm = elements["Trapezoidal fin set"]["DistanceToCG"]
        trapezoidal_fin = f'Calisto.addTrapezoidalFins({n_fin}, rootChord={root_chord}, tipChord={tip_chord}, span={span}, distanceToCM={fin_distance_to_cm}, radius=None, airfoil=None)'

        tail = bs.find('transition')
        if tail:
            topRadius = ork.getRocket().getChild(0).getChild(5).getForeRadius()
            bottomRadius = ork.getRocket().getChild(0).getChild(5).getAftRadius()
            tail_length = float(tail.find('length').text)
            tail = f'Calisto.addTail(\n    topRadius={topRadius}, bottomRadius={bottomRadius}, length={tail_length}, distanceToCM={elements["Boattail"]["DistanceToCG"]}\n)'
        else:
            tail = ''
        # aerodynamic surfaces
        nb['cells'][23]['source'] = f'{nosecone}\n\n{trapezoidal_fin}\n\n{tail}'

        # main_chute = bs.findAll('parachute')[0]
        # main_cds = float(main_chute.find('cd').text)
        # main_position = float(main_chute.find('position').text)
        # main_deploy_delay = float(main_chute.find('deploydelay').text)
        # main_deployaltitude = float(main_chute.find('deployaltitude').text)

        # main_chute = f'Main = Calisto.addParachute(\n    "Main",\n    CdS={main_cds},\n    trigger=mainTrigger,\n    samplingRate=,\n    lag={main_deploy_delay},\n    noise=(0, 8.3, 0.5),\n)'

        # drogue = bs.findAll('parachute')[0]
        # drogue_cds = float(drogue.find('cd').text)
        # drogue_position = float(drogue.find('position').text)
        # drogue_deploy_delay = float(drogue.find('deploydelay').text)
        # drogue_deployaltitude = float(drogue.find('deployaltitude').text)
        # drogue = f'Drogue = Calisto.addParachute(\n    "Drogue",\n    CdS={drogue_cds},\n    trigger=drogueTrigger,\n    samplingRate=,\n    lag={drogue_deploy_delay},\n    noise=(0, 8.3, 0.5),\n)'

        # nb['cells'][27]['source'] = f'def drogueTrigger(p, y):\n    # p = pressure\n    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]\n    # activate drogue when vz < 0 m/s.\n    return True if y[5] < 0 else False\n\n\ndef mainTrigger(p, y):\n    # p = pressure\n    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]\n    # activate main when vz < 0 m/s and z < 800 + 1400 m (+1400 due to surface elevation).\n    return True if y[5] < 0 and y[2] < 800 + 1400 else False\n\n\n{main_chute}\n\n{drogue}'

        nb['cells'][30]['source'] = f'TestFlight = Flight(rocket=Calisto, environment=Env, inclination=85, heading=0)'

    nbformat.write(nb, f'{team_name}.ipynb')

def calculate_rocket_cg(ork):
    # mass = ork.getMass()
    # print(f'{ork.getName()}, mass: {ork.getMass()}, section_mass: {ork.getSectionMass()} component_mass: {ork.getComponentMass()}, cg: {ork.getCG().x}, length: {ork.getLength()}, relative position: {ork.getRelativePosition().toString()}, position: {ork.getPositionValue()}')
    # has_child = True
    # i = 0
    # while has_child:
    #     try:
    #         mass += iterator_ork(ork.getChild(i))
    #         i += 1
    #     except Exception:
    #         print(f'{ork.getName()} has {i+1} children')
    #         has_child = False

    # top_position = 0
    cg_rocket = 0
    top_position = 0
    rocket_mass = ork.getSectionMass()
    for section in ork.getChild(0).getChildren():
        cg_rocket += section.getSectionMass() * (top_position + section.getCG().x) / rocket_mass
        top_position += section.getLength()
    
    return cg_rocket

def process_elements_position(ork, elements, rocket_cg, rocket_mass, top_position=0):
    #TODO: Use relative position

    element = {
            'length': ork.getLength(),
            'CM': ork.getCG().x,
            'DistanceToCG': rocket_cg - (top_position + ork.getLength() - ork.getCG().x),
        }
    elements[ork.getName()] = element
    has_child = True
    i = 0
    while has_child:
        try:
            new_elements = process_elements_position(ork.getChild(i), {}, rocket_cg, rocket_mass, top_position)
            elements.update(new_elements)
            top_position += ork.getChild(i).getLength()
            i += 1
        except Exception:
            print(f'{ork.getName()} has {i+1} children')
            has_child = False
            print(elements)
    return elements


# def find_element_cg():


# def find_element_position(rocket, parent_element_position):
#     position = 0
#     for idx, parent_section in enumerate(rocket.getChild(0).getChildren()):
#         if idx ==  parent_element_position:
#             return position
#         position =+ parent_section.getLength()
#     return position


def generate_nosecone_code(bs, empty_rocket_cm):
    nosecone = bs.find('nosecone')
    nosecone_length = float(nosecone.find('length').text)
    nosecone_shape = nosecone.find('shape').text
    nosecone_shape = 'lvhaack' if nosecone_shape == 'haack' else nosecone_shape
    nosecone_distanceToCM = empty_rocket_cm - nosecone_length
    nosecone = f'NoseCone = Calisto.addNose(length={nosecone_length}, kind="{nosecone_shape}", distanceToCM={nosecone_distanceToCM})'
    return nosecone

def generate_motor_code(name, burnout_position, burnout_time, thrust_vect, time_vect):
    thrust_vect = np.array([time_vect[0: burnout_position], thrust_vect[0: burnout_position]]).T
    thrust_source_name = f'{name}_thrust_source.csv'
    np.savetxt(thrust_source_name, thrust_vect, delimiter=",")
    code = f'Pro75M1670 = SolidMotor(\n    thrustSource="{thrust_source_name}",\n    burnOut={burnout_time},\n    grainNumber=5,\n    grainSeparation=5/1000,\n    grainDensity=1815,\n    grainOuterRadius=33/1000,\n    grainInitialInnerRadius=15/1000,\n    grainInitialHeight=120/1000,\n    nozzleRadius=0,\n    throatRadius=0,\n    interpolationMethod="linear",\n)'
    return code