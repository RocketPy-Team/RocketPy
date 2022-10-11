from math import e
from multiprocessing.sharedctypes import Value
from turtle import distance
from xml.sax.handler import property_lexical_handler
import os
import nbformat
from bs4 import BeautifulSoup
import numpy as np
import orhelper
from orhelper import FlightDataType, FlightEvent
import zipfile

def generate(ork_file, nb_file, eng_file, open_rocket_instance):
    #TODO Drag em função de mach
    nb = nbformat.read(nb_file, nbformat.NO_CONVERT)
    bs = BeautifulSoup(open(ork_file).read())
    team_name = ork_file.split("/")[2]

    if os.path.exists('GeneratedSimulations') is False:
        os.mkdir('GeneratedSimulations')

    if os.path.exists(f'GeneratedSimulations/{team_name}') is False:
        os.mkdir(f'GeneratedSimulations/{team_name}')

    path = f'GeneratedSimulations/{team_name}/'

    #rail button

    orh = orhelper.Helper(instance)
    ork = orh.load_doc(ork_file)
    launch_altitude = 160

    time_vect = [float(datapoint.text.split(',')[0]) for datapoint in bs.find_all('datapoint')]
    starting_pos = 0
    final_pos = len(time_vect) - 1
    for idx, position in enumerate(time_vect):
        if position == 0:
            starting_pos = idx
    time_vect = time_vect[starting_pos: final_pos]
    thrust_vect = [float(datapoint.text.split(',')[28]) for datapoint in bs.find_all('datapoint')][starting_pos: final_pos]
    cg_location_vect = [float(datapoint.text.split(',')[24]) for datapoint in bs.find_all('datapoint')][starting_pos: final_pos]
    cp_location_vect = [float(datapoint.text.split(',')[23]) for datapoint in bs.find_all('datapoint')][starting_pos: final_pos]
    altitude_vect = [float(datapoint.text.split(',')[1]) for datapoint in bs.find_all('datapoint')][starting_pos: final_pos]
    propelant_mass_vect = [float(datapoint.text.split(',')[20]) for datapoint in bs.find_all('datapoint')][starting_pos: final_pos]
    mass = [float(datapoint.text.split(',')[19]) for datapoint in bs.find_all('datapoint')][starting_pos: final_pos]
    drag_coefficient = [float(datapoint.text.split(',')[30]) for datapoint in bs.find_all('datapoint')][starting_pos: final_pos]
    mach_number = [float(datapoint.text.split(',')[26]) for datapoint in bs.find_all('datapoint')][starting_pos: final_pos]

    normalize_propelant_mass_vect = np.array(propelant_mass_vect)
    normalize_propelant_mass_vect = normalize_propelant_mass_vect - normalize_propelant_mass_vect[np.argmin(normalize_propelant_mass_vect)]
    propelant_mass_vect = list(normalize_propelant_mass_vect)
    propelant_mass = propelant_mass_vect[0]
    burnout_position = np.argwhere(np.array(propelant_mass_vect)==0)[0][0]

    rocket_mass = mass[0] - propelant_mass_vect[0]
    empty_rocket_cm = cg_location_vect[burnout_position] # calculate_rocket_cg(ork.getRocket(), rocket_mass)
    elements = process_elements_position(ork.getRocket(), {}, empty_rocket_cm, rocket_mass, top_position=0)
    rocket_radius = ork.getRocket().getChild(0).getChild(1).getAftRadius()
    
    #TODO Resolver isso
    last_element = [key for key in elements.keys()][-1]
    motor_cm = empty_rocket_cm - (cg_location_vect[0] * mass[0] - empty_rocket_cm * rocket_mass)/propelant_mass
    nozzle_cm = elements[last_element]['DistanceToCG'] - elements[last_element]['length']
    # proess burnout from eng_file
    # what to do about the propellent data
    # nozzle radius
    # throuat radius
    longitudinal_moment_of_inertia = [float(datapoint.text.split(',')[21]) for datapoint in bs.find_all('datapoint')][burnout_position]
    rotational_moment_of_inertia = [float(datapoint.text.split(',')[22]) for datapoint in bs.find_all('datapoint')][burnout_position]
    burnout_time = time_vect[burnout_position]
    tube_length = float(bs.find('motormount').find('length').text)
    motor_radius = float(bs.find('motormount').find('diameter').text) / 2

    # rocket_mass = mass[0] - propelant_mass_vect[0]
    nb['cells'][15]['source'] = generate_motor_code(path, 
                                                    burnout_position, 
                                                    burnout_time, 
                                                    thrust_vect, 
                                                    time_vect, 
                                                    propelant_mass, 
                                                    motor_radius, 
                                                    tube_length)
    drag_coefficient = np.sort(np.array([mach_number, drag_coefficient]).T, 0)
    drag_coefficient_source_path = f'{path}drag_coefficient.csv'
    np.savetxt(drag_coefficient_source_path, drag_coefficient, delimiter=",")
    nb['cells'][20]['source'] = f'Calisto = Rocket(\n    motor=Pro75M1670,\n    radius={rocket_radius},\n    mass={rocket_mass},\n    inertiaI={longitudinal_moment_of_inertia},\n    inertiaZ={rotational_moment_of_inertia},\n    distanceRocketNozzle={nozzle_cm},\n    distanceRocketPropellant={motor_cm},\n    powerOffDrag="drag_coefficient.csv",\n    powerOnDrag="drag_coefficient.csv",\n)\n\nCalisto.setRailButtons([0.2, -0.5])'

    nosecone = bs.find('nosecone')
    nosecone = generate_nosecone_code(bs, elements[bs.find('nosecone').find('name').text]['DistanceToCG'])        

    trapezoidal_fin = ''
    # migue na distancia pro cg
    for fin in bs.findAll('trapezoidfinset'):
        element_label = fin.find('name').text
        trapezoidal_fin += trapezoidal_fin_code(fin, elements[element_label])

    tail = tail_code(bs, elements, ork)
    # aerodynamic surfaces
    nb['cells'][23]['source'] = f'{nosecone}\n\n{trapezoidal_fin}\n\n{tail}'

    # filter by name = main
    # sampling rate
    # trigger functions
    chute_cell_code = ''
    chutes = bs.findAll('parachute')
    
    for main_chute in filter(lambda x: 'Main' in x.find('name').text, chutes):
        main_cds = search_cd_chute_if_auto(bs) if main_chute.find('cd').text == 'auto' else float(main_chute.find('cd').text)
        main_deploy_delay = float(main_chute.find('deploydelay').text)
        main_deploy_altitude = float(main_chute.find('deployaltitude').text)

        main_code = f'Main = Calisto.addParachute(\n    "Main",\n    CdS={main_cds},\n    trigger=mainTrigger,\n    samplingRate=105,\n    lag={main_deploy_delay},\n    noise=(0, 8.3, 0.5),\n)'
        main_trigger = f'def mainTrigger(p, y):\n    # p = pressure\n    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]\n    # activate main when vz < 0 m/s and z < 800 + 1400 m (+1400 due to surface elevation).\n    return True if y[5] < 0 and y[2] < {main_deploy_altitude} + {launch_altitude} else False'
        chute_cell_code += f'{main_trigger}\n\n{main_code}\n\n'

    
    for drogue in filter(lambda x: 'Drogue' in x.find('name').text, chutes):
        drogue_cds = search_cd_chute_if_auto(bs) if drogue.find('cd').text == 'auto' else float(drogue.find('cd').text)
        drogue_deploy_delay = float(drogue.find('deploydelay').text)
        drogue_code = f'Drogue = Calisto.addParachute(\n    "Drogue",\n    CdS={drogue_cds},\n    trigger=drogueTrigger,\n    samplingRate=105,\n    lag={drogue_deploy_delay},\n    noise=(0, 8.3, 0.5),\n)'
        drogue_trigger = 'def drogueTrigger(p, y):\n    # p = pressure\n    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]\n    # activate drogue when vz < 0 m/s.\n    return True if y[5] < 0 else False\n\n\n'
        chute_cell_code += f'{drogue_trigger}\n\n{drogue_code}\n'

    nb['cells'][27]['source'] = chute_cell_code

    launch_rod_length = float(bs.find('launchrodlength').text)
    nb['cells'][7]['source'] = f'Env = Environment(\n    railLength={launch_rod_length}, latitude=39.3897, longitude=-8.28896388889, elevation={launch_altitude}\n)'
    nb['cells'][30]['source'] = f'TestFlight = Flight(rocket=Calisto, environment=Env, inclination=84, heading=133)'
    print(np.max(altitude_vect))

    nbformat.write(nb, f'{path}Simulation.ipynb')

def search_cd_chute_if_auto(bs):
    return float(next(filter(lambda x: x.text.replace('.', '').isnumeric(), bs.findAll('cd'))).text)

def calculate_rocket_cg(ork, rocket_mass):
    cg_rocket = 0
    top_position = 0
    for section in ork.getChild(0).getChildren():
        cg_rocket += section.getSectionMass() * (top_position + section.getCG().x) / rocket_mass
        top_position += section.getLength()
    
    return cg_rocket

after_parent_element = ['After the parent component', 'Bottom of the parent component']
before_parent_element = ['Top of the parent component']

def calculate_distance_to_cg(ork, rocket_cg, top_position):
    element_position = ork.getRelativePosition().toString()
    if element_position in after_parent_element and (rocket_cg - top_position) > 0:
        relative_position = top_position + ork.getLength()
    elif (element_position in before_parent_element):
        relative_position = top_position
    elif element_position == 'Bottom of the parent component':
        relative_position = top_position + ork.getParent().getLength() + ork.getPositionValue() - ork.getLength()
    else:
        relative_position = top_position
        print(element_position)

    distance_to_cg = rocket_cg - (relative_position)
    return distance_to_cg
    


def process_elements_position(ork, elements, rocket_cg, rocket_mass, top_position=0):
    #TODO: Use relative position
    element = {
            'length': ork.getLength(),
            'CM': ork.getCG().x,
            'DistanceToCG': calculate_distance_to_cg(ork, rocket_cg, top_position)
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
            has_child = False
    return elements

def trapezoidal_fin_code(fin, element):
    n_fin = int(fin.find('fincount').text)
    root_chord = float(fin.find('rootchord').text)
    tip_chord = float(fin.find('tipchord').text)
    span = float(fin.find('height').text)
    fin_distance_to_cm = element["DistanceToCG"]
    trapezoidal_fin = f'Calisto.addTrapezoidalFins({n_fin}, rootChord={root_chord}, tipChord={tip_chord}, span={span}, distanceToCM={fin_distance_to_cm}, radius=None, airfoil=None)\n\n'        
    return trapezoidal_fin

def tail_code(bs, elements, ork):
    tail = bs.find('transition')
    if tail:
        tail_label = tail.find('name').text
        tail_ork = [ele for ele in ork.getRocket().getChild(0).getChildren()][-1]
        topRadius = tail_ork.getForeRadius()
        bottomRadius = tail_ork.getAftRadius()
        tail_length = float(tail.find('length').text)
        tail = f'Calisto.addTail(\n    topRadius={topRadius}, bottomRadius={bottomRadius}, length={tail_length}, distanceToCM={elements[tail_label]["DistanceToCG"]}\n)'
    else:
        tail = ''    
    return tail

def generate_nosecone_code(bs, cm):
    nosecone = bs.find('nosecone')
    nosecone_length = float(nosecone.find('length').text)
    nosecone_shape = nosecone.find('shape').text
    # TODO: Verificar shape da ogiva
    if nosecone_shape == 'haack':
        nosecone_shape_parameter = float(nosecone.find('shapeparameter').text)
        nosecone_shape = 'Von Karman' if nosecone_shape_parameter == 0.0 else 'lvhaack'
    nosecone_distanceToCM = cm
    nosecone = f'NoseCone = Calisto.addNose(length={nosecone_length}, kind="{nosecone_shape}", distanceToCM={nosecone_distanceToCM})'
    return nosecone

def generate_motor_code(path, burnout_position, burnout_time, thrust_vect, time_vect, propelant_mass, motor_radius, motor_height):
    inner_radius = motor_radius / 2
    grain_volume = np.pi * (motor_radius**2 - inner_radius**2) * motor_height
    grain_density = propelant_mass / grain_volume
    
    thrust_vect = np.array([time_vect[0: burnout_position], thrust_vect[0: burnout_position]]).T
    thrust_source_name = f'{path}/thrust_source.csv'
    np.savetxt(thrust_source_name, thrust_vect, delimiter=",")
    code = f'Pro75M1670 = SolidMotor(\n    thrustSource="thrust_source.csv",\n    burnOut={burnout_time},\n    grainNumber=1,\n    grainSeparation=0,\n    grainDensity={grain_density},\n    grainOuterRadius={motor_radius},\n    grainInitialInnerRadius={inner_radius},\n    grainInitialHeight={motor_height},\n    nozzleRadius={1.5*inner_radius},\n    throatRadius={inner_radius},\n    interpolationMethod="linear",\n)'
    return code

with orhelper.OpenRocketInstance() as instance:
    i = 0
    j = 0
    # for folder in os.listdir('Trajectory Simulations'):
    #     path = f'Trajectory Simulations/{folder}'
    #     for root, dirs, files in os.walk(path):
    #         for file in files:
    #             if file.endswith(".ork"):
    #                 try:
    #                     ork_file = os.path.join(root, file)
    #                     eng_file = ''
    #                     nb_file = './docs/notebooks/getting_started.ipynb'
    #                     generate(ork_file, nb_file, eng_file, instance)
    #                     i += 1
    #                 except UnicodeDecodeError as exc:
    #                     print('Found zipped file, skipping it')
    #                     continue
    #                 except ValueError as exc:
    #                     print('Incomplete file')
    #                     print(str(exc))
    #                     j += 1
    #                 except Exception as exc:
    #                     print('Error hile generating file')
    #                     print(str(exc))
    # print(f'{i} Sucessful file generation')
    # print(f'{j} Incomplete files')
    ork_file = './Trajectory Simulations/Team02_OpenRocketProject_v1/Periphas_EUROC/rocket.ork'
    eng_file = ''
    nb_file = './docs/notebooks/getting_started.ipynb'
    generate(ork_file, nb_file, eng_file, instance)