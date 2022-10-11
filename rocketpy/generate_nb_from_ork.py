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
import json

flight_parameters = {}

def generate(ork_file, nb_file, eng_file, open_rocket_instance):
    nb = nbformat.read(nb_file, nbformat.NO_CONVERT)
    bs = BeautifulSoup(open(ork_file).read())
    team_name = ork_file.split("/")[2]

    if os.path.exists('GeneratedSimulations') is False:
        os.mkdir('GeneratedSimulations')

    if os.path.exists(f'GeneratedSimulations/{team_name}') is False:
        os.mkdir(f'GeneratedSimulations/{team_name}')

    path = f'GeneratedSimulations/{team_name}/'

    #TODO: rail button
    #TODO: Correct the drag coefficient
    #TODO: equip 25

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
    
    drag_coefficient = np.array([mach_number, drag_coefficient]).T
    drag_coefficient = drag_coefficient[~np.isnan(drag_coefficient).any(axis=1), :]
    
    drag_coefficient_source_path = f'{path}drag_coefficient.csv'
    np.savetxt(drag_coefficient_source_path, drag_coefficient, delimiter=",")

    nosecone = bs.find('nosecone')
    nosecone = generate_nosecone_code(bs, elements[bs.find('nosecone').find('name').text]['DistanceToCG'])        

    trapezoidal_fin = ''
    # migue na distancia pro cg
    for idx, fin in enumerate(bs.findAll('trapezoidfinset')):
        element_label = fin.find('name').text
        trapezoidal_fin += trapezoidal_fin_code(fin, elements[element_label], idx)

    tail = tail_code(bs, elements, ork)

    chute_cell_code = ''
    chutes = bs.findAll('parachute')
    
    for main_chute in filter(lambda x: 'Main' in x.find('name').text, chutes):
        main_cds = search_cd_chute_if_auto(bs) if main_chute.find('cd').text == 'auto' else float(main_chute.find('cd').text)
        main_deploy_delay = float(main_chute.find('deploydelay').text)
        main_deploy_altitude = float(main_chute.find('deployaltitude').text)

        main_code = f'Main = Calisto.addParachute(\n    "Main",\n    CdS=parameters["MainCds{idx}"],\n    trigger=mainTrigger,\n    samplingRate=105,\n    lag=parameters["MainDeployDelay{idx}"],\n    noise=(0, 8.3, 0.5),\n)'
        main_trigger = f'def mainTrigger(p, y):\n    # p = pressure\n    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]\n    # activate main when vz < 0 m/s and z < 800 + 1400 m (+1400 due to surface elevation).\n    return True if y[5] < 0 and y[2] < parameters["MainDeployAltitude{idx}"] + parameters["elevation"] else False'
        chute_cell_code += f'{main_trigger}\n\n{main_code}\n\n'
        flight_parameters.update({
            f'MainCds{idx}': main_cds, f'MainDeployDelay{idx}': main_deploy_delay, f'MainDeployAltitude{idx}': main_deploy_altitude
        })

    
    for idx, drogue in enumerate(filter(lambda x: 'Drogue' in x.find('name').text, chutes)):
        drogue_cds = search_cd_chute_if_auto(bs) if drogue.find('cd').text == 'auto' else float(drogue.find('cd').text)
        drogue_deploy_delay = float(drogue.find('deploydelay').text)
        drogue_code = f'Drogue = Calisto.addParachute(\n    "Drogue",\n    CdS=parameters["DrogueCds{idx}"],\n    trigger=drogueTrigger,\n    samplingRate=105,\n    lag=parameters["DrogueDeployDelay{idx}"],\n    noise=(0, 8.3, 0.5),\n)'
        drogue_trigger = 'def drogueTrigger(p, y):\n    # p = pressure\n    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]\n    # activate drogue when vz < 0 m/s.\n    return True if y[5] < 0 else False\n\n\n'
        chute_cell_code += f'{drogue_trigger}\n\n{drogue_code}\n'
        flight_parameters.update({
            f'DrogueCds{idx}': drogue_cds, f'DrogueDeployDelay{idx}': drogue_deploy_delay
        })
    
    flight_parameters.update({
            'rocketMass': rocket_mass, 'elevation': 160,
            'emptyRocketCm': empty_rocket_cm, 'radius': rocket_radius, 'MotorCm': motor_cm, 'distanceRocketNozzle': nozzle_cm,
            'inertiaI': longitudinal_moment_of_inertia, 'inertiaZ': rotational_moment_of_inertia, 'railLength': 12,
            'dragCoefficientSourcePath': drag_coefficient_source_path, 'inclination': 84, 'heading': 133
        })
    nb['cells'][4]['source'] = f'%matplotlib widget\n\nimport json\n\nparameters = json.loads(open("parameters.json").read())'
    nb['cells'][15]['source'] = generate_motor_code(path, 
                                                    burnout_position, 
                                                    burnout_time, 
                                                    thrust_vect, 
                                                    time_vect, 
                                                    propelant_mass, 
                                                    motor_radius, 
                                                    tube_length)
    nb['cells'][20]['source'] = f'Calisto = Rocket(\n    motor=Pro75M1670,\n    radius=parameters["radius"],\n    mass=parameters["rocketMass"],\n    inertiaI=parameters["inertiaI"],\n    inertiaZ=parameters["inertiaZ"],\n    distanceRocketNozzle=parameters["distanceRocketNozzle"],\n    distanceRocketPropellant=parameters["MotorCm"],\n    powerOffDrag="drag_coefficient.csv",\n    powerOnDrag="drag_coefficient.csv",\n)\n\nCalisto.setRailButtons([0.2, -0.5])'

    nb['cells'][23]['source'] = f'{nosecone}\n\n{trapezoidal_fin}\n\n{tail}'

    nb['cells'][27]['source'] = chute_cell_code

    nb['cells'][7]['source'] = f'Env = Environment(\n    railLength=parameters["railLength"], latitude=39.3897, longitude=-8.28896388889, elevation=parameters["elevation"]\n)'
    nb['cells'][30]['source'] = f'TestFlight = Flight(rocket=Calisto, environment=Env, inclination=parameters["inclination"], heading=parameters["heading"])'
    print(np.max(altitude_vect))

    with open(f'{path}parameters.json', 'w') as convert_file:
        convert_file.write(json.dumps(flight_parameters))

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
    if is_sub_component(ork) == True:
        element_position = ork.getRelativePosition().toString()
        relative_position = top_position + ork.getPositionValue()

        ## TODO Tip of the nosecone
        
        if element_position == 'Bottom of the parent component':
            relative_position += ork.getParent().getLength()
        elif element_position == 'Middle of the parent component':
            relative_position += ork.getParent().getLength()/2
    else:
        relative_position = top_position + ork.getLength()

    if (rocket_cg - relative_position) < 0:
        relative_position -= ork.getLength()

    distance_to_cg = rocket_cg - (relative_position)
    return distance_to_cg
    
def is_sub_component(ork):
    i = 0
    root = ork.getRoot()
    while ork != ork.getRoot():
        ork = ork.getParent()
        if root != ork.getParent():
            i+=1
    return True if i >= 2 else False


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
            if is_sub_component(ork.getChild(i)) == False:
                top_position += ork.getChild(i).getLength()
            i += 1
        except Exception:
            has_child = False
    return elements

def trapezoidal_fin_code(fin, element, idx):
    n_fin = int(fin.find('fincount').text)
    root_chord = float(fin.find('rootchord').text)
    tip_chord = float(fin.find('tipchord').text)
    span = float(fin.find('height').text)
    fin_distance_to_cm = element["DistanceToCG"]

    flight_parameters.update({
        f'finN{idx}': n_fin,
        f'finRootChord{idx}': root_chord,
        f'finTipChord{idx}': tip_chord,
        f'finSpan{idx}': span,
        f'finDistanceToCm{idx}': fin_distance_to_cm
    })

    trapezoidal_fin = f'Calisto.addTrapezoidalFins(parameters["finN{idx}"], rootChord=parameters["finRootChord{idx}"], tipChord=parameters["finTipChord{idx}"], span=parameters["finSpan{idx}"], distanceToCM=parameters["finDistanceToCm{idx}"], radius=None, airfoil=None)\n\n'        
    return trapezoidal_fin

def tail_code(bs, elements, ork):
    tail = bs.find('transition')
    if tail:
        tail_label = tail.find('name').text
        tail_ork = [ele for ele in ork.getRocket().getChild(0).getChildren()][-1]
        topRadius = tail_ork.getForeRadius()
        bottomRadius = tail_ork.getAftRadius()
        tail_length = float(tail.find('length').text)
        tail = f'Calisto.addTail(\n    topRadius=parameters["tailTopRadius"], bottomRadius=parameters["tailBottomRadius"], length=parameters["tailLength"], distanceToCM=parameters["tailDistanceToCM"]\n)'
        
        flight_parameters.update({
            'tailTopRadius': topRadius,
            'tailBottomRadius': bottomRadius,
            'tailLength': tail_length,
            'tailDistanceToCM': elements[tail_label]['DistanceToCG']
        })
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
        flight_parameters.update({'noseShapeParameter': nosecone_shape_parameter})
        #TODO: Shape
    nosecone_distanceToCM = cm

    flight_parameters.update({
            'noseLength': nosecone_length,
            'noseShape': nosecone_shape,
            'noseDistanceToCM': nosecone_distanceToCM
    })

    nosecone = f'NoseCone = Calisto.addNose(length=parameters["noseLength"], kind=parameters["noseShape"], distanceToCM=parameters["noseDistanceToCM"])'
    return nosecone

def generate_motor_code(path, burnout_position, burnout_time, thrust_vect, time_vect, propelant_mass, motor_radius, motor_height):
    inner_radius = motor_radius / 2
    grain_volume = np.pi * (motor_radius**2 - inner_radius**2) * motor_height
    grain_density = propelant_mass / grain_volume
    
    thrust_vect = np.array([time_vect[0: burnout_position], thrust_vect[0: burnout_position]]).T
    thrust_source_name = f'{path}/thrust_source.csv'
    np.savetxt(thrust_source_name, thrust_vect, delimiter=",")

    flight_parameters.update({
            'burnOut': time_vect[burnout_position],
            'grainDensity': grain_density,
            'grainInitialInnerRadius': inner_radius,
            'grainOuterRadius': motor_radius,
            'grainInitialHeight': motor_height,
            'nozzleRadius': 1.5*inner_radius,
            'throatRadius': inner_radius,
    })

    code = f'Pro75M1670 = SolidMotor(\n    thrustSource="thrust_source.csv",\n    burnOut=parameters["burnOut"],\n    grainNumber=1,\n    grainSeparation=0,\n    grainDensity=parameters["grainDensity"],\n    grainOuterRadius=parameters["grainOuterRadius"],\n    grainInitialInnerRadius=parameters["grainInitialInnerRadius"],\n    grainInitialHeight=parameters["grainInitialHeight"],\n    nozzleRadius=parameters["nozzleRadius"],\n    throatRadius=parameters["throatRadius"],\n    interpolationMethod="linear",\n)'
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
    ork_file = './Trajectory Simulations/Team12_OpenRocketProject_v3.04/rocket.ork'
    eng_file = ''
    nb_file = './docs/notebooks/getting_started.ipynb'
    generate(ork_file, nb_file, eng_file, instance)
    print(flight_parameters)
