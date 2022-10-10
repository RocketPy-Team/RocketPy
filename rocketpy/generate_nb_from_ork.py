from turtle import distance
from xml.sax.handler import property_lexical_handler
import os
import nbformat
from bs4 import BeautifulSoup
import numpy as np
import orhelper
from orhelper import FlightDataType, FlightEvent

def generate(ork_file, nb_file, eng_file):
    nb = nbformat.read(nb_file, nbformat.NO_CONVERT)
    bs = BeautifulSoup(open(ork_file).read())
    team_name = ork_file.split("/")[2]

    if os.path.exists('GeneratedSimulations') is False:
        os.mkdir('GeneratedSimulations')

    if os.path.exists(f'GeneratedSimulations/{team_name}') is False:
        os.mkdir(f'GeneratedSimulations/{team_name}')

    path = f'GeneratedSimulations/{team_name}/'

    with orhelper.OpenRocketInstance() as instance:
        orh = orhelper.Helper(instance)
        ork = orh.load_doc(ork_file)
        launch_altitude = 160

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

        rocket_radius = ork.getRocket().getChild(0).getChild(1).getAftRadius()

        normalize_propelant_mass_vect = np.array(propelant_mass_vect)
        normalize_propelant_mass_vect = normalize_propelant_mass_vect - normalize_propelant_mass_vect[np.argmin(normalize_propelant_mass_vect)]
        propelant_mass_vect = list(normalize_propelant_mass_vect)
        propelant_mass = propelant_mass_vect[0]
        
        #TODO Resolver isso
        longitudinal_moment_of_inertia = [float(datapoint.text.split(',')[21]) for datapoint in bs.find_all('datapoint')]
        rotational_moment_of_inertia = [float(datapoint.text.split(',')[21]) for datapoint in bs.find_all('datapoint')]

        motor_label = bs.find('motormount').parent.find('name').text
        motor_cm = elements[motor_label]['DistanceToCG']
        nozzle_cm = motor_cm - elements[motor_label]['length']/2
        # proess burnout from eng_file
        # what to do about the propellent data
        # nozzle radius
        # throuat radius
        burnout_position = np.argwhere(np.array(propelant_mass_vect)==0)[0][0]
        burnout_time = time_vect[burnout_position]
        tube_length = elements[motor_label]["length"]

        # rocket_mass = mass[0] - propelant_mass_vect[0]
        nb['cells'][15]['source'] = generate_motor_code(path, 
                                                        burnout_position, 
                                                        burnout_time, 
                                                        thrust_vect, 
                                                        time_vect, 
                                                        propelant_mass, 
                                                        rocket_radius, 
                                                        tube_length)
        

        drag_coefficient = np.array([time_vect, drag_coefficient]).T
        drag_coefficient_source_path = f'{path}/drag_coefficient.csv'
        np.savetxt(drag_coefficient_source_path, drag_coefficient, delimiter=",")
        inertiaZ = 0.0351
        nb['cells'][20]['source'] = f'Calisto = Rocket(\n    motor=Pro75M1670,\n    radius={rocket_radius},\n    mass={mass},\n    inertiaI={longitudinal_moment_of_inertia[0]},\n    inertiaZ={inertiaZ},\n    distanceRocketNozzle={nozzle_cm},\n    distanceRocketPropellant={motor_cm},\n    powerOffDrag="drag_coefficient.csv",\n    powerOnDrag="drag_coefficient.csv",\n)\n\nCalisto.setRailButtons([0.2, -0.5])'

        nosecone = bs.find('nosecone')
        nosecone = generate_nosecone_code(bs, empty_rocket_cm)        

        trapezoidal_fin = trapezoidal_fin_code(bs, elements)

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

        launch_heading = float(bs.find('launchrodangle').text)
        launchrod_direction = min(87, float(bs.find('launchroddirection').text))
        nb['cells'][30]['source'] = f'TestFlight = Flight(rocket=Calisto, environment=Env, inclination={launchrod_direction}, heading={launch_heading})'
        print(np.max(altitude_vect))

    nbformat.write(nb, f'{path}Simulation.ipynb')

def search_cd_chute_if_auto(bs):
    return float(next(filter(lambda x: x.text.replace('.', '').isnumeric(), bs.findAll('cd'))).text)

def calculate_rocket_cg(ork):
    cg_rocket = 0
    top_position = 0
    rocket_mass = ork.getSectionMass()
    for section in ork.getChild(0).getChildren():
        cg_rocket += section.getSectionMass() * (top_position + section.getCG().x) / rocket_mass
        top_position += section.getLength()
    
    return cg_rocket

after_parent_element = ['After the parent component', 'Bottom of the parent component']
before_parent_element = ['Top of the parent component']

def calculate_distance_to_cg(ork, rocket_cg, top_position):

    element_position = ork.getRelativePosition().toString()
    if element_position in after_parent_element:
        relative_position = top_position + ork.getLength() - ork.getCG().x
    elif element_position in before_parent_element:
        relative_position = top_position - ork.getCG().x
    else:
        relative_position = top_position + ork.getLength() - ork.getCG().x
        print(element_position)

    distance_to_cg = rocket_cg - relative_position
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

def trapezoidal_fin_code(bs, elements):
    fin = bs.find('trapezoidfinset')
    n_fin = int(fin.find('fincount').text)
    root_chord = float(fin.find('rootchord').text)
    tip_chord = float(fin.find('tipchord').text)
    span = float(fin.find('sweeplength').text)
    fin_label = fin.find('name').text
    fin_distance_to_cm = elements[fin_label]["DistanceToCG"]
    trapezoidal_fin = f'Calisto.addTrapezoidalFins({n_fin}, rootChord={root_chord}, tipChord={tip_chord}, span={span}, distanceToCM={fin_distance_to_cm}, radius=None, airfoil=None)'        
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

def generate_nosecone_code(bs, empty_rocket_cm):
    nosecone = bs.find('nosecone')
    nosecone_length = float(nosecone.find('length').text)
    nosecone_shape = nosecone.find('shape').text
    nosecone_shape = 'lvhaack' if nosecone_shape == 'haack' else nosecone_shape
    nosecone_distanceToCM = empty_rocket_cm - nosecone_length
    nosecone = f'NoseCone = Calisto.addNose(length={nosecone_length}, kind="{nosecone_shape}", distanceToCM={nosecone_distanceToCM})'
    return nosecone

def generate_motor_code(path, burnout_position, burnout_time, thrust_vect, time_vect, propelant_mass, rocket_radius, rocket_height):
    grain_density = 1815 * 2
    grain_volume = propelant_mass / grain_density 
    inner_radius = np.sqrt(- grain_volume / rocket_height + rocket_radius**2)
    
    thrust_vect = np.array([time_vect[0: burnout_position], thrust_vect[0: burnout_position]]).T
    thrust_source_name = f'{path}/thrust_source.csv'
    np.savetxt(thrust_source_name, thrust_vect, delimiter=",")
    code = f'Pro75M1670 = SolidMotor(\n    thrustSource="thrust_source.csv",\n    burnOut={burnout_time},\n    grainNumber=1,\n    grainSeparation=0,\n    grainDensity=1815,\n    grainOuterRadius={rocket_radius},\n    grainInitialInnerRadius={inner_radius},\n    grainInitialHeight={rocket_height},\n    nozzleRadius=0,\n    throatRadius=0,\n    interpolationMethod="linear",\n)'
    return code