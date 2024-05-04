analysis_parameters = {
    # Mass Details
    # Rocket's dry mass without motor (kg) and its uncertainty (standard deviation)
    "rocket_mass": (7.257, 0.001),
    # Rocket's inertia moment perpendicular to its axis (kg*m^2)
    "rocket_inertia_11": (3.675, 0.03675),
    # Rocket's inertia moment relative to its axis (kg*m^2)
    "rocket_inertia_33": (0.007, 0.00007),
    # Motors's dry mass without propellant (kg) and its uncertainty (standard deviation)
    "motor_dry_mass": (1.000, 0.001),
    # Motor's dry inertia moment perpendicular to its axis (kg*m^2)
    "motor_inertia_11": (1.675, 0.01675),
    # Motors's dry inertia moment relative to its axis (kg*m^2)
    "motor_inertia_33": (0.003, 0.00003),
    # Distance between rocket's center of dry mass and motor's center of dry mass (m)
    "motor_dry_mass_position": (0.5, 0.001),
    # Propulsion Details - run help(SolidMotor) for more information
    # Motor total impulse (N*s)
    "impulse": (1415.15, 35.3),
    # Motor burn out time (s)
    "burn_time": (5.274, 1),
    # Motor's nozzle radius (m)
    "nozzle_radius": (21.642 / 1000, 0.5 / 1000),
    # Motor's nozzle throat radius (m)
    "throat_radius": (8 / 1000, 0.5 / 1000),
    # Motor's grain separation (axial distance between two grains) (m)
    "grain_separation": (6 / 1000, 1 / 1000),
    # Motor's grain density (kg/m^3)
    "grain_density": (1707, 50),
    # Motor's grain outer radius (m)
    "grain_outer_radius": (21.4 / 1000, 0.375 / 1000),
    # Motor's grain inner radius (m)
    "grain_initial_inner_radius": (9.65 / 1000, 0.375 / 1000),
    # Motor's grain height (m)
    "grain_initial_height": (120 / 1000, 1 / 1000),
    # Aerodynamic Details - run help(Rocket) for more information
    # Rocket's radius (kg*m^2)
    "radius": (40.45 / 1000, 0.001),
    # Distance between rocket's center of dry mass and nozzle exit plane (m) (negative)
    "nozzle_position": (-1.024, 0.001),
    # Distance between rocket's center of dry mass and and center of propellant mass (m) (negative)
    "grains_center_of_mass_position": (-0.571, 0.001),
    # Multiplier for rocket's drag curve. Usually has a mean value of 1 and a uncertainty of 5% to 10%
    "power_off_drag": (0.9081 / 1.05, 0.033),
    # Multiplier for rocket's drag curve. Usually has a mean value of 1 and a uncertainty of 5% to 10%
    "power_on_drag": (0.9081 / 1.05, 0.033),
    # Rocket's nose cone length (m)
    "nose_length": (0.274, 0.001),
    # Axial distance between rocket's center of dry mass and nearest point in its nose cone (m)
    "nose_distance_to_CM": (1.134, 0.001),
    # Fin span (m)
    "fin_span": (0.077, 0.0005),
    # Fin root chord (m)
    "fin_root_chord": (0.058, 0.0005),
    # Fin tip chord (m)
    "fin_tip_chord": (0.018, 0.0005),
    # Axial distance between rocket's center of dry mass and nearest point in its fin (m)
    "fin_distance_to_CM": (-0.906, 0.001),
    # Launch and Environment Details - run help(Environment) and help(Flight) for more information
    # Launch rail inclination angle relative to the horizontal plane (degrees)
    "inclination": (84.7, 1),
    # Launch rail heading relative to north (degrees)
    "heading": (53, 2),
    # Launch rail length (m)
    "rail_length": (5.7, 0.0005),
    # Members of the ensemble forecast to be used
    "ensemble_member": list(range(10)),
    # Parachute Details - run help(Rocket) for more information
    # Drag coefficient times reference area for the drogue chute (m^2)
    "cd_s_drogue": (0.349 * 1.3, 0.07),
    # Time delay between parachute ejection signal is detected and parachute is inflated (s)
    "lag_rec": (1, 0.5),
    # Electronic Systems Details - run help(Rocket) for more information
    # Time delay between sensor signal is received and ejection signal is fired (s)
    "lag_se": (0.73, 0.16),
}
