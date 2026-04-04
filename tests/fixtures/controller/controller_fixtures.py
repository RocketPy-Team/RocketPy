import pytest


## Controller
@pytest.fixture
def controller_function():
    """Create a controller function that updates the air brakes deployment level
    based on the altitude and vertical velocity of the rocket. This is the same
    controller function that is used in the air brakes example in the
    documentation.

    Returns
    -------
    function
        A controller function
    """

    def controller_function(  # pylint: disable=unused-argument
        time, sampling_rate, state, state_history, observed_variables, air_brakes
    ):
        z = state[2]
        vz = state[5]
        previous_vz = state_history[-1][5]
        if time < 3.9:
            return None
        if z < 1500:
            air_brakes.deployment_level = 0
        else:
            new_deployment_level = (
                air_brakes.deployment_level + 0.1 * vz + 0.01 * previous_vz**2
            )
            if new_deployment_level > air_brakes.deployment_level + 0.2 / sampling_rate:
                new_deployment_level = air_brakes.deployment_level + 0.2 / sampling_rate
            elif (
                new_deployment_level < air_brakes.deployment_level - 0.2 / sampling_rate
            ):
                new_deployment_level = air_brakes.deployment_level - 0.2 / sampling_rate
            else:
                new_deployment_level = air_brakes.deployment_level
            air_brakes.deployment_level = new_deployment_level

    return controller_function


@pytest.fixture
def controller_function_with_environment():
    """Create a controller function that uses the environment parameter to access
    atmospheric conditions without relying on global variables. This demonstrates
    the new environment parameter feature for air brakes controllers.

    Returns
    -------
    function
        A controller function that uses environment parameter
    """

    def controller_function(  # pylint: disable=unused-argument
        time,
        sampling_rate,
        state,
        state_history,
        observed_variables,
        air_brakes,
        sensors,
        environment,
    ):
        # state = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]
        altitude_asl = state[2]  # altitude above sea level
        altitude_agl = (
            altitude_asl - environment.elevation
        )  # altitude above ground level
        vx, vy, vz = state[3], state[4], state[5]

        # Use environment parameter instead of global variable
        wind_x = environment.wind_velocity_x(altitude_asl)
        wind_y = environment.wind_velocity_y(altitude_asl)

        # Calculate Mach number using environment data
        free_stream_speed = ((wind_x - vx) ** 2 + (wind_y - vy) ** 2 + (vz) ** 2) ** 0.5
        mach_number = free_stream_speed / environment.speed_of_sound(altitude_asl)

        if time < 3.9:
            return None

        if altitude_agl < 1500:
            air_brakes.deployment_level = 0
        else:
            previous_vz = state_history[-1][5] if state_history else vz
            new_deployment_level = (
                air_brakes.deployment_level + 0.1 * vz + 0.01 * previous_vz**2
            )
            # Rate limiting
            max_change = 0.2 / sampling_rate
            if new_deployment_level > air_brakes.deployment_level + max_change:
                new_deployment_level = air_brakes.deployment_level + max_change
            elif new_deployment_level < air_brakes.deployment_level - max_change:
                new_deployment_level = air_brakes.deployment_level - max_change

            air_brakes.deployment_level = new_deployment_level

        # Return observed variables including Mach number
        return (time, air_brakes.deployment_level, mach_number)

    return controller_function
