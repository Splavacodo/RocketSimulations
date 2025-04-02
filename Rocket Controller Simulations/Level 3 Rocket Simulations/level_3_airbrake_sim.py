from rocketpy import Environment, SolidMotor, Rocket, Flight
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import datetime


def main():
    # Spaceport (New Mexico) latitude/longitude/elevation: latitude=32.990254, longitude=-106.974998, elevation=735
    # Jean Lake (Nevada) latitude/longitude/elevation: latitude=35.78, longitude=-115.25, elevation=847
    # Grand Junction (Colorado) latitude/longitude/elevation: latitude=39.279167, longitude=109, elevation=1499
    # UROC (Frank Hunt Field) latitude/longitude/elevation: latitude=39.25024, longitude=-111.75103, elevation=1615
    env = initialize_flight_environment(latitude=39.25024, longitude=-111.75103, elevation=1615, day_delta=3)
    some_rocket_name = initialize_base_rocket()

    # some_rocket_name.draw()

    base_test_flight = Flight(
        rocket=some_rocket_name,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True
    )

    # desired_height: int = 460  # height in meters (for J250 motor)
    # desired_height: int = 1080  # height in meters (for K400 motor)
    desired_height: int = 2987  # height in meters (for M2400 motor)

    # Gain parameters determined from root locus & step responses
    K: float = 0.1523
    K_p: float = 0.025 * K
    K_d: float = 1 * K

    print("K_p:", K_p)
    print("K_d:", K_d)

    # The controller function is within main since it needs a reference to the environment
    def controller_function(
            time, sampling_rate, state, state_history, observed_variables, air_brakes
    ):
        # state = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]

        altitude_ASL = state[2]  # Height relative to sea level
        altitude_AGL = altitude_ASL - env.elevation  # Height relative to ground level
        vx, vy, vz = state[3], state[4], state[5]

        # Get winds in x and y directions
        wind_x, wind_y = env.wind_velocity_x(altitude_ASL), env.wind_velocity_y(altitude_ASL)

        # Calculate Mach number
        free_stream_speed = (
                                    (wind_x - vx) ** 2 + (wind_y - vy) ** 2 + (vz) ** 2
                            ) ** 0.5
        mach_number: float = free_stream_speed / env.speed_of_sound(altitude_ASL)
        air_density: float = env.density(altitude_ASL)

        prev_time, _, _, _, prev_height_err, prev_controller_output = observed_variables[-1]
        height_err: float = desired_height - altitude_AGL

        # Check if the rocket has reached burnout
        if time < some_rocket_name.motor.burn_out_time:  # mojito.motor.burn_out_time
            new_deployment_level: float = 0
            controller_output: float = 0
        # If burn out is finished, apply PD control
        else:
            # ------------ Feedback Control ----------------
            # The controller in the time domain can be expressed as G_c(t) = F(t)/e(t)
            # The force of the air brake will be equal to F(t) = K_p * e(t) + K_d * e_dot(t)

            # At one point in time e(t) = Desired height - current height and e_dot(t) = (e(t) - e_prev)/dt
            # There will then be another function that takes in the force and determines the best
            # air brake deployment level to match that force

            # To calculate the derivative of the error, the error should be stored in observed variables to reference
            # the previous error from the last iteration
            # ----------------------------------------------
            deltaT: float = time - prev_time

            if deltaT > 0:
                e_dot: float = (height_err - prev_height_err) / (time - prev_time)
                controller_output: float = K_p * height_err + K_d * e_dot
            else:
                controller_output: float = prev_controller_output

            if height_err > 0:
                new_deployment_level: float = find_deployment_level(
                    mach_num=mach_number,
                    curr_vel=vz,
                    drag_force=-controller_output,
                    air_density=air_density
                )
            else:
                new_deployment_level = 0

        air_brakes.deployment_level = new_deployment_level

        # Return variables of interest to be saved in the observed_variables list
        return (
            time,
            altitude_AGL,
            air_brakes.deployment_level,
            air_brakes.drag_coefficient(air_brakes.deployment_level, mach_number),
            height_err,
            controller_output
        )

    air_brakes = some_rocket_name.add_air_brakes(
        drag_coefficient_curve="air_brake_deployment_data_level_3.csv",
        controller_function=controller_function,
        sampling_rate=200,
        reference_area=None,
        clamp=True,
        initial_observed_variables=[0, 0, 0, 0, 0, 0],
        override_rocket_drag=False,
        name="Air Brakes"
    )

    # air_brakes.all_info()

    controlled_test_flight = Flight(
        rocket=some_rocket_name,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        time_overshoot=False,
        terminate_on_apogee=True
    )

    time_list, altitude_list, deployment_level_list, drag_coefficient_list, height_err_list, controller_output_list = [], [], [], [], [], []

    obs_vars = controlled_test_flight.get_controller_observed_variables()

    base_flight_time: list[float] = []
    base_flight_altitude: list[float] = []

    with open("level_3_base_flight_altitude.csv") as altitude_file:
        altitude_file.readline()  # to skip over the data labels

        for data_point in altitude_file.readlines():
            time, altitude = data_point.strip().split(",")

            base_flight_time.append(float(time))
            base_flight_altitude.append(float(altitude) - env.elevation)

    for time, altitude, deployment_level, drag_coefficient, height_err, controller_output in obs_vars:
        time_list.append(time)
        altitude_list.append(altitude)
        deployment_level_list.append(deployment_level)
        drag_coefficient_list.append(drag_coefficient)
        height_err_list.append(height_err)
        controller_output_list.append(-controller_output)

    # Plot deployment level by time
    plt.plot(time_list, deployment_level_list)
    plt.xlabel("Time (s)")
    plt.ylabel("Deployment Level")
    plt.title("Deployment Level vs. Time")
    plt.grid()
    plt.show()

    # Plot of controller output by time
    plt.plot(time_list, controller_output_list)
    plt.xlabel("Time (s)")
    plt.ylabel("Controller Force Output (N)")
    plt.title("Controller Force Output vs. Time")
    plt.grid()
    plt.show()

    # Plot drag coefficient by time
    # plt.plot(time_list, drag_coefficient_list)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Drag Coefficient")
    # plt.title("Drag Coefficient by Time")
    # plt.grid()
    # plt.show()

    # Plot of height err with time
    # plt.plot(time_list, height_err_list)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Height Error [m]")
    # plt.title("Height Error vs. Time")
    # plt.grid()
    # plt.show()

    # Plot of the altitude of the rocket with time
    plt.plot(time_list, altitude_list)
    plt.plot(base_flight_time, base_flight_altitude)
    plt.axhline(y=desired_height, color='k', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (Above Ground Level) [m]")
    plt.title("Altitude (Above Ground Level) vs. Time")
    plt.legend(["Controlled Rocket Flight", "Base Rocket Flight", "Desired Height = " + str(desired_height) + " m"])
    plt.grid()
    plt.show()

    # Comparison of flights
    # plots.CompareFlights([base_test_flight, controlled_test_flight]).trajectories_3d()
    # base_test_flight.altitude()
    base_test_flight.prints.apogee_conditions()

    # Uncomment the line below to export the base flight data
    # base_test_flight.export_data("level_3_base_flight_altitude.csv", "z")

    # controlled_test_flight.aerodynamic_drag()
    controlled_test_flight.prints.apogee_conditions()
    # controlled_test_flight.altitude()

    print("Apogee Error:", desired_height - altitude_list[-1], "m")


def find_deployment_level(mach_num: float, curr_vel: float, drag_force: float, air_density: float) -> float:
    """
    Determines the deployment level of the air brakes to provide a given drag force and the current speed of the rocket.

    Args:
        mach_num (float): The Mach number the rocket is traveling at
        curr_vel (float): The current velocity of the rocket in meters/sec
        drag_force (float): The drag force provided by the controller to reach the desired height
        air_density (float): The air density of air given the current height of the rocket in kg/m^3

    Returns:
        deployment_level (float): The deployment level of the air brake to provide the given drag force
    """
    air_brake_area: float = 0.008787078  # m², the full air brake area

    # Helper function to find the drag coefficient for a given deployment level
    def drag_force_error(deployment_level: float) -> float:
        # Interpolate drag coefficient based on Mach number and deployment level
        drag_coefficient: float = find_air_brake_drag_coefficient(mach_num, deployment_level)

        # F_drag = 1/2 * C_d * rho * A  * V^2, C_d is a function of deployment level, so it's not included here
        calculated_drag_force: float = 0.5 * drag_coefficient * air_density * air_brake_area * curr_vel ** 2

        return calculated_drag_force - drag_force

    max_force: float = 0.5 * find_air_brake_drag_coefficient(mach_num, 1) * air_density * air_brake_area * curr_vel ** 2

    if max_force < drag_force:
        return 1
    elif drag_force < 0:
        return 0

    # Use root finding to determine the deployment level
    deployment_level: float = root_scalar(
        drag_force_error,
        bracket=[0, 1],  # Deployment levels range from 0 (closed) to 1 (fully deployed)
        method='bisect',
        xtol=1e-3
    ).root

    return deployment_level


def find_air_brake_drag_coefficient(mach_number: float, deployment_level: float) -> float:
    """
    Determines the air brake drag coefficient based on Mach number and deployment level.

    Args:
        mach_number (float): The Mach number of the rocket
        deployment_level (float): The deployment level of the air brake (0 to 1)

    Returns:
        drag_coefficient (float): The interpolated drag coefficient.
    """
    if deployment_level == 0:
        return 0

    deployment_levels: list[float] = [0, 0.25, 0.5, 0.75, 1]
    mach_numbers: list[float] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    drag_coeffs: list[list[float]] = [
        [0, 0, 0, 0, 0],
        [0, 0.444486249, 0.507960781, 0.544841711, 0.594581464],
        [0, 0.417785388, 0.483700124, 0.525607668, 0.571021268],
        [0, 0.406705275, 0.479774019, 0.52385219, 0.568730609],
        [0, 0.409961944, 0.479198954, 0.522773668, 0.570856452],
        [0, 0.409147947, 0.483454287, 0.528634803, 0.578541851],
        [0, 0.411546029, 0.484154172, 0.537079146, 0.587973526],
        [0, 0.407979926, 0.491932723, 0.543695454, 0.598682104],
        [0, 0.413460626, 0.503072477, 0.55774353, 0.611637741],
        [0, 0.408190137, 0.510126955, 0.572806033, 0.627439726],
        [0, 0.408190137, 0.510126955, 0.572806033, 0.627439726]
    ]

    interpolator = RegularGridInterpolator((mach_numbers, deployment_levels), drag_coeffs)
    drag_coefficient: float = interpolator([[mach_number, deployment_level]])[0]

    return drag_coefficient


def initialize_flight_environment(latitude: float, longitude: float, elevation: float, day_delta: int):
    """
    Initializes the flight environment for the simulation given the latitude, longitude, and elevation of the launch site.

    Args:
        latitude (float): Launch site latitude in degrees
        longitude (float): Launch site longitude in degrees
        elevation (float): Elevation of launch site above sea level
        day_delta (int): The number of days following today of the scheduled flight
    """
    env = Environment(latitude=latitude, longitude=longitude, elevation=elevation)
    launch_day = datetime.date.today() + datetime.timedelta(days=day_delta)

    env.set_date((launch_day.year, launch_day.month, launch_day.day, 18))  # Hour given in UTC time
    env.set_atmospheric_model(type="Forecast", file="GFS")

    return env


def initialize_base_rocket():
    # -------------------- Specifications for M2400 motor --------------------
    AeroTech_M2400T = SolidMotor(
        thrust_source="AeroTech_M2400T.eng",
        dry_mass=2.758,  # mass of motor without propellant
        dry_inertia=(0.0819, 0.0819, 3.31e-3),  # I_x = I_y = 1/12 * mL^2 ; I_z = 1/2 * mR^2
        nozzle_radius=49 / 1000,  # mm to m conversion, took this as the diameter of the motor
        grain_number=4,
        grain_density=1815,
        grain_outer_radius=49 / 1000,
        grain_initial_inner_radius=37 / 1000,
        grain_initial_height=0.12,
        grain_separation=10 / 1000,
        grains_center_of_mass_position=0.3,
        center_of_dry_mass_position=0.28,
        nozzle_position=0,
        burn_time=3.28,  # seconds
        throat_radius=33 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    some_rocket_name = Rocket(
        radius=0.0762,  # radius in m
        mass=16.4,  # dry mass in kg
        # All mass moments of inertia are using the dry mass of the rocket
        # mass moment of inertia about z, I_z = mr^2 (of a hoop through the center)
        # mass moment of inertia about x and y, I_x = I_y = mL^2/12 (of long rod about CG)
        inertia=(9.17, 9.17, 0.381),  # mass moments of inertia about x, y, and z axes
        power_off_drag="powerOffDragCurveM2400.csv",
        power_on_drag="powerOnDragCurveM2400.csv",
        center_of_mass_without_motor=1.41,
        coordinate_system_orientation="nose_to_tail"
    )

    # All the positions have to be relative to the distance of the nose of the rocket
    some_rocket_name.add_motor(AeroTech_M2400T, position=2.565)

    some_rocket_name.set_rail_buttons(upper_button_position=1.8, lower_button_position=2.5)

    # Add aerodynamic components
    some_rocket_name.add_nose(length=0.796, kind="ogive", position=0)

    some_rocket_name.add_trapezoidal_fins(
        n=3,
        root_chord=0.356,
        tip_chord=0.102,
        span=0.203,
        position=2.218,
    )

    main_parachute = some_rocket_name.add_parachute(name="main", cd_s=2.2, trigger=369)
    drogue_parachute = some_rocket_name.add_parachute(name="drogue", cd_s=2.2, trigger='apogee')

    return some_rocket_name


if __name__ == "__main__":
    main()
