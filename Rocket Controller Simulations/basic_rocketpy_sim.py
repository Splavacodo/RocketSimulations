from rocketpy import Environment, SolidMotor, Rocket, Flight
import matplotlib.pyplot as plt
import datetime

def main():
    # Spaceport (New Mexico) latitude/longitude/elevation: latitude=32.990254, longitude=-106.974998, elevation=735
    # Jean Lake (Nevada) latitude/longitude/elevation: latitude=35.78, longitude=-115.25, elevation=847
    env = initialize_flight_environment(latitude=35.78, longitude=-115.25, elevation=847)
    mojito = initialize_base_rocket()

    # mojito.draw()

    base_test_flight = Flight(
        rocket=mojito,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False
    )

    desired_height = 1097  # height in meters

    # Gain parameters determined from root locus & step responses
    K_p = 0.64
    K_d = 8

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
        mach_number = free_stream_speed / env.speed_of_sound(altitude_ASL)
        air_density = env.density(altitude_ASL)

        prev_time, _, _, prev_height_err = observed_variables[-1]
        height_err = desired_height - altitude_AGL

        # Check if the rocket has reached burnout
        if time < 4.68:  # mojito.motor.burn_out_time
            new_deployment_level = 0
        # If burn out is finished, apply PD control
        else:
            # ------------ Feedback Control ----------------
            # The controller in the time domain can be expressed as G_c(t) = F(t)/e(t)
            # The force of the air brake will be equal to F(t) = K_p * e(t) + K_d * e_dot(t)

            # At one point in time e(t) = Desired height - current height and e_dot(t) = e(t) - e_prev(t)/dt
            # There will then be another function that takes in the force and determines the best
            # air brake deployment level to match that force

            # To calculate the derivative of the error, the error should be stored in observed variables to reference
            # the previous error from the last iteration
            # ----------------------------------------------
            e_dot = (height_err - prev_height_err) / (time - prev_time)
            controller_output = K_p * height_err + K_d * e_dot

            new_deployment_level = find_deployment_level(mach_num=mach_number, curr_vel=vz, drag_force=controller_output, air_density=air_density)

        air_brakes.deployment_level = new_deployment_level

        # Return variables of interest to be saved in the observed_variables list
        return (
            time,
            air_brakes.deployment_level,
            air_brakes.drag_coefficient(air_brakes.deployment_level, mach_number),
            height_err
        )

    air_brakes = mojito.add_air_brakes(
        drag_coefficient_curve="air_brake_deployment_data.csv",
        controller_function=controller_function,
        sampling_rate=40,
        reference_area=None,
        clamp=True,
        initial_observed_variables=[0, 0, 0, 0],
        override_rocket_drag=False,
        name="Air Brakes"
    )

    # air_brakes.all_info()

    controlled_test_flight = Flight(
        rocket=mojito,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        time_overshoot=False,
        terminate_on_apogee=True
    )

    time_list, deployment_level_list, drag_coefficient_list, height_err_list = [], [], [], []

    obs_vars = controlled_test_flight.get_controller_observed_variables()

    for time, deployment_level, drag_coefficient, height_err in obs_vars:
        time_list.append(time)
        deployment_level_list.append(deployment_level)
        drag_coefficient_list.append(drag_coefficient)
        height_err_list.append(height_err)

    # Plot deployment level by time
    # plt.plot(time_list, deployment_level_list)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Deployment Level")
    # plt.title("Deployment Level by Time")
    # plt.grid()
    # plt.show()

    # Plot drag coefficient by time
    # plt.plot(time_list, drag_coefficient_list)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Drag Coefficient")
    # plt.title("Drag Coefficient by Time")
    # plt.grid()
    # plt.show()

    # Plot of height err with time
    plt.plot(time_list, height_err_list)
    plt.xlabel("Times (s)")
    plt.ylabel("Height Error [m]")
    plt.title("Height Error by Time")
    plt.grid()
    plt.show()

    # Comparison of flights
    # plots.CompareFlights([base_test_flight, controlled_test_flight]).trajectories_3d()
    # base_test_flight.altitude()
    # base_test_flight.prints.apogee_conditions()

    # controlled_test_flight.aerodynamic_drag()
    # controlled_test_flight.prints.apogee_conditions()
    # controlled_test_flight.altitude()

    # Brakes shave off about 111 m or 365 ft, the goal of the controller could be to reach a height of 1100 meters

def find_deployment_level(mach_num: float, curr_vel: float, drag_force: float, air_density: float) -> float:
    '''
    Determines the deployment level of the air brakes to provide a given drag force and the current speed of the rocket.

    Args:
        mach_num (float): The mach number the rocket is traveling at
        curr_vel (float): The current velocity of the rocket in meters/sec
        drag_force (float): The drag force provided by the controller to reach the desired height
        air_density (float): The air density of air given the current height of the rocket in kg/m^3

    Returns:
        deployment_level (float): The deployment level of the air brake to provide the given drag force
    '''
    # F_drag = 1/2 * C_d * rho * A * V^2
    # A = (2 * F_drag) / rho * C_d * V^2

    return 1

def find_air_brake_drag_coefficient(mach_number: float) -> float:
    '''
    Determines the air brake drag coefficient
    '''
    # Each index in the lists corresponds to a deployment level
    # deployment level: [0, 0.25, 0.5, 0.75, 1]
    drag_coeffs = {
        0.1: [0.484455781, 0.370481556, 0.424189869, 0.460481711, 0.501541095],
        0.2: [0.49451265, 0.327569465, 0.409090087, 0.455112722, 0.49773203],
        0.3: [0.501742369, 0.31367034, 0.406985797, 0.456193802, 0.499135351],
        0.4: [0.504890439, 0.317494903, 0.409256946, 0.461219701, 0.503252629],
        0.5: [0.506025828, 0.323488649, 0.410878827, 0.464872337, 0.508892964],
        0.6: [0.506592807, 0.320294485, 0.413512723, 0.469369044, 0.514095489],
        0.7: [0.507668973, 0.30957922, 0.421336892, 0.471799453, 0.523979987],
        0.8: [0.51117623, 0.308752715, 0.424021613, 0.482733443, 0.533868504],
        0.9: [0.521776412, 0.303223303, 0.427707593, 0.486100686, 0.547969309]
    }


def initialize_flight_environment(latitude: float, longitude: float, elevation: float):
    '''
    Initializes the flight environment for the simulation given the latitude, longitude, and elevation of the launch site.

    Args:
        latitude (float): Launch site latitude in degrees
        longitude (float): Launch site longitude in degrees
        elevation (float): Elevation of launch site above sea level
    '''
    # Spaceport Location
    # env = Environment(latitude=32.990254, longitude=-106.974998, elevation=735)

    # Las Vegas Launch Site Environment
    env = Environment(latitude=latitude, longitude=-longitude, elevation=elevation)
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 21))  # Hour given in UTC time
    env.set_atmospheric_model(type="Forecast", file="GFS")

    return env

def initialize_base_rocket():
    AeroTech_K400C = SolidMotor(
        thrust_source="AeroTech_K400C.eng",
        dry_mass=0.701,  # mass of motor without propellant
        dry_inertia=(7.53e-3, 7.53e-3, 2.55e-4),  # I_x = I_y = 1/12 * mL^2 ; I_z = 1/2 * mR^2
        nozzle_radius=27 / 1000,  # mm to m conversion, took this as the diameter of the motor
        grain_number=3,
        grain_density=1815,
        grain_outer_radius=27 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=0.0718,
        grain_separation=5 / 1000,
        grains_center_of_mass_position=0.18,
        center_of_dry_mass_position=0.177,
        nozzle_position=0,
        burn_time=3.2,  # seconds
        throat_radius=11 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    mojito = Rocket(
        radius=0.0528,  # radius in m
        mass=5.24,  # dry mass in kg
        # All mass moments of inertia are using the dry mass of the rocket
        # mass moment of inertia about z, I_z = mr^2 (of a hoop through the center)
        # mass moment of inertia about x and y, I_x = I_y = mL^2/12 (of long rod about CG)
        inertia=(1.93, 1.93, 0.0147),  # mass moments of inertia about x, y, and z axes
        power_off_drag="powerOffDragCurveK400.csv",
        power_on_drag="powerOnDragCurveK400.csv",
        center_of_mass_without_motor=1.21,
        coordinate_system_orientation="nose_to_tail"
    )

    # All the positions have to be relative to the distance of the center of mass without a motor
    mojito.add_motor(AeroTech_K400C, position=2.105)
    mojito.set_rail_buttons(upper_button_position=1.473, lower_button_position=2.06)

    # Add aerodynamic components
    mojito.add_nose(length=0.590, kind="ogive", position=0)

    mojito.add_trapezoidal_fins(
        n=3,
        root_chord=0.2,
        tip_chord=0.01,
        span=0.081,
        position=1.887,
    )

    mojito.add_parachute(name="main", cd_s=2.2, trigger=369)

    return mojito

if __name__ == "__main__":
    main()