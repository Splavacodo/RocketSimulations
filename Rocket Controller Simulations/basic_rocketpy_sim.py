from rocketpy import Environment, SolidMotor, Rocket, Flight, plots
import matplotlib.pyplot as plt
import datetime

def main():
    env = initialize_flight_environment()
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

    # Only reason the controller function is within main is because it needs a reference
    # to the environment
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

        # Get previous state from state_history
        previous_state = state_history[-1]
        previous_vz = previous_state[5]

        # If we wanted to we could get the returned values from observed_variables:
        # returned_time, deployment_level, drag_coefficient = observed_variables[-1]

        # Check if the rocket has reached burnout
        if time < 4.68:  # mojito.motor.burn_out_time
            new_deployment_level = 0
        # If burn out is finished, fully deploy the air brakes
        else:
            new_deployment_level = 1

        # ------------ Work in Progress ----------------
        # The controller in the time domain can be expressed as G_c(t) = F(t)/e(t)
        # The force of the air brake will be equal to F = K_p * e(t) + K_d * e_dot(t)

        # At one point in time e(t) = Desired height - current height and e_dot(t) = e(t)/dt
        # There will then be another function that takes in the force and determines the best
        # air brake deployment level to match that force
        # ----------------------------------------------

        air_brakes.deployment_level = new_deployment_level

        # Return variables of interest to be saved in the observed_variables list
        return (
            time,
            air_brakes.deployment_level,
            air_brakes.drag_coefficient(air_brakes.deployment_level, mach_number),
        )

    air_brakes = mojito.add_air_brakes(
        drag_coefficient_curve="air_brake_deployment_data.csv",
        controller_function=controller_function,
        sampling_rate=40,
        reference_area=None,
        clamp=True,
        initial_observed_variables=[0, 0, 0],
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
        terminate_on_apogee=False
    )

    time_list, deployment_level_list, drag_coefficient_list = [], [], []

    obs_vars = controlled_test_flight.get_controller_observed_variables()

    for time, deployment_level, drag_coefficient in obs_vars:
        time_list.append(time)
        deployment_level_list.append(deployment_level)
        drag_coefficient_list.append(drag_coefficient)

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

    # Comparison of flights
    # plots.CompareFlights([base_test_flight, controlled_test_flight]).trajectories_3d()
    # base_test_flight.altitude()

    # controlled_test_flight.aerodynamic_drag()
    controlled_test_flight.prints.apogee_conditions()
    controlled_test_flight.altitude()

def initialize_flight_environment():
    # Spaceport Location
    # env = Environment(latitude=32.990254, longitude=-106.974998, elevation=735)

    # Las Vegas Launch Site Environment
    env = Environment(latitude=35.78, longitude=-115.25, elevation=735)
    # 35.78 latitude
    # -115.25 longitude
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