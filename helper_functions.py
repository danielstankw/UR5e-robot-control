import numpy as np
import numpy.linalg as LA
import angle_transformation as at

ERROR_TOP = 0.8 / 1000


def label_check(peg_xy, hole_xy):
    """
    Function responsible for labeling classes
    :param peg_xy: : x and y coordinates of the peg
    :param hole_xy: x and y coordinates of the hole
    :return: True: if there is sufficient overlap/ False: no sufficient overlap
    """
    hole_x = hole_xy[0]
    hole_y = hole_xy[1]
    peg_x = peg_xy[0]
    peg_y = peg_xy[1]

    radial_distance = np.sqrt((peg_x - hole_x) ** 2 + (peg_y - hole_y) ** 2)
    # print(radial_distance)
    if radial_distance < ERROR_TOP:
        print(radial_distance)
        print('Overlap')
        return True
    else:
        return False


def in_hole_stop(peg_xy, hole_xy):
    """
    Function responsible for early termination when performing spiral search, in case of
    peg getting close enough to get inserted in the hole
    :param peg_xy: : x and y coordinates of the peg
    :param hole_xy: x and y coordinates of the hole
    :return: True/ False flag whether the simulation should terminate
    """
    hole_x = hole_xy[0]
    hole_y = hole_xy[1]
    peg_x = peg_xy[0]
    peg_y = peg_xy[1]

    radial_distance = np.sqrt((peg_x - hole_x) ** 2 + (peg_y - hole_y) ** 2)
    # print(radial_distance)
    if radial_distance <= 0.4 / 1000:
        print('Inside Hole - Stopping!')
        return True
    else:
        return False


def next_spiral(theta_current, dt):
    """
    Function responsible for generating spiral trajectory. According to the research to ensure insertion
    pitch (p) and clearance (d) have to satisfy: p<=2d
    :param theta_current: current value of angle [rad]
    :param dt: time step of the simulation [sec]
    :return: theta_next, radius_next, x_next, y_next
    """
    # according to the article to assure successful insertion: p<=2d
    # where p is distance between consequent rings and d is clearance in centralized peg
    # v = 0.0009230 / 2   # total velocity (linear and angular)
    v = 0.0015
    # p = 0.0006  # distance between the consecutive rings
    p = 0.0012  # distance between the consecutive rings


    theta_dot_current = (2 * np.pi * v) / (p * np.sqrt(1 + theta_current ** 2))
    # todo: change +/- depending on the desired direction
    theta_next = theta_current + theta_dot_current * dt

    radius_next = (p / (2 * np.pi)) * theta_next

    x_next = radius_next * np.cos(theta_next)
    y_next = radius_next * np.sin(theta_next)
    return theta_next, radius_next, x_next, y_next


def next_circle(theta_current, dt):
    # overlap=error-2*radius_of_circle
    time_per_circle = 4  # 6 #4
    # 0.0025, 0.0026, 0.0027 with perturbation
    # 0.0028 always enters
    radius_of_circle = 0.002675  # 3 / 1000#0
    # increase in angle per dt
    d_theta = (2 * np.pi * dt) / time_per_circle
    theta_next = theta_current + d_theta

    x_next = radius_of_circle * np.cos(theta_next) - radius_of_circle
    y_next = radius_of_circle * np.sin(theta_next)

    return theta_next, radius_of_circle, x_next, y_next


def circular_wrench_limiter(wrench_cmd):
    # Limit the wrench in TOOL frame in a circular way. meaning that Fxy and Mxy consider as a vector with limited
    # radius
    wrench_safety_limits = dict(Fxy=20, Fz=15, Mxy=4, Mz=3)
    limited_wrench = wrench_cmd.copy()
    Fxy, Fz, Mxy, Mz = wrench_cmd[:2], wrench_cmd[2], wrench_cmd[3:5], wrench_cmd[5]
    Fxy_size, Mxy_size = LA.norm(Fxy), LA.norm(Mxy)

    if Fxy_size > wrench_safety_limits['Fxy']:
        # print('clipping_1')
        Fxy_direction = Fxy / Fxy_size
        limited_wrench[:2] = wrench_safety_limits['Fxy'] * Fxy_direction
    if Fz < -wrench_safety_limits['Fz'] or Fz > wrench_safety_limits['Fz']:
        # print('clipping_2')
        limited_wrench[2] = np.sign(Fz) * wrench_safety_limits['Fz']
    if Mxy_size > wrench_safety_limits['Mxy']:
        # print('clipping_3')
        Mxy_direction = Mxy / Mxy_size
        limited_wrench[3:5] = wrench_safety_limits['Mxy'] * Mxy_direction
    if Mz < -wrench_safety_limits['Mz'] or Mz > wrench_safety_limits['Mz']:
        # print('clipping_4')
        limited_wrench[5] = np.sign(Mz) * wrench_safety_limits['Mz']

    if np.inf in wrench_cmd:
        print('\n!!!! inf wrench !!!!\n')

    return limited_wrench


def external_calibrate(external_sensor_tool, current_pose):
    external_sensor_force = at.Tool2Base_vec(current_pose[3:], external_sensor_tool[:3])
    external_sensor_moment = at.Tool2Base_vec(current_pose[3:], external_sensor_tool[3:])
    external_sensor_base = np.append(external_sensor_force, external_sensor_moment)
    return external_sensor_base
