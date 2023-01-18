import os
import random

# ------ this for stopping the while loop with Ctrl+C (KeyboardInterrupt) this has to be at the top of the file!!--------
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

import numpy as np
import Onrobot
import URBasic
import time
from matplotlib import pyplot as plt
from min_jerk_planner import PathPlan
import angle_transformation as at
from Control import Control

from run_robot_functions import run_robot_BASE
from run_robot import run_robot
from run_robot_with_spiral import run_robot_with_spiral

# from func_KeyInteruppt import func_robot_test


host = '192.168.1.103'
robotModle = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=host, robotModel=robotModle)
FT_sensor = Onrobot.FT_sensor()
# robot.reset_registers()
# Simulation values
# ---------------- -free space---------------------
# init_pose = np.array([-0.2554, -0.3408, 0.2068, 0.1136, -3.1317, -0.0571])
# goal_pose = np.array([-0.0575, -0.4350, 0.2497, 0.3463, 2.9853, -0.2836])

# # -----------------------above hole: d_peg=8.5mm / d_hole=10mm ------------------
# init_pose = np.array([0.0859, -0.4810,  0.0837,   -0.0729,  3.1407, -0.0001])
# goal_pose = np.array([0.0859, -0.4810,  0.0730,   -0.0729,  3.1407, -0.0001])
# # -----------------------above hole: d_peg=4.2mm / d_hole=4.6mm ------------------
init_pose = np.array([-0.1379, -0.5095,  0.0765,   -0.0728,  3.1407, -0.0001])
goal_pose = init_pose.copy()
goal_pose[2] = 0.0663


# init_pose = np.array([0.1012, -0.4877, 0.0869, -0.0730, 3.1407, -0.0000])
# goal_pose = np.array([0.1012, -0.4877, 0.0731, -0.0730, 3.1407, -0.0000])


plot_graphs = True
use_spiral = True  # TODO: doesnt work yet
use_circle = True
error_type = "fixed"
error_vec = [6, 0.0, 0.0]  # mm
control_dim = 26
use_impedance = False
time_insertion = 10
time_trajectory = 5

# end of simulation values

# Control Loop
num_of_trials = 1
success_counter = 0
t_start = time.time()
for trial in range(1, num_of_trials + 1):
    # if trail == num_of_trails:
    #     is_plot = True
    pos_error = [0, 0, 0]

    if error_type == 'none':
        pos_error = np.array([0.0, 0.0, 0.0])
    if error_type == 'fixed':
        pos_error = np.array(error_vec) / 1000  # fixed error
    if error_type == 'ring':
        r_low = 0.6 / 1000
        r_high = 0.8 / 1000
        r = random.uniform(r_low, r_high)
        theta = random.uniform(0, 2 * np.pi)
        x_error = r * np.cos(theta)
        y_error = r * np.sin(theta)

        pos_error[:2] = np.array([x_error, y_error])
    pose_error = np.array([pos_error[0], pos_error[1], pos_error[2], 0.0, 0.0, 0.0])
    # start pose: above the hole
    start_pose = init_pose + pose_error
    # desired_pose: in the hole
    desired_pose = goal_pose + pose_error

    if use_spiral:
        success_flag = run_robot_with_spiral(robot=robot, start_pose=start_pose,
                                             pose_desired=desired_pose, pose_error=pose_error,
                                             control_dim=control_dim, use_impedance=use_impedance,
                                             plot_graphs=plot_graphs, circle=use_circle, sensor_class = FT_sensor, time_insertion=time_insertion,
                                 time_trajectory=time_trajectory)
    else:
        success_flag = run_robot(robot=robot, start_pose=start_pose,
                                 pose_desired=desired_pose, pose_error=pose_error,
                                 control_dim=control_dim, use_impedance=use_impedance, plot_graphs=plot_graphs, sensor_class = FT_sensor, time_insertion=time_insertion,
                                 time_trajectory=time_trajectory)

    if success_flag == 'error' or success_flag == 'interrupt':
        print('\n!!!There was error or Keyboard Interruption during simulation!!!\n')
        print('breaking the loop.')
        break

    success_counter += success_flag
    print(f'\ntrail {trial} for pos error = {np.round(pos_error, 4)}:')
    print(f'\nsuccess/trail = {success_counter}/{trial}. success rate = {success_counter / trial}')

robot.close()
print(f'\n * * * * * * * * * Experiment is Done * * * * * * * * * *\n')
print(f'success rate: {success_counter}/{trial} = {round(100 * success_counter / trial, 2)}%')
print(
    f'\ntime took for the experiment = {time.time() - t_start} seconds, which is {round((time.time() - t_start) / 60, 1)} minutes. succeed {success_counter} from {num_of_trials} trials.')
