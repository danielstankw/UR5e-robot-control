import os
import random

#------ this for stopping the while loop with Ctrl+C (KeyboardInterrupt) this has to be at the top of the file!!--------
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
#from func_KeyInteruppt import func_robot_test


host = '192.168.1.103'
robotModle = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=host, robotModel=robotModle)
FT_sensor = Onrobot.FT_sensor()


# Simulation values
init_pose = np.array([-0.2554, -0.3408,  0.2068,    0.1136, -3.1317, -0.0571])
goal_pose = np.array([-0.0575, -0.4350,  0.2497,    0.3463,  2.9853, -0.2836])
# goal_pose = np.array([-1.2360, -1.5677, -1.4024, -1.7797, 1.5725, 0.2638])

plot_graphs = False
render = False
error_type = "none"
error_vec = [2.0, 0.0, 0.0]

# end of simulation values

pos_error = [0, 0, 0]

if error_type == 'none':
    pos_error = np.array([0.0, 0.0, 0.0])
if error_type == 'fixed':
    pos_error = np.array(error_vec)/1000  # fixed error
if error_type == 'ring':
    r_low = 0.4 / 1000
    r_high = 0.8 / 1000
    r = random.uniform(r_low, r_high)
    theta = random.uniform(0, 2 * np.pi)
    x_error = r * np.cos(theta)
    y_error = r * np.sin(theta)

    pos_error[:2] = np.array([x_error, y_error])

# Control Loop
num_of_trails = 1
success_counter = 0
t_start = time.time()
for trail in range(1, num_of_trails+1):
    if trail == num_of_trails:
        is_plot = True

    pose_error = np.array([pos_error[0], pos_error[1], pos_error[2], 0.0, 0.0, 0.0])
    # start pose: above the hole
    start_pose = init_pose + pose_error
    # desired_pose: in the hole
    desired_pose = goal_pose + pose_error

    success_flag = run_robot(robot=robot, start_pose=start_pose, pose_desired=desired_pose, pose_error=pose_error)
    # success_flag = run_robot(robot=robot, FT_sensor=FT_sensor, start_pose=start_pose, desired_pose=desired_pose,
    #                               pose_error=pose_error, which_controller=which_controller, which_param=which_param,
    #                               is_plot=is_plot)

    # #For testing Key Interrupt:
    # ans = func_robot_test()
    # success_flag = 1

    if success_flag == 'error' or success_flag=='interrupt':
        print('\n!!!There was error or Keyboard Interruption during simulation!!!\n')
        print('breaking the loop.')
        break

    success_counter += success_flag
    print(f'\ntrail {trail} for pos error = {np.round(pos_error, 4)}:')
    print(f'\nsuccess/trail = {success_counter}/{trail}. success rate = {success_counter/trail}')

robot.close()
is_plot = False
print(f'\n * * * * * * * * * Experiment is Done * * * * * * * * * *\n')
print(f'success rate: {success_counter}/{trail} = {round(100*success_counter/trail,2)}%')
print(f'\ntime took for the experiment = {time.time() - t_start} seconds, which is {round((time.time() - t_start) / 60, 1)} minutes. succeed {success_counter} from {num_of_trails} trials.')

