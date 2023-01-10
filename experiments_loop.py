import os
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
#from func_KeyInteruppt import func_robot_test


def random_error(max_rad, min_rad):
    radius = np.random.random()*(max_rad-min_rad) + min_rad
    #radius = max_rad
    max_theta = 2*np.pi
    min_theta = 0
    theta = np.random.random()*(max_theta-min_theta) + min_theta
    pos_error = np.array([radius*np.cos(theta), radius*np.sin(theta), 0.0])
    return pos_error

host = '192.168.1.103'
robotModle = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=host, robotModel=robotModle)
FT_sensor = Onrobot.FT_sensor()

which_controller= 'imp'  #'PD' #'imp'
which_param= 'learned_params'  #'20mm_circular_peg'
is_plot = False  # Keep it false. The program will plot after the final trial


# * *   *   *   *   *   NEW ROBOT   *   *   *   *   *   *   *   *

# # For the new robot, Black connector, 3.5mm Sofit wire:
# init_pose = np.array([-0.527, -0.4031,  0.1168,   -0.0305, -3.1414,  0.0000])
# goal_pose = np.array([-0.527, -0.4031,  0.100,   -0.0305, -3.1414,  0.0000])

# # For the new robot, Driver connector, 0.8mm white wire:
# init_pose = np.array([-0.4595, -0.5536,  0.060,   -0.0306, -3.1414, -0.0000])
# goal_pose = np.array([-0.4595, -0.5536,  0.048,   -0.0306, -3.1414, -0.0000])

# For the new robot, 10mm Hole, 8mm Peg (Elad installation):
init_pose = np.array([-0.4886, -0.3985,  0.0884,   -0.0172, -3.1390,  0.0000])
goal_pose = np.array([-0.4886, -0.3985,  0.078,   -0.0172, -3.1390,  0.0000])

# # For the new robot 8.8mm Tube (Elad installation):
# init_pose = np.array([-0.5659, -0.4977,  0.1945,   -0.0173, -3.1390, -0.0000])
# goal_pose = np.array([-0.5659, -0.4977,  0.1845,   -0.0173, -3.1390, -0.0000])

# # For the new robot USB (Elad installation):
# init_pose = np.array([-0.4309, -0.5669,  0.0915,   -0.0523, -3.1385,  0.0000])
# goal_pose = np.array([-0.4309, -0.5669,  0.0802,   -0.0523, -3.1385,  0.0000])



# * *   *   *   *   *   OLD ROBOT   *   *   *   *   *   *   *   *

# # For 8.2mm peg, 10mm hole (EE no fingers installaion (in Elad folder)):
# init_pose = np.array([-0.2918, -0.581,  0.0892,   -0.0085,  3.1406,  0.0043])
# goal_pose = np.array([-0.2918, -0.581,  0.072,   -0.0085,  3.1406,  0.0043])
#  # with 180 rotation:
# init_pose = np.array([0.2434, -0.5861,  0.0891,    3.1327,  0.0450,  0.0069])
# goal_pose = np.array([0.2434, -0.5861,  0.0804,    3.1327,  0.0450,  0.0069])
# new position (closer to the door):
# init_pose = np.array([ 0.2398, -0.5834,  0.0895,    0.0441, -3.1403, -0.0011])
# goal_pose = np.array([ 0.2398, -0.5834,  0.079,    0.0441, -3.1403, -0.0011])

# # # For 4.5mm PEG, 6mm hole (EE no fingers installaion (in Elad folder)):
# init_pose = np.array([-0.3926, -0.5814,  0.0707,    0.0046, -3.1370,  0.0226])
# goal_pose = np.array([-0.3926, -0.5814,  0.062,    0.0046, -3.1370,  0.0226])
#   with 180 rotation
# init_pose = np.array([0.1423, -0.5851,  0.0747,    3.1327,  0.0450,  0.0069])
# goal_pose = np.array([0.1423, -0.5851,  0.0676,    3.1327,  0.0450,  0.0069])

# # For wire with 3.5mm Sofit, with 5-6mm black conecctor (EE no fingers installaion (in Elad folder)):
# init_pose = np.array([-0.0007, -0.6645,  0.1186,    0.0047, -3.1370,  0.0226])
# goal_pose = np.array([-0.0007, -0.6645,  0.103,    0.0047, -3.1370,  0.0226])

# # For PEG with 4.2mm (EE no fingers installaion (in Elad folder)):
# # No rotation ("On robot" symbol is towards the PC):
# init_pose = np.array([-0.5167, -0.6088,  0.0748,    0.0441, -3.1404, -0.0129])
# goal_pose = np.array([-0.5167, -0.6088,  0.0625,    0.0441, -3.1404, -0.0129])
# # with 90deg rotation around Z:
# init_pose = np.array([-0.5185, -0.6125,  0.0748,   -2.2277,  2.2108,  0.0223])
# goal_pose = np.array([-0.5185, -0.6125,  0.0625,   -2.2277,  2.2108,  0.0223])
# # No rotation ("On robot" symbol is towards the PC) different location (near the door):
# init_pose = np.array([ 0.0162, -0.6067,  0.0741,    0.0441, -3.1404, -0.0128])
# goal_pose = np.array([ 0.0162, -0.6067,  0.0625,    0.0441, -3.1404, -0.0128])
# # with 90deg rotation around Z, ("On robot" symbol is towards the PC) different location (near the door):
# init_pose = np.array([0.0143, -0.6102,  0.0738,   -2.2277,  2.2107,  0.0222])
# goal_pose = np.array([0.0143, -0.6102,  0.0625,   -2.2277,  2.2107,  0.0222])
# with 180deg rotation around Z, ("On robot" symbol is towards the wall) different location (near the door):
# init_pose = np.array([0.0177, -0.6116,  0.0742,    3.1328,  0.0449,  0.0069])
# goal_pose = np.array([0.0177, -0.6116,  0.0625,    3.1328,  0.0449,  0.0069])


# Pos error settings:
min_rad = 0.0025 #0.002  # 0.0015 #0.0025  #0.0022
max_rad = 0.0015 #0.002  #0.0025  #0.0015 #0.0025  #0.0022 #25 #0.0035

# Experiments Loop:
num_of_trails = 1 #20 #20 #25  #100
success_counter = 0
t_before = time.time()
for trail in range(1,num_of_trails+1):
    if trail == num_of_trails:
        is_plot=True
    pos_error = random_error(max_rad, min_rad)
    pose_error = np.array([pos_error[0], pos_error[1], pos_error[2], 0.0, 0.0, 0.0])
    start_pose = init_pose + pose_error
    desired_pose = goal_pose + pose_error

    success_flag = run_robot_BASE(robot=robot, FT_sensor=FT_sensor, start_pose=start_pose, desired_pose=desired_pose,
                                  pose_error=pose_error, which_controller=which_controller, which_param=which_param,
                                  is_plot=is_plot)

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
print(f'\ntime took for the experiment = {time.time() - t_before} seconds, which is {round( (time.time() - t_before)/60, 1)} minutes. succeed {success_counter} from {num_of_trails} trials.')