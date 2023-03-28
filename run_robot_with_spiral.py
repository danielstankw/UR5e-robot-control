import numpy as np
import numpy.linalg as LA
import Onrobot
import URBasic
import os
import pandas as pd
import time
from copy import deepcopy
from minimum_jerk_planner import PathPlan
import angle_transformation as at
from matplotlib import pyplot as plt
from Controller import Controller
from helper_functions import label_check, in_hole_stop, next_spiral, next_circle, circular_wrench_limiter, external_calibrate


ERROR_TOP = 0.8 / 1000
ERROR_MIN = 0.4 / 1000


# import traceback
# import os
# # ------ this for stopping the while loop with Ctrl+C (KeyboardInterrupt) this has to be at the top of the file!!--------
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

def run_robot_with_spiral(robot, start_pose, pose_desired, pose_error, control_dim, use_impedance, plot_graphs, circle,
                          sensor_class, time_insertion, time_trajectory, episode):
    # Check if the UR controller is powered on and ready to run.
    real_goal_pose = pose_desired - pose_error
    if use_impedance:
        control = Controller(control_dim=control_dim)
    time.sleep(0.1)
    robot.reset_error()
    # robot.set_payload_mass(m=1.12)
    # robot.set_payload_cog(CoG=(0.005, 0.00, 0.084))
    save_data = True

    data_collection = True
    if data_collection:
        save_data = True

    #  move above the hole
    robot.movel(start_pose)
    robot.movel(start_pose)
    time.sleep(0.5)  # Wait for the robot/measurements to be stable before starting the insertion
    robot.zero_ftsensor()
    robot.force_mode_set_damping(0)
    robot.force_mode_set_gain_scaling(1)  # or decrease P[2] in free space and increase here to 1.5

    pose_init = np.array(robot.get_actual_tcp_pose(wait=True))
    # pose_init = deepcopy(start_pose)
    vel_init = np.array(robot.get_actual_tcp_speed(wait=False))

    time_for_simulation = time_trajectory + time_insertion

    planner = PathPlan(pose_init, pose_desired, time_trajectory)

    # orientation given as rotVec from 1 to 2 im (0) frame of ref
    # orientation as minimizing of the magnitude
    [position_ref, orientation_ref, lin_vel_ref, ang_vel_ref] = planner.trajectory_planning(0)
    desired_pos = np.concatenate((position_ref, orientation_ref, lin_vel_ref, ang_vel_ref), axis=0)

    # initialize impedance parameters
    pose_mod = np.zeros(6)
    vel_mod = np.zeros(6)
    # ----------- Control Settings -------------------
    # Control params for the free space:
    kp = np.array([4500.0, 4500.0, 2250.0, 50.0, 50.0, 50.0])
    # kp = np.array([4500.0, 4500.0, 5.0, 50.0, 50.0, 50.0]) # change internal gain
    kd = 2 * 7 * np.sqrt(kp)

    # ------------ Forces initialization ------------
    internal_sensor_bias = np.copy(robot.get_tcp_force(wait=True))
    external_sensor_bias_tool = np.copy(sensor_class.force_moment_feedback())
    external_sensor_bias_base = external_calibrate(external_sensor_bias_tool, pose_init)

    print(f'internal sensor reading (base) = {internal_sensor_bias}')
    print(f'external sensor reading (tool) = {external_sensor_bias_tool}')
    print(f'external sensor reading (base) = {external_sensor_bias_base}')

    # plotting
    if True:
        time_vec = []
        # robot measurements
        ee_pos_x_vec, ee_pos_y_vec, ee_pos_z_vec = [], [], []
        ee_vel_x_vec, ee_vel_y_vec, ee_vel_z_vec = [], [], []
        ee_ori_x_vec, ee_ori_y_vec, ee_ori_z_vec = [], [], []
        ee_ori_vel_x_vec, ee_ori_vel_y_vec, ee_ori_vel_z_vec = [], [], []
        # minimum jerk
        pos_min_jerk_x, pos_min_jerk_y, pos_min_jerk_z = [], [], []
        vel_min_jerk_x, vel_min_jerk_y, vel_min_jerk_z = [], [], []
        ori_min_jerk_x, ori_min_jerk_y, ori_min_jerk_z = [], [], []
        ori_vel_min_jerk_x, ori_vel_min_jerk_y, ori_vel_min_jerk_z = [], [], []
        # wrench - based on PD
        applied_wrench_fx, applied_wrench_fy, applied_wrench_fz = [], [], []
        applied_wrench_mx, applied_wrench_my, applied_wrench_mz = [], [], []
        # sensor readings
        sensor_fx, sensor_fy, sensor_fz = [], [], []
        sensor_mx, sensor_my, sensor_mz = [], [], []
        # spiral
        spiral_x, spiral_y = [], []
        robot_spiral_x, robot_spiral_y = [], []
        unclipped_wrench_fx, unclipped_wrench_fy, unclipped_wrench_fz = [], [], []
        unclipped_wrench_mx, unclipped_wrench_my, unclipped_wrench_mz = [], [], []
        # for labeling
        labels = []
        time_labels = []

    try:
        # ------------ Simulation(Loop) setup ----------------
        # Initialize force remote
        robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[0, 0, 0, 0, 0, 0],
                               wrench=(-internal_sensor_bias), f_type=2, limits=[2, 2, 1.5, 1, 1, 1])

        t_init = time.time()
        t_curr = 0
        contact_flag = False
        contact_time = 0
        f0 = np.zeros(6)

        contact_fz_threshold = 1.5  # [N]
        success_flag = False

        deviation_from_goal_z = 0.002  # 0.004  # Deviation from final goal in [m] which break the loop
        deviation_from_goal_xy = 0.005  # 0.0015

        theta_current = 0
        x_spiral_next = 0
        y_spiral_next = 0
        #   *   *   *   *   *   *   *   *   = = = = = = = Operation Loop = = = = = = = =    *   *   *   *   *   *   *
        # print('\n* * * * * * *  =  =  =  =  =  =  Simulation Begin =  =  =  =  =  =  * * * * * * *')
        try:
            while t_curr < time_for_simulation:
                t_prev = t_curr
                t_curr = time.time() - t_init
                dt = t_curr - t_prev  # 1/125
                # print(f'\nt = {t_curr} (dt ={dt}):')

                # print('kp', kp)
                # print('kd', kd)

                # ---------- Read Sensors -------------------
                ee_pose = robot.get_actual_tcp_pose(wait=True)  # [x,y,z,rx,ry,rz] - ri: axis-angles
                ee_vel = robot.get_actual_tcp_speed(wait=False)
                internal_sensor_reading = robot.get_tcp_force(wait=False)

                # ----------- F/T calibration ----------------
                f_int = internal_sensor_reading - internal_sensor_bias

                external_sensor_tool = sensor_class.force_moment_feedback() - external_sensor_bias_tool
                f_ext = external_calibrate(external_sensor_tool, ee_pose)

                # print(f'f_int = {f_int} [N]')
                # print(f'f_ext = {f_ext} [N]')

                # detect contact with the surface
                if np.abs(f_int[2]) > contact_fz_threshold and contact_flag is False:
                    print('%%%%%%%%%% Contact established %%%%%%%%%%%')
                    contact_time = t_curr
                    contact_flag = True
                    print('initializing contact pd params')
                    # initialize impedance params
                    pose_mod = deepcopy(desired_pos[:6])
                    vel_mod = deepcopy(desired_pos[6:])
                    # contact pd parameters
                    kp = np.array([2250.0 * 5, 2250.0 * 5, 1250.0, 50.0, 50.0, 50.0])
                    kd = 2 * np.sqrt(kp) * 5
                    # introduces delay!!! in ref trajectory
                    # robot.force_mode_set_gain_scaling(2, wait=False)
                    # time.sleep(1)
                    # robot.force_mode_set_damping(0.5)

                # ------------- Minimum Jerk Trajectory updating ----------------------
                # Check if updating reference values with the minimum-jerk trajectory is necessary
                if t_curr <= time_trajectory:
                    [position_ref, orientation_ref, lin_vel_ref, ang_vel_ref] = planner.trajectory_planning(t_curr)
                    desired_pos = np.concatenate((position_ref, orientation_ref, lin_vel_ref, ang_vel_ref), axis=0)

                else:
                    # continue
                    [position_ref, orientation_ref, lin_vel_ref, ang_vel_ref] = planner.trajectory_planning(
                        time_trajectory)
                    desired_pos = np.concatenate((position_ref, orientation_ref, lin_vel_ref, ang_vel_ref), axis=0)

                # when contact is established
                if contact_flag:
                    if circle:
                        """Circle mode"""
                        theta_next, radius_next, x_spiral_next, y_spiral_next = next_circle(theta_current, dt)

                        spiral_x.append(x_spiral_next + desired_pos[0])
                        spiral_y.append(y_spiral_next + desired_pos[1])
                    else:
                        """Spiral Search mode"""
                        # print('Using spiral Mode')
                        theta_next, radius_next, x_spiral_next, y_spiral_next = next_spiral(theta_current, dt)
                        # add shift to the spiral search which is planned at (0,0)
                        spiral_x.append(x_spiral_next + desired_pos[0])
                        spiral_y.append(y_spiral_next + desired_pos[1])

                    theta_current = deepcopy(theta_next)

                    # we collect spiral trajectory at this point to exclude everything before contact was made
                    robot_spiral_x.append(ee_pose[0])
                    robot_spiral_y.append(ee_pose[1])
                    # print('Spiral')

                    if use_impedance:
                        # f_int or f_ext
                        print('Using Impedance')
                        X_next = control.impedance_equation(pose_ref=desired_pos[:6], vel_ref=desired_pos[6:],
                                                            pose_mod=pose_mod, vel_mod=vel_mod,
                                                            f_int=f_ext, f0=f0, dt=dt)
                        desired_pos = deepcopy(X_next)
                        pose_mod = X_next[:6]
                        vel_mod = X_next[6:]
                # ----------- - - - - = = = Control = = = - - - - -------------
                desired_pos[:2] += np.array([x_spiral_next, y_spiral_next])
                ori_real = at.AxisAngle_To_RotationVector(pose_desired[3:], ee_pose[3:])

                # Compute desired force and torque based on errors
                position_error = desired_pos[:3].T - ee_pose[:3]
                ori_error = desired_pos[3:6] - ori_real

                vel_pos_error = desired_pos[6:9].T - ee_vel[:3]
                vel_ori_error = desired_pos[9:12] - ee_vel[3:]

                desired_force = (np.multiply(np.array(position_error), np.array(kp[0:3]))
                                 + np.multiply(vel_pos_error, kd[0:3]))

                desired_torque = (np.multiply(np.array(ori_error), np.array(kp[3:6]))
                                  + np.multiply(vel_ori_error, kd[3:6]))


                if contact_flag:
                    if use_impedance:
                        # print('Impedance Control')
                        compensation = [0, 0, 1, 0, 0, 0] * internal_sensor_reading + [1, 1, 0, 1, 1,
                                                                                       1] * internal_sensor_bias
                        wrench_task = np.concatenate([desired_force, desired_torque]) - compensation
                        wrench_task[2] = -5 - internal_sensor_bias[2]

                    else:
                        # PD
                        # print('PD Control')
                        compensation = deepcopy(internal_sensor_bias)
                        wrench_task = np.concatenate([desired_force, desired_torque]) - compensation
                        wrench_task[2] = -5 - internal_sensor_bias[2]
                        # wrench_task[:2] = wrench_task[:2] - internal_sensor_reading[:2]  # Interaction forces compensation in xy.

                else:
                    # Free space
                    # print('Free Space Control')
                    compensation = deepcopy(internal_sensor_reading)
                    wrench_task = np.concatenate([desired_force, desired_torque]) - compensation

                # ---------------- Sending the wrench to the robot --------------------
                # print('wrench_task:', wrench_task)
                # Verify wrench safety limits:
                wrench_safe = circular_wrench_limiter(wrench_cmd=wrench_task)

                robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[1, 1, 1, 1, 1, 1],
                                       wrench=wrench_safe, f_type=2, limits=[2, 2, 1.5, 1, 1, 1])

                label = label_check(peg_xy=ee_pose[:2], hole_xy=real_goal_pose[:2])

                if data_collection:
                    if in_hole_stop(peg_xy=ee_pose[:2], hole_xy=real_goal_pose[:2]):
                        print('#############In HOLE!################')
                        robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[0, 0, 0, 0, 0, 0],
                                               wrench=[0, 0, 0, 0, 0, 0], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
                        robot.end_force_mode()
                        robot.reset_error()
                        success_flag = True
                        break

                if label:
                    time_labels.append(t_curr)

                if contact_flag:
                    # for graphs:
                    time_vec.append(t_curr)
                    # robot measurements
                    ee_pos_x_vec.append(ee_pose[0])
                    ee_pos_y_vec.append(ee_pose[1])
                    ee_pos_z_vec.append(ee_pose[2])
                    ee_vel_x_vec.append(ee_vel[0])
                    ee_vel_y_vec.append(ee_vel[1])
                    ee_vel_z_vec.append(ee_vel[2])
                    ee_ori_x_vec.append(ori_real[0])
                    ee_ori_y_vec.append(ori_real[1])
                    ee_ori_z_vec.append(ori_real[2])
                    ee_ori_vel_x_vec.append(ee_vel[3])
                    ee_ori_vel_y_vec.append(ee_vel[4])
                    ee_ori_vel_z_vec.append(ee_vel[5])
                    # minimum jerk
                    pos_min_jerk_x.append(desired_pos[0])
                    pos_min_jerk_y.append(desired_pos[1])
                    pos_min_jerk_z.append(desired_pos[2])
                    ori_min_jerk_x.append(desired_pos[3])
                    ori_min_jerk_y.append(desired_pos[4])
                    ori_min_jerk_z.append(desired_pos[5])
                    vel_min_jerk_x.append(desired_pos[6])
                    vel_min_jerk_y.append(desired_pos[7])
                    vel_min_jerk_z.append(desired_pos[8])
                    ori_vel_min_jerk_x.append(desired_pos[9])
                    ori_vel_min_jerk_y.append(desired_pos[10])
                    ori_vel_min_jerk_z.append(desired_pos[11])
                    # applied wrench
                    applied_wrench_fx.append(wrench_safe[0])
                    applied_wrench_fy.append(wrench_safe[1])
                    applied_wrench_fz.append(wrench_safe[2])
                    applied_wrench_mx.append(wrench_safe[3])
                    applied_wrench_my.append(wrench_safe[4])
                    applied_wrench_mz.append(wrench_safe[5])
                    # unclipped wrench
                    unclipped_wrench_fx.append(wrench_task[0])
                    unclipped_wrench_fy.append(wrench_task[1])
                    unclipped_wrench_fz.append(wrench_task[2])
                    unclipped_wrench_mx.append(wrench_task[3])
                    unclipped_wrench_my.append(wrench_task[4])
                    unclipped_wrench_mz.append(wrench_task[5])
                    # sensor readings
                    sensor_fx.append(f_int[0])
                    sensor_fy.append(f_int[1])
                    sensor_fz.append(f_int[2])
                    sensor_mx.append(f_int[3])
                    sensor_my.append(f_int[4])
                    sensor_mz.append(f_int[5])
                    # label
                    labels.append(label)

                # ---------- Stop simulation if the robot reaches the goal --------
                # goal without any error
                if in_hole_stop(peg_xy=ee_pose[:2], hole_xy=real_goal_pose[:2]) or (np.abs(ee_pose[2] - real_goal_pose[2]) <= deviation_from_goal_z):
                        # (np.abs(ee_pose[2] - real_goal_pose[2]) <= deviation_from_goal_z) \
                        # and (LA.norm(ee_pose[:2] - real_goal_pose[:2]) <= deviation_from_goal_xy):
                    print('-------------------------- :) ----------------------')
                    print(f"Goal has been reached at time {t_curr}")
                    robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[0, 0, 0, 0, 0, 0],
                                           wrench=[0, 0, 0, 0, 0, 0], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
                    robot.end_force_mode()
                    robot.reset_error()
                    success_flag = True
                    break

        except Exception as e:  # (Exception, KeyboardInterrupt) as e:
            print("Error has occurred during the simulation")
            print(e)
            success_flag = 'error'
            # traceback.print_exc()
            robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[0, 0, 0, 0, 0, 0],
                                   wrench=[0, 0, 0, 0, 0, 0], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
            robot.end_force_mode()
            # robot.reset_error()
            # robot.close()

    except KeyboardInterrupt:
        print("\n!! Ctrl+C Keyboard Interrupt !!\n")
        success_flag = 'interrupt'
        robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[0, 0, 0, 0, 0, 0],
                               wrench=[0, 0, 0, 0, 0, 0], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
        robot.end_force_mode()
        print('\nended force mode.\n')
        robot.reset_error()
        # robot.close()
        # print('closed the robot (finish the RTDE communication)')
        # return success_flag

    # = =   =   =   =   =   =   =   = End of KeyboardInterrupt Try and Exception    =   =   =   =   =   =   =

    # Stop the robot and End the force mode
    robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[0, 0, 0, 0, 0, 0],
                           wrench=[0, 0, 0, 0, 0, 0], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
    robot.end_force_mode()
    print('\nended force mode.\n')

    # End the communication with the robot
    robot.movel(start_pose - pose_error)
    robot.reset_error()

    # ****** = = = = = = = = = = * * * * * Plots Section * * * * * = = = = = = = = = = = = *******
    if save_data:
        df = pd.DataFrame({'t': time_vec, 'x': ee_pos_x_vec, 'y': ee_pos_y_vec, 'z': ee_pos_z_vec,
                           'rx': ee_ori_x_vec, 'ry': ee_ori_y_vec, 'rz': ee_ori_z_vec,
                           'vx': ee_vel_x_vec, 'vy': ee_vel_y_vec, 'vz': ee_vel_z_vec,
                           'Fx': sensor_fx, 'Fy': sensor_fy, 'Fz': sensor_fz,
                           'Mx': sensor_mx, 'My': sensor_my, 'Mz': sensor_mz, 'Case': labels})

        filename = "ep"+str(int(episode))+".csv"
        # filename = 'ep2.csv'
        filepath = os.path.join('/home/danieln7/Desktop/RobotCode2023/spiral', filename)
        df.to_csv(filepath)
        print('Successfully saved measurements')

    if plot_graphs:
        t = time_vec

        theta = np.linspace(0, 2 * np.pi, 100)

        x_error_top = ERROR_TOP * np.cos(theta) + real_goal_pose[0]
        y_error_top = ERROR_TOP * np.sin(theta) + real_goal_pose[1]
        x_error_bottom = ERROR_MIN * np.cos(theta) + real_goal_pose[0]
        y_error_bottom = ERROR_MIN * np.sin(theta) + real_goal_pose[1]

        if circle:
            print((np.abs(max(robot_spiral_x)) - np.abs(real_goal_pose[0])) * 1000)

        plt.figure("Spiral")
        plt.title(f'Spiral for: $e=3.5[mm]$, $v=1.5[m/s]$, $p=1.2[mm]$')
        plt.plot(spiral_x, spiral_y, 'g', label='Ref position')
        plt.plot(robot_spiral_x, robot_spiral_y, 'b', label='Robot position')
        plt.plot(real_goal_pose[0], real_goal_pose[1], "ro", label='Hole Center')
        plt.plot(spiral_x[0], spiral_y[0], "go", label='Spiral Start Point')
        plt.plot(x_error_top, y_error_top, 'r', label='Max Impedance Error: 0.8mm')
        plt.plot(x_error_bottom, y_error_bottom, 'g', label='Max Free Insertion Error: 0.4mm')
        plt.plot(robot_spiral_x[0], robot_spiral_y[0], "bo")
        plt.legend()
        plt.grid()

        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(pos_min_jerk_x, pos_min_jerk_y, 'g', label='Ref position')
        ax1.plot(ee_pos_x_vec, ee_pos_y_vec, 'b', label='Robot position')
        ax1.legend()
        ax1.grid()
        ax1.set_ylabel('X')
        ax1.set_xlabel('Y')

        ax2 = plt.subplot(312)
        ax2.plot(t, pos_min_jerk_x, 'g--', label='X_ref position')
        ax2.plot(t, ee_pos_x_vec, 'b', label='Xr position')
        ax2.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        ax2.legend()
        ax2.grid()
        ax2.set_title('X Position [m]')

        ax3 = plt.subplot(313)
        ax3.plot(t, pos_min_jerk_y, 'g--', label='Y_ref position')
        ax3.plot(t, ee_pos_y_vec, 'b', label='Yr position')
        ax3.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        ax3.legend()
        ax3.grid()
        ax3.set_title('Y Position [m]')

        plt.figure("Position")
        ax1 = plt.subplot(311)
        ax1.plot(t, ee_pos_x_vec, 'b', label='Xr position')
        ax1.plot(t, pos_min_jerk_x, 'r--', label='X_ref position')
        ax1.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        ax1.legend()
        ax1.grid()
        ax1.set_title('X Position [m]')

        ax2 = plt.subplot(312)
        ax2.plot(t, ee_pos_y_vec, 'b', label='Yr position')
        ax2.plot(t, pos_min_jerk_y, 'r--', label='Y_ref position')
        ax2.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        ax2.legend()
        ax2.grid()
        ax2.set_title('Y Position [m]')

        ax3 = plt.subplot(313)
        ax3.plot(t, ee_pos_z_vec, 'b', label='Zr position')
        ax3.plot(t, pos_min_jerk_z, 'r--', label='Z_ref position')
        ax3.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        ax3.legend()
        ax3.grid()
        ax3.set_title('Z Position [m]')
        ################################################################################################################
        # plt.figure("Linear velocity")
        # ax1 = plt.subplot(311)
        # ax1.plot(t, ee_vel_x_vec, 'b', label='Xr vel')
        # ax1.plot(t, vel_min_jerk_x, 'r--', label='X_ref vel')
        # ax1.legend()
        # ax1.set_title('X Velocity [m/s]')
        #
        # ax2 = plt.subplot(312)
        # ax2.plot(t, ee_vel_y_vec, 'b', label='Yr vel')
        # ax2.plot(t, vel_min_jerk_y, 'r--', label='Y_ref vel')
        # ax2.legend()
        # ax2.set_title('Y Velocity [m/s]')
        #
        # ax3 = plt.subplot(313)
        # ax3.plot(t, ee_vel_z_vec, 'b', label='Zr vel')
        # ax3.plot(t, vel_min_jerk_z, 'r--', label='Z_ref vel')
        # ax3.legend()
        # ax3.set_title('Z Velocity [m/s]')
        # ################################################################################################################
        # plt.figure("Angular Velocity")
        # ax1 = plt.subplot(311)
        # ax1.plot(t, ee_ori_vel_x_vec, 'b', label='Xr')
        # ax1.plot(t, ori_vel_min_jerk_x, 'r--', label='X_ref ')
        # ax1.legend()
        # ax1.set_title('X ori vel [rad/s]')
        #
        # ax2 = plt.subplot(312)
        # ax2.plot(t, ee_ori_vel_y_vec, 'b', label='Yr ')
        # ax2.plot(t, ori_vel_min_jerk_y, 'r--', label='Y_ref ')
        # ax2.legend()
        # ax2.set_title('Y ori vel [rad/s]')
        #
        # ax3 = plt.subplot(313)
        # ax3.plot(t, ee_ori_vel_z_vec, 'b', label='Zr ')
        # ax3.plot(t, ori_vel_min_jerk_z, 'r--', label='Z_ref ')
        # ax3.legend()
        # ax3.set_title('Z ori vel [rad/s]')
        # ################################################################################################################
        # plt.figure("Orientation")
        # ax1 = plt.subplot(311)
        # ax1.plot(t, ee_ori_x_vec, 'b', label='Xr')
        # ax1.plot(t, ori_min_jerk_x, 'r', label='X_ref ')
        # ax1.legend()
        # ax1.set_title('X ori [rad]')
        #
        # ax2 = plt.subplot(312)
        # ax2.plot(t, ee_ori_y_vec, 'b', label='Yr ')
        # ax2.plot(t, ori_min_jerk_y, 'r', label='Y_ref ')
        # ax2.legend()
        # ax2.set_title('Y ori [rad]')
        #
        # ax3 = plt.subplot(313)
        # ax3.plot(t, ee_ori_z_vec, 'b', label='Zr ')
        # ax3.plot(t, ori_min_jerk_z, 'r', label='Z_ref ')
        # ax3.legend()
        # ax3.set_title('Z ori[rad]')
        # ################################################################################################################
        plt.figure("Forces")
        ax1 = plt.subplot(311)
        ax1.plot(t, sensor_fx, 'b', label='Fx_sensor')
        ax1.plot(t, unclipped_wrench_fx, 'r', label='Fx_unclipped')
        ax1.plot(t, applied_wrench_fx, 'g', label='Fx_applied')
        ax1.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        # ax1.axvline(x=time_labels[0], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[0], 2)}")
        # ax1.axvline(x=time_labels[-1], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[-1], 2)}")
        ax1.grid()
        ax1.legend()
        ax1.set_title('Fx [N]')

        ax2 = plt.subplot(312)
        ax2.plot(t, sensor_fy, 'b', label='Fy_sensor')
        ax2.plot(t, unclipped_wrench_fy, 'r', label='Fy_unclipped')
        ax2.plot(t, applied_wrench_fy, 'g', label='Fy_applied')
        ax2.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        # ax2.axvline(x=time_labels[0], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[0], 2)}")
        # ax2.axvline(x=time_labels[-1], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[-1], 2)}")
        ax2.grid()
        ax2.legend()
        ax2.set_title('Fy [N]')

        ax3 = plt.subplot(313)
        ax3.plot(t, sensor_fz, 'b', label='Fz_sensor')
        ax3.plot(t, unclipped_wrench_fz, 'r', label='Fz_unclipped')
        ax3.plot(t, applied_wrench_fz, 'g', label='Fz_applied')
        ax3.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        # ax3.axvline(x=time_labels[0], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[0], 2)}")
        # ax3.axvline(x=time_labels[-1], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[-1], 2)}")
        ax3.grid()
        ax3.legend()
        ax3.set_title('Fz [N]')
        # ################################################################################################################
        plt.figure("Moments")
        ax1 = plt.subplot(311)
        ax1.plot(t, sensor_mx, 'b', label='Mx_sensor')
        ax1.plot(t, unclipped_wrench_mx, 'r', label='Mx_unclipped')
        ax1.plot(t, applied_wrench_mx, 'g', label='Mx_applied')
        ax1.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        # ax1.axvline(x=time_labels[0], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[0], 2)}")
        # ax1.axvline(x=time_labels[-1], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[-1], 2)}")
        ax1.legend()
        ax1.grid()
        ax1.set_title('Mx [Nm]')

        ax2 = plt.subplot(312)
        ax2.plot(t, sensor_my, 'b', label='My_sensor')
        ax2.plot(t, unclipped_wrench_my, 'r', label='My_unclipped')
        ax2.plot(t, applied_wrench_my, 'g', label='My_applied')
        ax2.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        # ax2.axvline(x=time_labels[0], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[0], 2)}")
        # ax2.axvline(x=time_labels[-1], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[-1], 2)}")
        ax2.legend()
        ax2.grid()
        ax2.set_title('My [Nm]')

        ax3 = plt.subplot(313)
        ax3.plot(t, sensor_mz, 'b', label='Mz_sensor')
        ax3.plot(t, unclipped_wrench_mz, 'r', label='Mz_unclipped')
        ax3.plot(t, applied_wrench_mz, 'g', label='Mz_applied')
        ax3.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        # ax3.axvline(x=time_labels[0], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[0], 2)}")
        # ax3.axvline(x=time_labels[-1], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[-1], 2)}")
        ax3.legend()
        ax3.grid()
        ax3.set_title('Mz [Nm]')

        plt.figure("Labels")
        plt.scatter(t, labels, label='classes')
        plt.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time, 2)}")
        # plt.axvline(x=time_labels[0], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[0], 2)}")
        # plt.axvline(x=time_labels[-1], color='k', linestyle='--', label=f"Overlap: {np.round(time_labels[-1], 2)}")
        plt.grid()
        plt.legend()
        plt.ylabel('Classes [0/1]')
        plt.xlabel('Time [sec]')

        plt.show()
    return success_flag
