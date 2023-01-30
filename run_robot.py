import numpy as np
import numpy.linalg as LA
import Onrobot
import URBasic
import time
from copy import deepcopy
from minimum_jerk_planner import PathPlan
import angle_transformation as at
from matplotlib import pyplot as plt
from Controller import Controller
from helper_functions import circular_wrench_limiter, external_calibrate


# import traceback
# import os
# # ------ this for stopping the while loop with Ctrl+C (KeyboardInterrupt) this has to be at the top of the file!!--------
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'


def run_robot(robot, start_pose, pose_desired, pose_error, control_dim, use_impedance, plot_graphs, sensor_class, time_insertion, time_trajectory):
    # Check if the UR controller is powered on and ready to run.
    if use_impedance:
        control = Controller(control_dim=control_dim)
    time.sleep(0.1)
    robot.reset_error()
    # robot.set_payload_mass(m=1.12)
    # robot.set_payload_cog(CoG=(0.005, 0.00, 0.084))

    # print(sensor_class.force_moment_feedback())

    #  move above the hole
    robot.movel(start_pose)
    robot.movel(start_pose)
    time.sleep(0.5)  # Wait for the robot/measurements to be stable before starting the insertion
    robot.zero_ftsensor()
    robot.force_mode_set_damping(0)
    robot.force_mode_set_gain_scaling(1)

    pose_init = np.array(robot.get_actual_tcp_pose(wait=True))
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
    # ----------for short distance to prevent floating---------------
    kp = np.array([4500.0, 4500.0, 2250.0, 50.0, 50.0, 50.0])
    kd = 2 * 7 * np.sqrt(kp)
    # ---------for long distances-------------
    # kp = np.array([4500.0, 4500.0, 5.0, 50.0, 50.0, 50.0])
    # kd = 2 * 0.707 * np.sqrt(kp)

    # ------------ Forces initialization ------------
    internal_sensor_bias = np.copy(robot.get_tcp_force(wait=True))
    external_sensor_bias_tool = np.copy(sensor_class.force_moment_feedback())
    external_sensor_bias_base = external_calibrate(external_sensor_bias_tool, pose_init)

    print(f'int sensor reading (base) = {internal_sensor_bias}')
    print(f'ext sensor reading (base) = {external_sensor_bias_base}')
    print(f'ext sensor reading (tool) = {external_sensor_bias_tool}')

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

        #   *   *   *   *   *   *   *   *   = = = = = = = Operation Loop = = = = = = = =    *   *   *   *   *   *   *
        # print('\n* * * * * * *  =  =  =  =  =  =  Simulation Begin =  =  =  =  =  =  * * * * * * *')
        try:
            while t_curr < time_for_simulation:
                t_prev = t_curr
                t_curr = time.time() - t_init
                dt = t_curr - t_prev  # 1/125
                # print(f'\nt = {t_curr} (dt ={dt}):')

                # ---------- Read Sensors -------------------
                ee_pose = robot.get_actual_tcp_pose(wait=True)  # [x,y,z,rx,ry,rz] - ri: axis-angles
                ee_vel = robot.get_actual_tcp_speed(wait=False)
                internal_sensor_reading = robot.get_tcp_force(wait=False)

                # ----------- F/T calibration ----------------
                f_int = internal_sensor_reading - internal_sensor_bias
                # external sensor
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
                    # Daniel simulation PD parameters
                    kp = np.array([5000.0, 5000.0, 2500.0, 450.0, 450.0, 450.0])
                    kd = 2 * np.sqrt(kp) * np.sqrt(2)
                    # Shir PD for impedance
                    # kp = np.array([700.0, 700.0, 200.0, 50.0, 50.0, 50.0])
                    # kd = 2 * 0.707 * np.sqrt(kp)

                    # robot.force_mode_set_damping(0.5)

                # ------------- Minimum Jerk Trajectory updating ----------------------
                # Check if updating reference values with the minimum-jerk trajectory is necessary
                if t_curr <= time_trajectory:
                    [position_ref, orientation_ref, lin_vel_ref, ang_vel_ref] = planner.trajectory_planning(t_curr)
                    desired_pos = np.concatenate((position_ref, orientation_ref, lin_vel_ref, ang_vel_ref), axis=0)
                else:
                    # continue
                    [position_ref, orientation_ref, lin_vel_ref, ang_vel_ref] = planner.trajectory_planning(time_trajectory)
                    desired_pos = np.concatenate((position_ref, orientation_ref, lin_vel_ref, ang_vel_ref), axis=0)

                    # when contact is established
                if contact_flag:
                    if use_impedance:
                        # f_int or f_ext
                        print('Using impedance')
                        X_next = control.impedance_equation(pose_ref=desired_pos[:6], vel_ref=desired_pos[6:],
                                                            pose_mod=pose_mod, vel_mod=vel_mod,
                                                            f_int=f_ext, f0=f0, dt=dt)
                        desired_pos = deepcopy(X_next)
                        pose_mod = X_next[:6]
                        vel_mod = X_next[6:]
                # ----------- - - - - = = = Control = = = - - - - -------------
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

                # TODO: shir
                if contact_flag:
                    if use_impedance:
                        print('Using impedance')
                        compensation = [0, 0, 1, 0, 0, 0] * internal_sensor_reading + [1, 1, 0, 1, 1, 1] * internal_sensor_bias
                        wrench_task = np.concatenate([desired_force, desired_torque]) - compensation
                        wrench_task[2] = -5 - internal_sensor_bias[2]

                    else:
                        # PD
                        print('Using PD')
                        compensation = internal_sensor_bias
                        wrench_task = np.concatenate([desired_force, desired_torque]) - compensation

                else:
                    # Free space
                    # print('Using Free Space')
                    compensation = deepcopy(internal_sensor_reading)
                    wrench_task = np.concatenate([desired_force, desired_torque]) - compensation

                # ---------------- Sending the wrench to the robot --------------------
                # print('wrench_task:', wrench_task)
                # Verify wrench safety limits:
                wrench_safe = circular_wrench_limiter(wrench_cmd=wrench_task)

                robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[1, 1, 1, 1, 1, 1],
                                       wrench=wrench_safe, f_type=2, limits=[2, 2, 1.5, 1, 1, 1])

                if True:
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
                    # # wrench - based on PD
                    applied_wrench_fx.append(wrench_safe[0])
                    applied_wrench_fy.append(wrench_safe[1])
                    applied_wrench_fz.append(wrench_safe[2])
                    applied_wrench_mx.append(wrench_safe[3])
                    applied_wrench_my.append(wrench_safe[4])
                    applied_wrench_mz.append(wrench_safe[5])
                    # sensor readings
                    sensor_fx.append(f_int[0])
                    sensor_fy.append(f_int[1])
                    sensor_fz.append(f_int[2])
                    sensor_mx.append(f_int[3])
                    sensor_my.append(f_int[4])
                    sensor_mz.append(f_int[5])

                # ---------- Stop simulation if the robot reaches the goal --------
                real_goal_pose = pose_desired - pose_error  # goal without any error
                if (np.abs(ee_pose[2] - real_goal_pose[2]) <= deviation_from_goal_z) \
                        and (LA.norm(ee_pose[:2] - real_goal_pose[:2]) <= deviation_from_goal_xy):
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
    if plot_graphs:
        t = time_vec
        plt.figure(f"Position: with impedance {use_impedance}")
        ax1 = plt.subplot(311)
        ax1.plot(t, ee_pos_x_vec, 'b', label='Xr position')
        ax1.plot(t, pos_min_jerk_x, 'r--', label='X_ref position')
        ax1.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time,2)}")
        ax1.legend()
        ax1.grid()
        ax1.set_title('X Position [m]')

        ax2 = plt.subplot(312)
        ax2.plot(t, ee_pos_y_vec, 'b', label='Yr position')
        ax2.plot(t, pos_min_jerk_y, 'r--', label='Y_ref position')
        ax2.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time,2)}")
        ax2.legend()
        ax2.grid()
        ax2.set_title('Y Position [m]')

        ax3 = plt.subplot(313)
        ax3.plot(t, ee_pos_z_vec, 'b', label='Zr position')
        ax3.plot(t, pos_min_jerk_z, 'r--', label='Z_ref position')
        ax3.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time,2)}")
        ax3.legend()
        ax3.grid()
        ax3.set_title('Z Position [m]')
        ################################################################################################################
        plt.figure(f"Linear velocity with impedance {use_impedance}")
        ax1 = plt.subplot(311)
        ax1.plot(t, ee_vel_x_vec, 'b', label='Xr vel')
        ax1.plot(t, vel_min_jerk_x, 'r--', label='X_ref vel')
        ax1.legend()
        ax1.grid()
        ax1.set_title('X Velocity [m/s]')

        ax2 = plt.subplot(312)
        ax2.plot(t, ee_vel_y_vec, 'b', label='Yr vel')
        ax2.plot(t, vel_min_jerk_y, 'r--', label='Y_ref vel')
        ax2.legend()
        ax2.grid()
        ax2.set_title('Y Velocity [m/s]')

        ax3 = plt.subplot(313)
        ax3.plot(t, ee_vel_z_vec, 'b', label='Zr vel')
        ax3.plot(t, vel_min_jerk_z, 'r--', label='Z_ref vel')
        ax3.legend()
        ax3.grid()
        ax3.set_title('Z Velocity [m/s]')
        ################################################################################################################
        plt.figure("Angular Velocity")
        ax1 = plt.subplot(311)
        ax1.plot(t, ee_ori_vel_x_vec, 'b', label='Xr')
        ax1.plot(t, ori_vel_min_jerk_x, 'r--', label='X_ref ')
        ax1.legend()
        ax1.grid()
        ax1.set_title('X ori vel [rad/s]')

        ax2 = plt.subplot(312)
        ax2.plot(t, ee_ori_vel_y_vec, 'b', label='Yr ')
        ax2.plot(t, ori_vel_min_jerk_y, 'r--', label='Y_ref ')
        ax2.legend()
        ax2.grid()
        ax2.set_title('Y ori vel [rad/s]')

        ax3 = plt.subplot(313)
        ax3.plot(t, ee_ori_vel_z_vec, 'b', label='Zr ')
        ax3.plot(t, ori_vel_min_jerk_z, 'r--', label='Z_ref ')
        ax3.legend()
        ax3.grid()
        ax3.set_title('Z ori vel [rad/s]')
        ################################################################################################################
        plt.figure(f"Orientation with impedance {use_impedance}")
        ax1 = plt.subplot(311)
        ax1.plot(t, ee_ori_x_vec, 'b', label='Xr')
        ax1.plot(t, ori_min_jerk_x, 'r', label='X_ref ')
        ax1.legend()
        ax1.grid()
        ax1.set_title('X ori [rad]')

        ax2 = plt.subplot(312)
        ax2.plot(t, ee_ori_y_vec, 'b', label='Yr ')
        ax2.plot(t, ori_min_jerk_y, 'r', label='Y_ref ')
        ax2.legend()
        ax2.grid()
        ax2.set_title('Y ori [rad]')

        ax3 = plt.subplot(313)
        ax3.plot(t, ee_ori_z_vec, 'b', label='Zr ')
        ax3.plot(t, ori_min_jerk_z, 'r', label='Z_ref ')
        ax3.legend()
        ax3.grid()
        ax3.set_title('Z ori[rad]')
        ################################################################################################################
        plt.figure(f"Forces with impedance {use_impedance}")
        ax1 = plt.subplot(311)
        ax1.plot(t, sensor_fx, 'b', label='Fx_sensor')
        ax1.plot(t, applied_wrench_fx, 'g', label='Fx_wrench')
        ax1.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time,2)}")
        ax1.grid()
        ax1.legend()
        ax1.set_title('Fx [N]')

        ax2 = plt.subplot(312)
        ax2.plot(t, sensor_fy, 'b', label='Fy_sensor')
        ax2.plot(t, applied_wrench_fy, 'g', label='Fy_wrench')
        ax2.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time,2)}")
        ax2.grid()
        ax2.legend()
        ax2.set_title('Fy [N]')

        ax3 = plt.subplot(313)
        ax3.plot(t, sensor_fz, 'b', label='Fz_sensor')
        ax3.plot(t, applied_wrench_fz, 'g', label='Fz_wrench')
        ax3.axvline(x=contact_time, color='k', linestyle='--', label=f"Time of contact: {np.round(contact_time,2)}")
        ax3.grid()
        ax3.legend()
        ax3.set_title('Fz [N]')
        ################################################################################################################
        plt.figure(f"Moments with impedance {use_impedance}")
        ax1 = plt.subplot(311)
        ax1.plot(t, sensor_mx, 'b', label='Mx_sensor')
        ax1.plot(t, applied_wrench_mx, 'g', label='Mx_wrench')
        ax1.legend()
        ax1.grid()
        ax1.set_title('Mx [Nm]')

        ax2 = plt.subplot(312)
        ax2.plot(t, sensor_my, 'b', label='My_sensor')
        ax2.plot(t, applied_wrench_my, 'g', label='My_wrench')
        ax2.legend()
        ax2.grid()
        ax2.set_title('My [Nm]')

        ax3 = plt.subplot(313)
        ax3.plot(t, sensor_mz, 'b', label='Mz_sensor')
        ax3.plot(t, applied_wrench_mz, 'g', label='Mz_wrench')
        ax3.legend()
        ax3.grid()
        ax3.set_title('Mz [Nm]')
        plt.show()

    return success_flag
