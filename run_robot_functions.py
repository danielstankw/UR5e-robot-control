import numpy as np
import numpy.linalg as LA
import Onrobot
import URBasic
import time
from min_jerk_planner import PathPlan
import angle_transformation as at
from Control import Control
from matplotlib import pyplot as plt

# import traceback
# import os
# # ------ this for stopping the while loop with Ctrl+C (KeyboardInterrupt) this has to be at the top of the file!!--------
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

def run_robot_BASE(robot, FT_sensor, start_pose, desired_pose, pose_error, which_controller, which_param, is_plot=False):
    # host = '192.168.1.103'
    # robotModle = URBasic.robotModel.RobotModel()
    # robot = URBasic.urScriptExt.UrScriptExt(host=host, robotModel=robotModle)
    robot.reset_error()
    #FT_sensor = Onrobot.FT_sensor()

    Controller = Control(which_param=which_param)
    which_controller = which_controller

    # TCP and Payload initialization
    #robot.set_tcp(pose=[])
    robot.set_payload_mass(m=1.12)    #(m=0.0)
    robot.set_payload_cog(CoG=(0.005, 0.00, 0.084))    #(CoG=(0.0, 0.00, 0.0))

    # Compute goal_pose for success criterion:
    goal_pose = desired_pose - pose_error  # The real goal

    robot.movel(start_pose)
    time.sleep(0.5)  # Wait for the robot/measurements to be stable before starting the insertion
    pose_init = np.array(robot.get_actual_tcp_pose(wait=True))
    vel_init = np.array(robot.get_actual_tcp_speed(wait=False))
    vel_init[3:] = -vel_init[3:]
    #print(f"pose_init = {pose_init}. vel_init {vel_init}")

    # ------- redefining the angles ---------
    Vrot_initial = at.RotationVector(start_pose[3:6], desired_pose[3:6])  # creates rotation vector
    Vrot_curr = Vrot_initial
    # follow trajectory by decreasing magnitude of the rotation vector. example: 5*[0.707,0, -0.707]->0*[0.707,0,-0.707]
    Vrot_desired = np.array([0, 0, 0])

    # ------ Minimum jerk trajectory Initialization --------
    # Use for plotting only. no practical use # delete!! move this 2 lines to the plotting section:
    pose_real_init = np.append(start_pose[:3], Vrot_initial)   # Use for plotting only. no practical use # delete!! move this line to the plotting section
    pose_real = np.copy(pose_real_init)
    #pose_desired = np.append(desired_pose[:3], Vrot_desired)

    time_for_trajectory = 10 #10 #6 #5
    planner = PathPlan(pose_init, desired_pose, time_for_trajectory)  #PathPlan(pose_real_init, pose_desired, time_for_trajectory)
    [position_ref, Vrot_ref, lin_vel_ref, ang_vel_ref] = planner.trajectory_planning(0)

    # -------- Impedance model Initialization ------------

    # Here you can set F0 for the impedance control and the force control in Base sys
    F0_imp = np.zeros(6)   #np.array([0, 0, -7, 0, 0, 0])
    F0_fc = np.zeros(6)   #np.array([0, 0, -7, 0, 0, 0])

    # ----------- Control Settings -------------------
    # Control params for the free space:
    Kp_f = 0.5 * 4500 * np.array([1, 1, 1])
    Kd_f = 2 * 7 * np.sqrt(Kp_f)
    Kp_m = 0.5 *100 * np.array([1, 1, 1])
    Kd_m = 2 * 0.707 * np.sqrt(Kp_m)

    # ------------ Forces initialization ------------
    force_reading_init = np.copy(robot.get_tcp_force(wait=True))
    F_internal_init = np.copy(force_reading_init)
    F_external_init_Tool = np.copy(FT_sensor.force_moment_feedback())

    # Convert F_external measurements to BASE coordinate system:
    F_ext_force = at.Tool2Base_vec(pose_real_init[3:], F_external_init_Tool[:3])
    F_ext_moment = at.Tool2Base_vec(pose_real_init[3:], F_external_init_Tool[3:])
    F_external_init = np.append(F_ext_force, F_ext_moment)
    print(f'force_reading_init = {force_reading_init}')
    print(f"F_external_init(Tool) = {F_external_init_Tool}, F_external_init(Base) = {F_external_init}")

    # ------------- Plot's data initialization ----------------------
    plot_positions = True
    plot_oriantations = True
    plot_forces = True
    plot_moments = True
    t_curr_plot = [0]

    vel_real = np.copy(vel_init)
    pose_ref = np.copy(pose_real_init)
    vel_ref = np.copy(vel_init)
    pose_mod = np.copy(pose_ref)
    vel_mod = np.copy(vel_ref)

    pose_ref_next, vel_ref_next = np.copy(pose_ref), np.copy(vel_ref)
    F_external, force_reading = np.copy(F_external_init), np.copy(force_reading_init)

    if plot_positions or plot_oriantations:
        pose_real_plot = np.copy(pose_real)
        pose_ref_plot = np.copy(pose_ref)
        pose_mod_plot = np.copy(pose_mod)

    if plot_forces or plot_moments:
        F_internal_plot = np.zeros(6)
        F_external_plot = np.zeros(6)
        wrench_fc_plot = np.zeros(6)
        wrench_safe_plot = np.zeros(6)
        wrench_task_plot = np.zeros(6)


    # = =   =   =   =   =   =   =   =   Try and Exception with KeyboardInterrupt    =   =   =   =   =   =   =

    try:
        # ------------ Simulation(Loop) setup ----------------
        # Initialize force remote
        robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[0, 0, 0, 0, 0, 0],
                               wrench=(-force_reading_init), f_type=2, limits=[2, 2, 1.5, 1, 1, 1])

        t_init = time.time()
        t_curr = 0
        flag_contact_recognition = False  # flag_contact_recognition equals "True" after we hit the surface. than we change to inserting phase.
        time_for_simulation = time_for_trajectory + 5  # 3  # 2 #3
        time_contact_recognition = 0

        Fz_contact_recognition_factor = 1.5 #2 # 1.5 #[N]
        deviation_from_goal_z = 0.002  # 0.004  # Deviation from final goal in [m] which break the loop
        deviation_from_goal_xy = 0.005  # 0.0015
        # wrench_safety_limits = np.array([15,15,15,3,3,3])  #np.array([15,15,15,1,1,1])   #np.array([10,10,10,1,1,1])  #np.array([15,15,15,3,3,3])   #np.array([20,20,20,5,5,5])  #np.array([15,15,15,3,3,3])  #np.array([10, 10, 12, 1.2, 1.2,1.2])  # np.array([20, 20, 20, 2.5, 2.5, 2.5]) #np.array([3, 3, 5, 0.2, 0.2, 0.2]) #np.array([9, 9, 9, 1.2, 1.2, 1.2])  # An absolute values for limiting the wrench commands

        success_flag = False

        #   *   *   *   *   *   *   *   *   = = = = = = = Operation Loop = = = = = = = =    *   *   *   *   *   *   *

        # print('\n* * * * * * *  =  =  =  =  =  =  Simulation Begin =  =  =  =  =  =  * * * * * * *')
        try:
            while t_curr < time_for_simulation:
                t_prev = t_curr
                t_curr = time.time() - t_init
                dt = t_curr - t_prev   #1/125
                # print(f'\nt = {t_curr} (dt ={dt}):')

                # ---------- Read Sensors -------------------
                pose_reading = robot.get_actual_tcp_pose(wait=True)
                vel_real = robot.get_actual_tcp_speed(wait=False)
                force_reading = robot.get_tcp_force(wait=False)
                # print(f'force_reading = {force_reading}')
                F_external_Tool = FT_sensor.force_moment_feedback() - F_external_init_Tool
                Vrot_real = at.RotationVector(pose_reading[3:], desired_pose[
                                                                3:])  # Converting axis angles to Rotation vector (from the current pose to the desired pose)
                pose_real = np.append(pose_reading[:3], Vrot_real)
                vel_real[3:] = -vel_real[
                                3:]  # when working with Vrot method we use the opposite sign of the angular velocity

                # ----------- F/T calibration ----------------
                F_internal = force_reading - force_reading_init
                #F_external_prev = F_external.copy()
                F_ext_force = at.Tool2Base_vec(pose_reading[3:], F_external_Tool[:3])
                F_ext_moment = at.Tool2Base_vec(pose_reading[3:], F_external_Tool[3:])
                F_external = np.append(F_ext_force, F_ext_moment)
                F_external_normalized = F_external.copy()  #F_external_prev.copy() #F_external.copy()

                # # Delete2 if not necessary
                # # F/T normalization:
                # F_ext_norm = F_external[:3] / max(1e-10, LA.norm(F_external[:3]))
                # M_ext_norm = F_external[3:] / max(1e-10, LA.norm(F_external[3:]))
                # F_external_normalized = np.append(F_ext_norm, M_ext_norm)

                # # F/T normalization in xy directions only (FMxynorm):
                # F_ext_xy_norm, M_ext_xy_norm = F_external[:2], F_external[3:5]
                # if LA.norm(F_external[:2]) > 1 or True:
                #     F_ext_xy_norm = F_external[:2] / max(1e-10, LA.norm(F_external[:2]))
                # if LA.norm(F_external[3:5]) > 1 or True:
                #     M_ext_xy_norm = F_external[3:5] / max(1e-10, LA.norm(F_external[3:5]))
                # F_external_normalized = np.block([F_ext_xy_norm, F_external[2], M_ext_xy_norm, F_external[5]])  # delete!! change to F_external[2]


                # print('F_internal =', F_internal, '\nF_external =', F_external, '\nF_ext_tool =', F_external_Tool)

                # --------- Impedance model(generate X_next of the modified trajectory) ----------------
                pose_ref, vel_ref = np.copy(pose_ref_next), np.copy(vel_ref_next)
                # X_next = Controller.Impedance_equation(pose_mod,vel_mod,pose_ref,vel_ref,F_external,F0_imp,dt)

                if which_controller == 'imp':
                    # # For using F_internal as Fint in the impedance equations (Fits to Shir's method/Increasing Vrot oriantation method):
                    # X_next = Controller.Impedance_equation(pose_mod, vel_mod, pose_ref, vel_ref, F_internal, F0_imp, dt)
                    # # For using F_external as Fint in the impedance equtation (Used for Shir's project):
                    # X_next = Controller.Impedance_equation(pose_mod, vel_mod, pose_ref, vel_ref, F_external, F0_imp, dt)

                    # # For using F_internal as Fint in the impedance equtation !! When using decreasing Vrot oriantation method !!:
                    # X_next = Controller.Impedance_equation(pose_mod, vel_mod, pose_ref, vel_ref,[1, 1, 1, -1, -1, -1] * F_internal, F0_imp, dt)

                    # # For using F_external as Fint in the impedance equtation !! When using decreasing Vrot oriantation method !!:
                    X_next = Controller.Impedance_equation(pose_mod, vel_mod, pose_ref, vel_ref, [1,1,1,-1,-1,-1]*F_external_normalized, F0_imp, dt)

                    # if np.abs(F_external[2]) > 2.5 or flag_contact_recognition:
                    #     X_next = Controller.Impedance_equation(pose_mod,vel_mod,pose_ref,vel_ref,F_internal,F0_imp,dt)
                    # else:
                    #     X_next = np.append(pose_ref, vel_ref)
                    pose_mod = X_next[:6]
                    vel_mod = X_next[6:]

                    # # Delete!!! Avoid impedance in Z:
                    # pose_mod[2] = pose_ref[2]

                    # For updating pose_mod and vel_mod only after hitting the surface:
                    if not flag_contact_recognition:
                        pose_mod = np.copy(pose_ref)
                        vel_mod = np.copy(vel_ref)

                    # Compute wrench:
                    # wrench_fc = Controller.Force_controler(F_external, F_internal, F_internal_init, F0_fc)

                    wrench_imp = Controller.PD_controler(pose_real, pose_mod, vel_real, vel_mod, F_internal_init)

                    # For resist external forces in Z  (Used for Shir's project):
                    wrench_imp = Controller.PD_controler(pose_real, pose_mod, vel_real, vel_mod,
                                                         [0, 0, 1, 0, 0, 0] * force_reading + [1, 1, 0, 1, 1,
                                                                                               1] * F_internal_init)

                    # # # # Use wrench of -5[N] in Z direction (Used for Shir's project):
                    # wrench_imp[2] = -5 - F_internal_init[2]  #- F_internal[2] #F_internal_init[2]  # -8.5 - F_internal_init[2]

                    # # # PD for z direction:
                    # wrench_imp[2] = 2500*(pose_ref[2]-pose_real[2]) + 2*0.707*np.sqrt(2500)*(vel_ref[2]-vel_real[2]) - F_internal[2]


                # ------------- Minimum Jerk Trajectory updating ----------------------

                # Check if updating reference values with the minimum-jerk trajectory is necessary
                if t_curr <= time_for_trajectory:
                    [position_ref, Vrot_ref, lin_vel_ref, ang_vel_ref] = planner.trajectory_planning(t_curr)
                pose_ref_next = np.append(position_ref, Vrot_ref)
                vel_ref_next = np.append(lin_vel_ref, ang_vel_ref)

                # ----------- - - - - = = = Control = = = - - - - -------------

                # # For PD only (without impedance):
                if which_controller == 'PD':
                    # print(f'check!!!!')
                    Controller.A_imp_inv = 0 * Controller.A_imp
                    force_command = 300 * (pose_ref[:3] - pose_real[:3]) + 30 * (
                                vel_ref[:3] - vel_real[:3]) - F_internal_init[:3]
                    moment_command = 4 * (pose_real[3:] - pose_ref[3:]) + 1 * (
                                vel_real[3:] - vel_ref[3:]) - F_internal_init[3:]
                    wrench_PD = np.append(force_command, moment_command)
                    # Use wrench of -5[N] in Z direction:
                    wrench_PD[2] = -5 - F_internal_init[2]  # -8.5 - F_internal_init[2]
                    # # PD for z direction:
                    # wrench_PD[2] = 2500*(pose_ref[2]-pose_real[2]) + 2*0.707*np.sqrt(2500)*(vel_ref[2]-vel_real[2]) - F_internal[2]

                # wrench_PD_imp = Controller.PD_controler(pose_real, pose_mod, vel_real, vel_mod, force_reading)
                # For checking
                # wrench_PD_imp[2] = -5-F_internal_init[2]
                # wrench_PD_imp = np.array([0,0,-5,0,0,0])-F_internal_init

                # Enter insertion mode only if F_external in the direction of the insertion is above the threshold
                if np.abs(F_external[2]) > Fz_contact_recognition_factor or flag_contact_recognition:
                    if not (flag_contact_recognition):
                        time_contact_recognition = t_curr
                        print('Entered insertion mode at time:', time_contact_recognition)
                        flag_contact_recognition = True

                    if which_controller == 'PD':
                        wrench_task = np.copy(wrench_PD)
                    elif which_controller == 'imp':
                        wrench_task = np.copy(wrench_imp)


                    else:
                        raise RuntimeError(
                            f'!!Error in the Operation loop: "which_controller"={which_controller} is not a name of a known controler!')

                else:
                    # PD Controller for the free space:
                    force_command = Kp_f * (pose_ref[:3] - pose_real[:3]) + Kd_f * (
                            vel_ref[:3] - vel_real[:3]) - force_reading[:3]
                    moment_command = Kp_m * (pose_real[3:] - pose_ref[3:]) + Kd_m * (
                            vel_real[3:] - vel_ref[3:]) - force_reading[3:]
                    wrench_task = np.append(force_command, moment_command)

                # ---------------- Sending the wrench_fc to the robot --------------------

                # print('wrench_task:', wrench_task)

                # Verify wrench safety limits:
                wrench_safe = Controller.circular_wrench_limiter(wrench_cmd=wrench_task)
                # wrench_safety_criterion = np.abs(wrench_task) > wrench_safety_limits
                # if wrench_safety_criterion.any():
                #     # print('wrench safety limits is on')
                #     not_wrench_safety_criterion = np.logical_not(wrench_safety_criterion)
                #     wrench_task = (not_wrench_safety_criterion * wrench_task) + (
                #             wrench_safety_criterion * wrench_safety_limits * np.sign(wrench_task))
                # # print('wrench_task after verifying safety limits:', wrench_task)
                # wrench_safe = wrench_task.copy()

                robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[1, 1, 1, 1, 1, 1],
                                       wrench=wrench_safe, f_type=2, limits=[2, 2, 1.5, 1, 1, 1])

                # --------------- Collecting Plot's data ---------------------------
                t_curr_plot.append(t_curr)

                if is_plot:
                    if plot_positions or plot_oriantations:
                        pose_real_plot = np.vstack((pose_real_plot, pose_real))
                        pose_ref_plot = np.vstack((pose_ref_plot, pose_ref))
                        pose_mod_plot = np.vstack((pose_mod_plot, pose_mod))

                    if plot_forces or plot_moments:
                        F_internal_plot = np.vstack((F_internal_plot, F_internal))
                        F_external_plot = np.vstack((F_external_plot, F_external_normalized))  #F_external))
                        # wrench_fc_plot = np.vstack((wrench_fc_plot, wrench_fc + F_internal_init))
                        wrench_task_plot = np.vstack((wrench_task_plot, wrench_task + F_internal_init))
                        wrench_safe_plot = np.vstack((wrench_safe_plot, wrench_safe + F_internal_init))


                # ---------- Stop simulation if the robot reaches the goal (regarding to the err acceptable deviation) --------
                # if ( np.abs(pose_real[2] - desired_pose[2]) <= deviation_from_goal_z ) and ( LA.norm(pose_real[:2] - desired_pose[:2]) <= deviation_from_goal_xy ):
                if (np.abs(pose_real[2] - goal_pose[2]) <= deviation_from_goal_z) and (LA.norm(pose_real[:2] - goal_pose[:2]) <= deviation_from_goal_xy):
                    print(f"Goal is reached!!! at time {t_curr}")
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

    # # Print final errors that result from the simulation
    # pose_curr = np.array(robot.get_actual_tcp_pose(wait=True))
    # print(f"final error in x is {(np.abs(pose_curr[0] - desired_pose[0]))} for robot")
    # print(f"final error in y is {(np.abs(pose_curr[1] - desired_pose[1]))} for robot")
    # print(f"final error in z is {(np.abs(pose_curr[2] - desired_pose[2]))} for robot")
    # print(f"Total time of simulation {time.time() - t_init}")

    # End the communication with the robot
    robot.movel(start_pose - pose_error)
    robot.reset_error()
    # robot.close()
    #
    # print('closed the robot (finish the RTDE communication)')

    # If no contact accrued set the final time as the time of contact:
    if time_contact_recognition == 0:
        time_contact_recognition = t_curr_plot[-1]

    # ****** = = = = = = = = = = * * * * * Plots Section * * * * * = = = = = = = = = = = = *******
    if is_plot:
        if plot_positions:
            [x_real, y_real, z_real] = [pose_real_plot[:, 0], pose_real_plot[:, 1],
                                        pose_real_plot[:, 2]]  # np.array([10 ** 3]) *
            [x_ref, y_ref, z_ref] = [pose_ref_plot[:, 0], pose_ref_plot[:, 1], pose_ref_plot[:, 2]]  # np.array([10 ** 3]) *
            [x_mod, y_mod, z_mod] = [pose_mod_plot[:, 0], pose_mod_plot[:, 1], pose_mod_plot[:, 2]]  # np.array([10 ** 3]) *

            plt.figure('x position')
            plt.title('x Position')
            plt.plot(t_curr_plot, x_real, 'b', label='$x_{real}$')
            plt.plot(t_curr_plot, x_ref, '--', color='orange', label='$x_{ref}$')
            plt.plot(t_curr_plot, x_mod, 'g', label='$x_{impedance}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(x_mod), max(x_mod)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Position [m]')

            plt.figure('y position')
            plt.title('y Position')
            plt.plot(t_curr_plot, y_real, 'b', label='$y_{real}$')
            plt.plot(t_curr_plot, y_ref, '--', color='orange', label='$y_{ref}$')
            plt.plot(t_curr_plot, y_mod, 'g', label='$y_{impedance}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(y_mod), max(y_mod)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Position [m]')

            plt.figure('z position')
            plt.title('z Position')
            plt.plot(t_curr_plot, z_real, 'b', label='$z_{real}$')
            plt.plot(t_curr_plot, z_ref, '--', color='orange', label='$z_{ref}$')
            plt.plot(t_curr_plot, z_mod, 'g', label='$z_{impedance}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(z_mod), max(z_mod)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Position [m]')

            plt.figure('x-y')
            plt.title('x-y')
            plt.plot(x_real, y_real, 'b', label='real')
            plt.plot(x_ref, y_ref, '--', color='orange', label='ref')
            plt.plot(x_mod, y_mod, 'g', label='impedance')
            indx_contact = t_curr_plot.index(time_contact_recognition)
            contact_point = np.array([x_real[indx_contact], y_real[indx_contact], z_real[indx_contact]])
            plt.scatter(contact_point[0], contact_point[1], s=100, facecolors='k', edgecolors='r', zorder=10)
            goal_point = desired_pose - pose_error
            plt.scatter(goal_point[0], goal_point[1], s=100, facecolors='k', edgecolors='g', zorder=10)
            plt.legend()
            plt.grid()
            plt.axis('equal')
            plt.xlabel('X[m]')
            plt.ylabel('Y[m]')

        if plot_oriantations:
            [Rx_real, Ry_real, Rz_real] = [pose_real_plot[:, 3], pose_real_plot[:, 4], pose_real_plot[:, 5]]
            [Rx_ref, Ry_ref, Rz_ref] = [pose_ref_plot[:, 3], pose_ref_plot[:, 4], pose_ref_plot[:, 5]]
            [Rx_mod, Ry_mod, Rz_mod] = [pose_mod_plot[:, 3], pose_mod_plot[:, 4], pose_mod_plot[:, 5]]

            plt.figure('Rx rotation')
            plt.title('Rx Rotation')
            plt.plot(t_curr_plot, Rx_real, 'b', label='$Rx_{real}$')
            plt.plot(t_curr_plot, Rx_ref, '--', color='orange', label='$Rx_{ref}$')
            plt.plot(t_curr_plot, Rx_mod, 'g', label='$Rx_{impedance}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(Rx_ref), max(Rx_ref)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Rotation [rad]')

            plt.figure('Ry rotation')
            plt.title('Ry Rotation')
            plt.plot(t_curr_plot, Ry_real, 'b', label='$Ry_{real}$')
            plt.plot(t_curr_plot, Ry_ref, '--', color='orange', label='$Ry_{ref}$')
            plt.plot(t_curr_plot, Ry_mod, 'g', label='$Ry_{impedance}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(Ry_ref), max(Ry_ref)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Rotation [rad]')

            plt.figure('Rz rotation')
            plt.title('Rz Rotation')
            plt.plot(t_curr_plot, Rz_real, 'b', label='$Rz_{real}$')
            plt.plot(t_curr_plot, Rz_ref, '--', color='orange', label='$Rz_{ref}$')
            plt.plot(t_curr_plot, Rz_mod, 'g', label='$Rz_{impedance}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(Rz_ref), max(Rz_ref)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Rotation [rad]')

        if plot_forces:
            [Fx_internal, Fy_internal, Fz_internal] = [F_internal_plot[:, 0], F_internal_plot[:, 1], F_internal_plot[:, 2]]
            [Fx_external, Fy_external, Fz_external] = [F_external_plot[:, 0], F_external_plot[:, 1], F_external_plot[:, 2]]
            [Fx_wrench, Fy_wrench, Fz_wrench] = [wrench_task_plot[:, 0], wrench_task_plot[:, 1], wrench_task_plot[:, 2]]
            [Fx_safe, Fy_safe, Fz_safe] = [wrench_safe_plot[:, 0], wrench_safe_plot[:, 1], wrench_safe_plot[:, 2]]

            plt.figure('Fx force')
            plt.title('Fx Force')
            plt.plot(t_curr_plot, Fx_wrench, 'r', label='$Fx_{wrench}$')
            plt.plot(t_curr_plot, Fx_internal, 'b', label='$Fx_{internal}$')
            plt.plot(t_curr_plot, Fx_external, color='orange', label='$Fx_{external}$')
            plt.plot(t_curr_plot, Fx_safe, 'g--', label='$Fx_{safe}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(Fx_safe), max(Fx_safe)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Force [N]')

            plt.figure('Fy force')
            plt.title('Fy Force')
            plt.plot(t_curr_plot, Fy_wrench, 'r', label='$Fy_{wrench}$')
            plt.plot(t_curr_plot, Fy_internal, 'b', label='$Fy_{internal}$')
            plt.plot(t_curr_plot, Fy_external, color='orange', label='$Fy_{external}$')
            plt.plot(t_curr_plot, Fy_safe, 'g--', label='$Fy_{safe}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(Fy_safe), max(Fy_safe)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Force [N]')

            plt.figure('Fz force')
            plt.title('Fz Force')
            plt.plot(t_curr_plot, Fz_wrench, 'r', label='$Fz_{wrench}$')
            plt.plot(t_curr_plot, Fz_internal, 'b', label='$Fz_{internal}$')
            plt.plot(t_curr_plot, Fz_external, color='orange', label='$Fz_{external}$')
            plt.plot(t_curr_plot, Fz_safe, 'g--', label='$Fz_{safe}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(Fz_safe), max(Fz_safe)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Force [N]')

        if plot_moments:
            [Mx_internal, My_internal, Mz_internal] = [F_internal_plot[:, 3], F_internal_plot[:, 4], F_internal_plot[:, 5]]
            [Mx_external, My_external, Mz_external] = [F_external_plot[:, 3], F_external_plot[:, 4], F_external_plot[:, 5]]
            [Mx_wrench, My_wrench, Mz_wrench] = [wrench_task_plot[:, 3], wrench_task_plot[:, 4], wrench_task_plot[:, 5]]
            [Mx_safe, My_safe, Mz_safe] = [wrench_safe_plot[:, 3], wrench_safe_plot[:, 4], wrench_safe_plot[:, 5]]

            plt.figure('Mx Moment')
            plt.title('Mx Moment')
            plt.plot(t_curr_plot, Mx_internal, 'b', label='$Mx_{internal}$')
            plt.plot(t_curr_plot, Mx_external, color='orange', label='$Mx_{external}$')
            plt.plot(t_curr_plot, Mx_wrench, 'r', label='$Mx_{Wrench}$')
            plt.plot(t_curr_plot, Mx_safe, 'g--', label='$Mx_{safe}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(Mx_wrench), max(Mx_wrench)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Moment [N]')

            plt.figure('My Moment')
            plt.title('My Moment')
            plt.plot(t_curr_plot, My_internal, 'b', label='$My_{internal}$')
            plt.plot(t_curr_plot, My_external, color='orange', label='$My_{external}$')
            plt.plot(t_curr_plot, My_wrench, 'r', label='$My_{Wrench}$')
            plt.plot(t_curr_plot, My_safe, 'g--', label='$My_{safe}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(My_wrench), max(My_wrench)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Moment [N]')

            plt.figure('Mz Moment')
            plt.title('Mz Moment')
            plt.plot(t_curr_plot, Mz_internal, 'b', label='$Mz_{internal}$')
            plt.plot(t_curr_plot, Mz_external, color='orange', label='$Mz_{external}$')
            plt.plot(t_curr_plot, Mz_wrench, 'r', label='$Mz_{Wrench}$')
            plt.plot(t_curr_plot, Mz_safe, 'g--', label='$Mz_{safe}$')
            plt.plot([time_contact_recognition, time_contact_recognition], [min(Mz_wrench), max(Mz_wrench)], linestyle='--',
                     label='insertion mode', color='black')
            plt.legend()
            plt.grid()
            plt.xlabel('Time [sec]')
            plt.ylabel('Moment [N]')

        # Show all plots simultaneously
        plt.show()



    return success_flag

