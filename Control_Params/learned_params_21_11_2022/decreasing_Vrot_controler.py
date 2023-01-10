
from robosuite.controllers.base_controller import Controller
import numpy as np
from numpy import linalg as LA

from scipy.spatial.transform import Rotation as R
from robosuite.utils.min_jerk_planner import PathPlan, spiral_search_next_step
import robosuite.utils.angle_transformation as at

from matplotlib import pyplot as plt
import time
from robosuite.control_parameters.Control_param_20mm_circular_peg import Control_param_20mm_circular_peg
from robosuite.utils.impedance_control import Impedance_Controler
from robosuite.utils.action_convertors import Action_Object


class decreasing_Vrot_controler(Controller):
    """
    *********** Robosuite previouse notationts **************

    Controller for controlling the robot arm's joint torques. As the actuators at the mujoco sim level are already
    torque actuators, this "controller" usually simply "passes through" desired torques, though it also includes the
    typical input / output scaling and clipping, as well as interpolator features seen in other controllers classes
    as well

    NOTE: Control input actions assumed to be taken as absolute joint torques. A given action to this
    controller is assumed to be of the form: (torq_j0, torq_j1, ... , torq_jn-1) for an n-joint robot

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or list of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or list of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or list of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or list of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        torque_limits (2-list of float or 2-list of list of floats): Limits (N-m) below and above which the magnitude
            of a calculated goal joint torque will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)
            If not specified, will automatically set the limits to the actuator limits for this robot arm

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current joint torques to
            the goal joint torques during each timestep between inputted actions

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error
    """

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=0.05,
                 output_min=-0.05,
                 policy_freq=1,
                 torque_limits=None,
                 interpolator=None,
                 action_method_dict = None,
                 pos_err = np.zeros(3),
                 goal_pos_ee = None,
                 adaptive_xy_ref = False,
                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        # Control dimension
        self.control_dim = len(joint_indexes["joints"])

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # limits (if not specified, set them to actuator limits by default)
        self.torque_limits = np.array(torque_limits) if torque_limits is not None else self.actuator_limits

        # control frequency and time variables
        self.control_freq = policy_freq
        # For shir use 500 Hz
        self.freq_of_controller = 125   #125/2 delete2  #500 #125 #67.5 #125 #125/2   # the frequency of computing and updating the wrench commands
        self.dt_of_controller = 1/self.freq_of_controller
        self.freq_ratio = round(500 / self.freq_of_controller)   # The period time of simulation is 0.002[s] i.e. 500Hz
        self.steps_counter = self.freq_ratio
        self.cumulative_time_impedance_eq = 0
        self.cumulative_time_collect_data = 0

        # interpolator
        self.interpolator = interpolator

        # initialize torques
        self.goal_torque = None  # Goal torque desired, pre-compensation
        self.current_torque = np.zeros(self.control_dim)  # Current torques being outputted, pre-compensation
        self.torques = None  # Torques returned every time run_controller is called

        # ------- Pose initial and desired definitions ---------
        pos_init = self.initial_ee_pos  # np.array([-0.21141299, -0.01941388, 0.99609204])
        self.ori_init = R.from_matrix(
            self.initial_ee_ori_mat).as_rotvec()

        self.pos_err = pos_err  # i used "self" just so it can be use in other files.

        shifting_above_the_hole = 0.016 #0.04 + 0.016 #Delete + 0.04!!! 0.016 #0.04 #0.015   #0.04  #0.015 #shir uses 0.04       #0.015  # start from above the hole with this shifting value.
        self.pos_goal = goal_pos_ee + np.array([0, 0, shifting_above_the_hole]) + pos_err   #0 * pos_init + 1 * np.array([0.087, -0.05,
                                                    # 1.05 - 0.136]) + 1 * pos_err  # [-0.217, -0.02, 0.8365])  # pos_init + np.array([0,0,-0.15]) #0*np.array([0, 0.3, 0.1]) #
        self.ori_goal = np.array([np.pi, 0, 0])  #+ [-0.5, 0.2, 0]  # delete additional vector!!!  #1 * np.array([np.pi, 0, 0]) + 0 * self.ori_init + 0 * 1.2 * np.array(
        self.pos_goal2 = goal_pos_ee + pos_err #+ [0,0,0.02] #delete addition vector!!! #self.pos_goal + np.array([0, 0, -0.02])
        self.ori_goal2 = np.copy(self.ori_goal)
        self.ori_init2 = np.copy(self.ori_goal)
        self.ee_ori_mat_init2 = R.from_rotvec(self.ori_init2).as_matrix()
        self.ee_ori_mat_goal2 = R.from_rotvec(self.ori_goal2).as_matrix()

        pose_init = np.concatenate((pos_init, [0, 0, 0]))  # initial rotation is [0,0,0]

        # Set the rotation matrix of the Tool coordinate system:
        # Note: I always use -Z as the direction of insertion
        self.Tool_to_World_Rot_Mat = R.from_rotvec([0,0,np.pi/2]).as_matrix()  #delete2 [0,0,0]  # This set -Y to point towards the viewer like in the real robot
        self.World_to_Tool_Rot_Mat = self.Tool_to_World_Rot_Mat.T
        pose_init_TOOL = np.append( self.World_to_Tool_Rot_Mat @ pose_init[:3], self.World_to_Tool_Rot_Mat @ pose_init[3:] )

        # --------- Initialization of F/T sensor parameters ----------
        self.force_reading_bias = np.zeros(6)
        self.force_reading_bias_BASE = np.zeros(6)
        self.F_external_filtered = np.zeros(6)
        self.alpha = 1 #0.8 for 67.5HZ    #0.5 #0.3  #1 #0.8 #0.5 #1  # 0.7  # filtered_measurament = alpha*current + (1-alpha)*previouse

        # ---------- Minimum jerk trajectory Initialization ------------
        orientation_method = 'decreasing Vrot'
        self.t_init = self.sim.data.time
        self.t_real_prev = time.time()
        self.delay = 1  # stay stationary "self.delay" seconds between trajectories
        self.is_spiral_search = action_method_dict['spiral_p'] != None   # If is_spral_search == True than a spiral trajectory would add to the min jerk trajectory.
        self.sprial_theta = 0
        self.spiral_rad = 0
        self.spiral_p = 0
        self.spiral_v = 0


        # Compose the 1st trajectory for the free space (to via point):
        self.time_for_trajectory = 5  #2.5 #5       #3  # 4  # in simulation time seconds
        initial_pose, target_pose = np.concatenate((pos_init, self.ori_init)), np.concatenate(
            (self.pos_goal, self.ori_goal))
        self.planner = PathPlan(initial_pose, target_pose, self.time_for_trajectory, orientation_method)

        # Compose a second trajectory for insertion (from via point):
        self.time_for_trajectory2 = 10 #6  #10  # Delete 10  #6 #4  #14.5  #4 #3 #5  # in simulation time seconds
        initial_pose2, target_pose2 = target_pose, np.concatenate((self.pos_goal2, self.ori_goal2))
        self.planner2 = PathPlan(initial_pose2, target_pose2, self.time_for_trajectory2, orientation_method)

        # -------- Initialize Control Parameters -----------------
        self.use_impedance_flag = True #False #not (action_method =='PD')  # If use_impedance_flag set to false than PD controller will be used with no impedance
        self.switch_to_impedance = 2 #2.5  # the measured force in Z uses as a factor to switch to impedance control
        self.switching_occur_flag = False  # this flag would change to True if switching to impedance control will occur
        self.wrench_safety_limits_TOOL = dict(Fxy=20, Fz=15, Mxy=4, Mz=3)  #np.array([15, 15, 15, 3, 3, 3]) #dict(Fxy=20, Fz=20, Mxy=3, Mz=3)   #np.array([10, 10, 10, 2, 2, 2]) #np.array([20, 20, 20, 5, 5, 5])  #np.array([15, 15, 15, 3, 3, 3])  #np.array([15, 15, 15, 5, 5, 5])   #np.array([12, 12, 12, 2.5, 2.5, 2.5]) #np.array([20, 20, 20, 5, 5, 5])   #np.abs([12, 12, 18, 1.5, 1.5,
        self.wrench_filtered_TOOL = np.zeros(6)
        self.beta = 1  #delete2 0.8 #1  # 0.7  # filtered_measurament = alpha*current + (1-alpha)*previouse

        self.adaptive_xy_ref = adaptive_xy_ref

        # PD parameters for free space:
        Kp_pos = 0.5 * 1 * 4500 * np.ones(3)  # 5*4500*np.ones(3)
        Kp_ori = 0.5 * 1 * 100 * np.ones(3)  # 5*100*np.ones(3)
        self.Kp = np.concatenate((Kp_pos, Kp_ori))
        Kd_pos = 1 * 5 * 0.707 * 2 * np.sqrt(Kp_pos)  # 5*0.707*2*np.sqrt(Kp_pos)
        Kd_ori = 0.707 * 2 * np.sqrt(Kp_ori)  # 2*0.707*2*np.sqrt(Kp_ori)
        self.Kd = np.concatenate((Kd_pos, Kd_ori))

        # PD parameters for impedance control:
        control_param = Control_param_20mm_circular_peg()
        self.Kp_pos_imp = np.diag(control_param.Kf_pd)
        self.Kd_pos_imp = np.diag(control_param.Cf_pd)
        self.Kp_ori_imp = np.diag(control_param.Km_pd)
        self.Kd_ori_imp = np.diag(control_param.Cm_pd)

        # Impedance initialization:
        self.copy_ref_as_mod_flag = True
        self.solve_impedance_flag = False
        self.impedance_controler = Impedance_Controler(control_param)
        self.pose_mod_TOOL = np.copy(pose_init_TOOL)
        self.vel_mod_TOOL = np.zeros(6)

        # Set ActionConvertor and params_dict:
        self.ActionConvertor = Action_Object(action_method_dict=action_method_dict)
        # self.params_dict = {
        #     params_name: 0 if (params_name == 'spiral_p' or params_name == 'spiral_v') else np.zeros((6, 6))
        #     for params_name, params_type in action_method_dict.items()}
        # Delete3 rewite it in a more elegant way for all control params:
        Kp_vec = np.append(self.Kp_pos_imp, self.Kp_ori_imp)
        # self.params_dict['Kp'] = np.diag(Kp_vec)
        Kd_vec = np.append(self.Kd_pos_imp, self.Kd_ori_imp)
        # self.params_dict['Kd'] = np.diag(Kd_vec)
        self.params_dict = dict(K_imp=self.impedance_controler.K_imp, C_imp=self.impedance_controler.C_imp,
                                M_imp=self.impedance_controler.M_imp, Kp=np.diag(Kp_vec), Kd=np.diag(Kd_vec),
                                spiral_p=self.spiral_p, spiral_v=self.spiral_v)

        # ------- initialize plot settings and data ---------
        self.to_print_data = False #True #False
        self.plot_positions = True  # True
        self.plot_xy = True
        self.plot_oriantations = True
        self.plot_forces = True  # True
        self.plot_moments = True  # True
        self.plot_lin_vel = True #False #True
        self.plot_ang_vel = True #False #True

        self.t_curr_plot = [0]
        self.switch_to_impedance_time = self.time_for_trajectory + self.delay + self.time_for_trajectory2

        if self.plot_positions or self.plot_oriantations:
            self.pose_real_plot = np.copy(pose_init_TOOL)
            self.pose_ref_plot = np.copy(pose_init_TOOL)
            self.pose_mod_plot = np.copy(self.pose_mod_TOOL)

        if self.plot_forces or self.plot_moments:
            self.F_external_plot = np.zeros(6)
            self.PD_wrench_plot = np.zeros(6)
            self.wrench_command_plot = np.zeros(6)
            self.F_external_filtered_plot = np.zeros(6)

        if self.plot_ang_vel or self.plot_lin_vel:
            self.vel_real_plot = np.zeros(6)
            self.vel_ref_plot = np.zeros(6)
            self.vel_mod_plot = np.zeros(6)

    def orientation_error(self, desired, current):
        """
        This function calculates a 3-dimensional orientation error vector for use in the
        impedance controller. It does this by computing the delta rotation between the
        inputs and converting that rotation to exponential coordinates (axis-angle
        representation, where the 3d vector is axis * angle).
        See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
        Optimized function to determine orientation error from matrices

        Args:
            desired (np.array): 2d array representing target orientation matrix
            current (np.array): 2d array representing current orientation matrix

        Returns:
            np.array: 2d array representing orientation error as a matrix
        """
        rc1 = current[0:3, 0]
        rc2 = current[0:3, 1]
        rc3 = current[0:3, 2]
        rd1 = desired[0:3, 0]
        rd2 = desired[0:3, 1]
        rd3 = desired[0:3, 2]

        error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

        return error

    def wrench_limiter(self, wrench_cmd):
        # Limit the wrench in TOOL frame
        wrench_safety_criterion = np.abs(wrench_cmd) > self.wrench_safety_limits_TOOL  #self.wrench_safety_limits
        if wrench_safety_criterion.any():
            limited_wrench = np.copy(wrench_cmd)
            if self.to_print_data:
                print('!! wrench safety limits is on !!')
            if np.inf in wrench_cmd:
                print('\n!!!! inf wrench !!!!\n')
            limited_wrench[wrench_safety_criterion] = np.sign(wrench_cmd[wrench_safety_criterion]) * self.wrench_safety_limits_TOOL[wrench_safety_criterion]

            return limited_wrench
        else:
            return wrench_cmd

    def circular_wrench_limiter(self, wrench_cmd):
        # Limit the wrench in TOOL frame in a circular way. meaning that Fxy and Mxy consider as a vector with limited radius
        limited_wrench = wrench_cmd.copy()
        Fxy, Fz, Mxy, Mz = wrench_cmd[:2], wrench_cmd[2], wrench_cmd[3:5], wrench_cmd[5]
        Fxy_size, Mxy_size = LA.norm(Fxy), LA.norm(Mxy)


        if Fxy_size > self.wrench_safety_limits_TOOL['Fxy']:
            Fxy_direction = Fxy / Fxy_size
            limited_wrench[:2] = self.wrench_safety_limits_TOOL['Fxy'] * Fxy_direction
        if Fz < -self.wrench_safety_limits_TOOL['Fz'] or Fz > self.wrench_safety_limits_TOOL['Fz']:
            limited_wrench[2] = np.sign(Fz) * self.wrench_safety_limits_TOOL['Fz']
        if Mxy_size > self.wrench_safety_limits_TOOL['Mxy']:
            Mxy_direction = Mxy / Mxy_size
            limited_wrench[3:5] = self.wrench_safety_limits_TOOL['Mxy'] * Mxy_direction
        if Mz < -self.wrench_safety_limits_TOOL['Mz'] or Mz > self.wrench_safety_limits_TOOL['Mz']:
            limited_wrench[5] = np.sign(Mz) * self.wrench_safety_limits_TOOL['Mz']

        if np.inf in wrench_cmd:
            print('\n!!!! inf wrench !!!!\n')

        return limited_wrench

    def set_wrench(self):
        self.t_curr = self.sim.data.time - self.t_init
        if self.to_print_data:
            print(f'\nt_curr = {self.t_curr}:')

        # ------ Get real pose and velocity -----
        pos_real = self.ee_pos
        ori_real = R.from_matrix(self.ee_ori_mat).as_rotvec()
        lin_vel_real = self.ee_pos_vel
        ang_vel_real = -self.ee_ori_vel

        # ----------- Minimum Jerk (with 2 trajectories) -------------

        if self.t_curr <= self.time_for_trajectory:  # step <= self.steps_for_trajectory:
            [pos_ref, Vrot_ref, lin_vel_ref, ang_vel_ref] = self.planner.get_reference_pose(self.t_curr)
            Vrot_real = at.RotationVector(ori_real, self.ori_goal)
        elif self.t_curr <= self.time_for_trajectory + self.delay:
            [pos_ref, Vrot_ref, lin_vel_ref, ang_vel_ref] = self.planner.get_reference_pose(self.time_for_trajectory)
            Vrot_real = at.RotationVector(ori_real, self.ori_goal)
            if self.to_print_data:
                print(f'self.joint_pos = {self.joint_pos}')
        elif self.t_curr <= self.time_for_trajectory + self.delay + self.time_for_trajectory2:
            [pos_ref, Vrot_ref, lin_vel_ref, ang_vel_ref] = self.planner2.get_reference_pose(
                self.t_curr - self.delay - self.time_for_trajectory)
            Vrot_real = at.RotationVector(ori_real, self.ori_goal2)
            self.solve_impedance_flag = True
        else:
            [pos_ref, Vrot_ref, lin_vel_ref, ang_vel_ref] = self.planner2.get_reference_pose(self.time_for_trajectory2)
            Vrot_real = at.RotationVector(ori_real, self.ori_goal2)

        # Add Spiral Search contribution (when p=0 or v=0 there would be no contribution):
        if self.switching_occur_flag:
            self.sprial_theta, self.spiral_rad, x_spiral, y_spiral = spiral_search_next_step(theta=self.sprial_theta, v=self.spiral_v,
                                                                             p=self.spiral_p, dt=self.dt_of_controller)
            pos_ref[:2] += [x_spiral, y_spiral]

        pose_real = np.concatenate((pos_real, Vrot_real))
        vel_real = np.concatenate((lin_vel_real, ang_vel_real))
        pose_ref = np.concatenate((pos_ref, Vrot_ref))
        vel_ref = np.concatenate((lin_vel_ref, ang_vel_ref))

        if self.to_print_data:
            print(f'pos_real={pos_real}\npos_ref={pos_ref}\nVrot_real={Vrot_real}\nVrot_ref={Vrot_ref}')

        # ------- F/T sensors reading ---------------
        if self.time_for_trajectory + 0.5 * self.delay <= self.t_curr and self.t_curr < self.time_for_trajectory + self.delay:
            self.force_reading_bias = self.force_reading
            self.force_reading_bias_BASE = self.force_reading_BASE
        # Calibrate the measurement by substructing the force_reading which was taken before the insertion phase
        F_external = self.force_reading - self.force_reading_bias
        F_external_BASE = self.force_reading_BASE - self.force_reading_bias_BASE
        self.F_external_filtered = self.alpha * F_external_BASE + (1 - self.alpha) * self.F_external_filtered

        if self.to_print_data:
            print(f'F_external = {F_external}')
            print(f'F_external_BASE = {F_external_BASE}')
            print(f'force_reading_bias = {self.force_reading_bias}')

        # # --------- Convert to Tool coordinate system ----------------------
        self.pose_real_TOOL = np.append(self.World_to_Tool_Rot_Mat @ pose_real[:3], self.World_to_Tool_Rot_Mat @ pose_real[3:])
        self.vel_real_TOOL = np.append(self.World_to_Tool_Rot_Mat @ vel_real[:3], self.World_to_Tool_Rot_Mat @ vel_real[3:])
        self.pose_ref_TOOL = np.append(self.World_to_Tool_Rot_Mat @ pose_ref[:3], self.World_to_Tool_Rot_Mat @ pose_ref[3:])
        self.vel_ref_TOOL = np.append(self.World_to_Tool_Rot_Mat @ vel_ref[:3], self.World_to_Tool_Rot_Mat @ vel_ref[3:])
        self.F_external_TOOL = np.append(self.World_to_Tool_Rot_Mat @ F_external_BASE[:3], self.World_to_Tool_Rot_Mat @ F_external_BASE[3:])
        self.F_external_filtered_TOOL = np.append(self.World_to_Tool_Rot_Mat @ self.F_external_filtered[:3], self.World_to_Tool_Rot_Mat @ self.F_external_filtered[3:])
        self.F_external = F_external_BASE


        # --------- Impedance Solver (generate X_next of the modified trajectory) ----------------
        #if self.solve_impedance_flag and not self.copy_ref_as_mod_flag and self.use_impedance_flag:
        # Change this!! consider change the way solve_impedance_flag is set:
        if self.solve_impedance_flag and not self.copy_ref_as_mod_flag and self.use_impedance_flag and self.switching_occur_flag:
        # # delete!!!: For Shir PD only checks:
        # if False:
            if self.adaptive_xy_ref:
                # Use adaptive referance for xy (i.e. use the real xy as the referance):
                self.pose_ref_TOOL[:2] = self.pose_real_TOOL[:2]
                #print(f'adaptive_xy is on')

            time_before = time.time()

            # # Use normalized F and M:
            # F_ext, M_ext = self.F_external_filtered_TOOL[:3], self.F_external_filtered_TOOL[3:]
            # F_normalized = np.append(F_ext/LA.norm(F_ext) , M_ext/LA.norm(M_ext))

            # Use normaliztion in xy directions only (FMxynorm):
            F_ext_xy, M_ext_xy = self.F_external_filtered_TOOL[:2], self.F_external_filtered_TOOL[3:5]
            F_ext_xy_norm, M_ext_xy_norm = F_ext_xy/max( 1e-10, LA.norm(F_ext_xy) ), M_ext_xy/max( 1e-10, LA.norm(M_ext_xy) )   # For normalizing xy only
            F_normalized = np.block([F_ext_xy_norm, self.F_external_filtered_TOOL[2], M_ext_xy_norm, self.F_external_filtered_TOOL[5]])
            self.F_external_filtered_TOOL = F_normalized

            # # Use Fz for normalization:
            # Fz = self.F_external_filtered_TOOL[2]
            # F_normalized = self.F_external_filtered_TOOL / Fz
            # F_normalized[2] = Fz
            # self.F_external_filtered_TOOL = F_normalized

            X_next = self.impedance_controler.Impedance_equation(self.pose_mod_TOOL, self.vel_mod_TOOL, self.pose_ref_TOOL,
                                                                 self.vel_ref_TOOL,
                                                                 [1, 1, 1, -1, -1, -1] * self.F_external_filtered_TOOL,
                                                                 dt=1 / self.freq_of_controller)

            delta_time = time.time() - time_before
            self.cumulative_time_impedance_eq += delta_time
            self.pose_mod_TOOL = X_next[:6]
            self.vel_mod_TOOL = X_next[6:]
            if self.to_print_data:
                print('time it takes to solve impedance equations =', delta_time)

        elif self.solve_impedance_flag:
            self.pose_mod_TOOL = self.pose_ref_TOOL.copy()   #np.concatenate((pos_ref, Vrot_ref))
            self.vel_mod_TOOL = self.vel_ref_TOOL.copy()
            self.copy_ref_as_mod_flag = False
        else:
            self.pose_mod_TOOL = self.pose_ref_TOOL.copy()  # np.concatenate((pos_ref, Vrot_ref))
            self.vel_mod_TOOL = self.vel_ref_TOOL.copy()

        # --------- Calculate Wrench Commands using PD control law --------------
        # switch to impedance control if Fz is greater than the switching factor:

        if np.abs(self.F_external_TOOL[2]) < self.switch_to_impedance and not self.switching_occur_flag or self.t_curr < self.time_for_trajectory + self.delay:
        # # delete!!! for shir no switching to impedance:
        # if True:
            desired_force_TOOL = self.Kp[:3] * (self.pose_ref_TOOL[:3] - self.pose_real_TOOL[:3]) + self.Kd[:3] * (self.vel_ref_TOOL[:3] - self.vel_real_TOOL[:3])
            desired_torque_TOOL = self.Kp[3:] * -(self.pose_ref_TOOL[3:] - self.pose_real_TOOL[3:]) + self.Kd[3:] * -(self.vel_ref_TOOL[3:] - self.vel_real_TOOL[3:])
        else:
            desired_force_TOOL = self.Kp_pos_imp * (self.pose_mod_TOOL[:3] - self.pose_real_TOOL[:3]) + self.Kd_pos_imp * (self.vel_mod_TOOL[:3] - self.vel_real_TOOL[:3])
            desired_torque_TOOL = self.Kp_ori_imp * -(self.pose_mod_TOOL[3:] - self.pose_real_TOOL[3:]) + self.Kd_ori_imp * -(self.vel_mod_TOOL[3:] - self.vel_real_TOOL[3:])

            if not self.switching_occur_flag:
                self.switching_occur_flag = True
                self.switch_to_impedance_time = self.t_curr    # The time when initial contact with the surface is recognized.
                self.switch_to_impedance_Z = self.pose_real_TOOL[2]   # The position in Z when initial contact with the surface is recognized (this variablegftfbbbds is also beening used inside the env).

        self.PD_wrench_TOOL = np.concatenate((desired_force_TOOL, desired_torque_TOOL))

        if self.to_print_data: #or True:
            print('wrench_command (TOOL) before verifying safety limits:', self.PD_wrench_TOOL)
        self.wrench_safe_TOOL = self.circular_wrench_limiter(self.PD_wrench_TOOL)  #self.wrench_limiter(self.PD_wrench_TOOL)  # Limit the applied wrench
        if self.to_print_data: #or True:
            print('wrench_command (TOOL) after verifying safety limits:', self.wrench_safe_TOOL)

        # If you set beta <= 1 than this block applies wrench filter (expected to be more similar to the behviour of the real robot because of it's force control):
        # if self.switching_occur_flag: # consider not use "if" or use filter only from the via point (time condition)
        self.wrench_filtered_TOOL = self.beta * self.wrench_safe_TOOL + (1 - self.beta) * self.wrench_filtered_TOOL

        # Covert wrench_command after verifiying saftey limits and applying filter to WORLD frame:
        self.wrench_command = np.append(self.Tool_to_World_Rot_Mat @ self.wrench_filtered_TOOL[:3], self.Tool_to_World_Rot_Mat @ self.wrench_filtered_TOOL[3:])

        # Convert wrench command to joint torques command:
        self.goal_torque = self.J_full.T @ self.wrench_command

    def collect_data(self):
        self.t_curr_plot.append(self.t_curr)

        if self.plot_positions or self.plot_oriantations:
            self.pose_real_plot = np.vstack((self.pose_real_plot, self.pose_real_TOOL))
            self.pose_ref_plot = np.vstack((self.pose_ref_plot, self.pose_ref_TOOL))
            self.pose_mod_plot = np.vstack((self.pose_mod_plot, self.pose_mod_TOOL))

        if self.plot_forces or self.plot_moments:
            self.F_external_plot = np.vstack((self.F_external_plot, self.F_external_TOOL))
            self.F_external_filtered_plot = np.vstack((self.F_external_filtered_plot, self.F_external_filtered_TOOL))
            self.PD_wrench_plot = np.vstack((self.PD_wrench_plot, self.PD_wrench_TOOL))
            self.wrench_command_plot = np.vstack((self.wrench_command_plot, self.wrench_filtered_TOOL))


        if self.plot_ang_vel or self.plot_lin_vel:
            self.vel_real_plot = np.vstack((self.vel_real_plot, self.vel_real_TOOL))
            self.vel_ref_plot = np.vstack((self.vel_ref_plot, self.vel_ref_TOOL))
            self.vel_mod_plot = np.vstack((self.vel_mod_plot, self.vel_mod_TOOL))

    def set_goal(self, action):
        """
        """
        action_values = action[0]
        self.ActionConvertor.update_params_values(action_values)
        self.ActionConvertor.params_placement_for_outer_source(self.params_dict)

        # delete3 use pointer instead:
        # Update params according to the received action:
        self.spiral_p = self.params_dict['spiral_p']
        self.spiral_v = self.params_dict['spiral_v']
        Kp = self.params_dict['Kp'].diagonal()
        Kd = self.params_dict['Kd'].diagonal()
        self.Kp_pos_imp = Kp[:3]
        self.Kd_pos_imp = Kd[:3]
        self.Kp_ori_imp = Kp[3:]
        self.Kd_ori_imp = Kd[3:]
        self.impedance_controler.K_imp = self.params_dict['K_imp']
        self.impedance_controler.C_imp = self.params_dict['C_imp']
        self.impedance_controler.M_imp = self.params_dict['M_imp']
        # Update the Impedance State-Space equations matrices:
        M_imp_inv = LA.pinv(self.impedance_controler.M_imp)
        self.impedance_controler.A_imp = np.block([[np.zeros([6, 6]), np.eye(6)],
                                                   [-M_imp_inv @ self.impedance_controler.K_imp,
                                                    -M_imp_inv @ self.impedance_controler.C_imp]])
        self.impedance_controler.B_imp = np.block([[np.zeros([6, 18])],
                                                   [M_imp_inv, M_imp_inv @ self.impedance_controler.K_imp,
                                                    M_imp_inv @ self.impedance_controler.C_imp]])
        self.impedance_controler.A_imp_inv = LA.pinv(self.impedance_controler.A_imp)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # **************** SET GOAL PART *************************

        # Update state
        self.update()
        # self.run_counter += 1
        # print(f'run_controller for the {self.run_counter} time. steps_counter = {self.steps_counter}')
        if self.steps_counter == self.freq_ratio:
            #print(f'set wrench!! steps_counter = {self.steps_counter}\n')
            self.steps_counter = 0
            # Compute desired wrench command:
            t_before = time.time()

            self.set_wrench()
            if self.to_print_data:
                print('the time took for one step of set_wrench =', time.time() - t_before)

            # Collect data for plots:
            t_temp = time.time()
            self.collect_data()
            delta_time = time.time() - t_temp
            if self.to_print_data:
                print('collecting data from one controler step took =', delta_time)
            self.cumulative_time_collect_data += delta_time

        self.steps_counter += 1

        # **************** RUN CONTROLLER PART *********************

        # Make sure goal has been set
        if self.goal_torque is None:
            # delete or change it to a rellavant operation
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        # self.update()
        # print(f't_run_controller = {self.sim.data.time-self.t_init}')

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                self.current_torque = self.interpolator.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            self.current_torque = np.array(self.goal_torque)
            # print(f'current_torque = {self.current_torque}')

        # Add gravity compensation
        self.torques = self.current_torque + self.torque_compensation

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # Return final torques
        return self.torques

    def reset_goal(self):
        """
        Resets joint torque goal to be all zeros (pre-compensation)
        """
        self.goal_torque = np.zeros(self.control_dim)

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_torque)

    def plot_simulation(self):

        print(f'The cumulative time for solving impedance equations = {self.cumulative_time_impedance_eq}')
        print(f'The cumulative time for collecting data for plots = {self.cumulative_time_collect_data}')

        # If no contact accure with the surface use the time for initial contact as the final time:
        if not self.switching_occur_flag:
            self.switch_to_impedance_time = self.t_curr_plot[-1]
        # ****** = = = = = = = = = = * * * * * Plots Section * * * * * = = = = = = = = = = = = *******

        if self.plot_positions:
            [x_real, y_real, z_real] = [self.pose_real_plot[:, 0], self.pose_real_plot[:, 1],
                                        self.pose_real_plot[:, 2]]  # np.array([10 ** 3]) *
            [x_ref, y_ref, z_ref] = [self.pose_ref_plot[:, 0], self.pose_ref_plot[:, 1],
                                     self.pose_ref_plot[:, 2]]  # np.array([10 ** 3]) *
            [x_mod, y_mod, z_mod] = [self.pose_mod_plot[:, 0], self.pose_mod_plot[:, 1],
                                     self.pose_mod_plot[:, 2]]  # np.array([10 ** 3]) *
            plt.figure('Position')
            plt.title('Position')

            ax1 = plt.subplot(311)
            ax1.plot(self.t_curr_plot, x_real, 'b', label='$x_{real}$')
            ax1.plot(self.t_curr_plot, x_ref, '--', color='orange', label='$x_{ref}$')
            ax1.plot(self.t_curr_plot, x_mod, 'g', label='$x_{mod}$')
            ax1.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(x_real), max(x_real)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax1.legend(loc='upper left')
            ax1.grid()
            ax1.set_xlabel('Time [sec]')
            ax1.set_ylabel('Position [m]')
            ax2 = plt.subplot(312)
            ax2.plot(self.t_curr_plot, y_real, 'b', label='$y_{real}$')
            ax2.plot(self.t_curr_plot, y_ref, '--', color='orange', label='$y_{ref}$')
            ax2.plot(self.t_curr_plot, y_mod, 'g', label='$y_{mod}$')
            ax2.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(y_real), max(y_real)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax2.legend(loc='upper left')
            ax2.grid()
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('Position [m]')
            ax3 = plt.subplot(313)
            ax3.plot(self.t_curr_plot, z_real, 'b', label='$z_{real}$')
            ax3.plot(self.t_curr_plot, z_ref, '--', color='orange', label='$z_{ref}$')
            ax3.plot(self.t_curr_plot, z_mod, 'g', label='$z_{mod}$')
            ax3.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(z_real), max(z_real)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax3.legend(loc='upper left')
            ax3.grid()
            ax3.set_xlabel('Time [sec]')
            ax3.set_ylabel('Position [m]')
            plt.tight_layout()

        if self.plot_xy:
            plt.figure('x-y')
            plt.title('x-y')
            plt.plot(x_real, y_real, 'b', label='$real$')
            plt.plot(x_ref, y_ref, '--', color='orange', label='$ref$')
            plt.plot(x_mod, y_mod, 'g', label='$mod$')
            indx_contact = self.t_curr_plot.index(self.switch_to_impedance_time)  #np.where(self.t_curr_plot == self.switch_to_impedance_time)[0]
            contact_point = np.array([x_real[indx_contact], y_real[indx_contact], z_real[indx_contact]])
            print(f'contact point = {contact_point}')
            plt.scatter(contact_point[0], contact_point[1], s=100, facecolors='k', edgecolors='r', zorder=10)
            goal_point = self.World_to_Tool_Rot_Mat @ (self.pos_goal2 - self.pos_err)
            plt.scatter(goal_point[0], goal_point[1], s=100, facecolors='k', edgecolors='g', zorder=10)
            plt.legend()
            plt.grid()
            plt.xlabel('X[m]')
            plt.ylabel('Y[m]')

        if self.plot_oriantations:
            [Rx_real, Ry_real, Rz_real] = [self.pose_real_plot[:, 3], self.pose_real_plot[:, 4],
                                           self.pose_real_plot[:, 5]]
            [Rx_ref, Ry_ref, Rz_ref] = [self.pose_ref_plot[:, 3], self.pose_ref_plot[:, 4], self.pose_ref_plot[:, 5]]
            [Rx_mod, Ry_mod, Rz_mod] = [self.pose_mod_plot[:, 3], self.pose_mod_plot[:, 4], self.pose_mod_plot[:, 5]]

            plt.figure('Rotation')
            plt.title('Rotation')

            ax1 = plt.subplot(311)
            ax1.plot(self.t_curr_plot, Rx_real, 'b', label='$Rx_{real}$')
            ax1.plot(self.t_curr_plot, Rx_ref, '--', color='orange', label='$Rx_{ref}$')
            ax1.plot(self.t_curr_plot, Rx_mod, 'g', label='$Rx_{mod}$')
            ax1.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Rx_real), max(Rx_real)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax1.legend(loc='upper left')
            ax1.grid()
            ax1.set_xlabel('Time [sec]')
            ax1.set_ylabel('Rotation [rad]')

            ax2 = plt.subplot(312)
            ax2.plot(self.t_curr_plot, Ry_real, 'b', label='$Ry_{real}$')
            ax2.plot(self.t_curr_plot, Ry_ref, '--', color='orange', label='$Ry_{ref}$')
            ax2.plot(self.t_curr_plot, Ry_mod, 'g', label='$Ry_{mod}$')
            ax2.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Ry_real), max(Ry_real)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax2.legend(loc='upper left')
            ax2.grid()
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('Rotation [rad]')

            ax3 = plt.subplot(313)
            ax3.plot(self.t_curr_plot, Rz_real, 'b', label='$Rz_{real}$')
            ax3.plot(self.t_curr_plot, Rz_ref, '--', color='orange', label='$Rz_{ref}$')
            ax3.plot(self.t_curr_plot, Rz_mod, 'g', label='$Rz_{mod}$')
            ax3.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Rz_real), max(Rz_real)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax3.legend(loc='upper left')
            ax3.grid()
            ax3.set_xlabel('Time [sec]')
            ax3.set_ylabel('Rotation [rad]')
            plt.tight_layout()

        if self.plot_forces:
            [Fx_external, Fy_external, Fz_external] = [self.F_external_plot[:, 0], self.F_external_plot[:, 1],
                                                       self.F_external_plot[:, 2]]
            [Fx_wrench, Fy_wrench, Fz_wrench] = [self.PD_wrench_plot[:, 0], self.PD_wrench_plot[:, 1], self.PD_wrench_plot[:, 2]]
            [Fx_wrench_applied, Fy_wrench_applied, Fz_wrench_applied] = [self.wrench_command_plot[:, 0], self.wrench_command_plot[:, 1],
                                        self.wrench_command_plot[:, 2]]
            [Fx_filtered, Fy_filtered, Fz_filtered] = [self.F_external_filtered_plot[:, 0], self.F_external_filtered_plot[:, 1],
                                                       self.F_external_filtered_plot[:, 2]]

            plt.figure('Forces')
            plt.title('Forces')
            ax1 = plt.subplot(311)
            # ax1.plot(self.t_curr_plot, Fx_external, 'b', label='$Fx_{external}$')
            ax1.plot(self.t_curr_plot, Fx_filtered, 'g', label='$Fx_{filtered}$')
            ax1.plot(self.t_curr_plot, Fx_wrench, 'r', label='$Fx_{wrench}$')
            ax1.plot(self.t_curr_plot, Fx_wrench_applied, linestyle='--', color='orange', label='$Fx_{applied}$')  #ax1.plot(self.t_curr_plot, Fx_wrench_applied, 'g--', label='$Fx_{wrench_applied}$')
            ax1.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Fx_wrench), max(Fx_wrench)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax1.legend(loc='upper left')
            ax1.grid()
            ax1.set_xlabel('Time [sec]')
            ax1.set_ylabel('Force [N]')
            ax2 = plt.subplot(312)
            # ax2.plot(self.t_curr_plot, Fy_external, 'b', label='$Fy_{external}$')
            ax2.plot(self.t_curr_plot, Fy_filtered, 'g', label='$Fy_{filtered}$')
            ax2.plot(self.t_curr_plot, Fy_wrench, 'r', label='$Fy_{wrench}$')
            ax2.plot(self.t_curr_plot, Fy_wrench_applied, linestyle='--', color='orange', label='$Fy_{applied}$')   # ax2.plot(self.t_curr_plot, Fy_wrench_applied, 'g--', label='$Fy_{wrench_applied}$')
            ax2.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Fy_wrench), max(Fy_wrench)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax2.legend(loc='upper left')
            ax2.grid()
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('Force [N]')
            ax3 = plt.subplot(313)
            #ax3.plot(self.t_curr_plot, Fz_external, 'b', label='$Fz_{external}$')
            ax3.plot(self.t_curr_plot, Fz_filtered, 'g', label='$Fz_{filtered}$')
            ax3.plot(self.t_curr_plot, Fz_wrench, 'r', label='$Fz_{wrench}$')
            ax3.plot(self.t_curr_plot, Fz_wrench_applied, linestyle='--', color='orange', label='$Fz_{applied}$') #   # ax3.plot(self.t_curr_plot, Fz_wrench_applied, 'g--', label='$Fz_{wrench_applied}$')
            ax3.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Fz_wrench), max(Fz_wrench)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax3.legend(loc='upper left')
            ax3.grid()
            ax3.set_xlabel('Time [sec]')
            ax3.set_ylabel('Force [N]')
            plt.tight_layout()

        if self.plot_moments:
            [Mx_external, My_external, Mz_external] = [self.F_external_plot[:, 3], self.F_external_plot[:, 4],
                                                       self.F_external_plot[:, 5]]
            [Mx_filtered, My_filtered, Mz_filtered] = [self.F_external_filtered_plot[:, 0],
                                                       self.F_external_filtered_plot[:, 1],
                                                       self.F_external_filtered_plot[:, 2]]
            [Mx_wrench, My_wrench, Mz_wrench] = [self.PD_wrench_plot[:, 3], self.PD_wrench_plot[:, 4], self.PD_wrench_plot[:, 5]]
            [Mx_wrench_applied, My_wrench_applied, Mz_wrench_applied] = [self.wrench_command_plot[:, 3],
                                                                         self.wrench_command_plot[:, 4],
                                                                         self.wrench_command_plot[:, 5]]

            plt.figure('Moments')
            plt.title('Moments')

            ax1 = plt.subplot(311)
            # ax1.plot(self.t_curr_plot, Mx_external, 'b', label='$Mx_{external}$')
            ax1.plot(self.t_curr_plot, Mx_filtered, 'g', label='$Mx_{filtered}$')
            ax1.plot(self.t_curr_plot, Mx_wrench, 'r', label='$Mx_{wrench}$')
            ax1.plot(self.t_curr_plot, Mx_wrench_applied, linestyle='--', color='orange', label='$Mx_{applied}$')
            ax1.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Mx_wrench), max(Mx_wrench)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax1.legend(loc='upper left')
            ax1.grid()
            ax1.set_xlabel('Time [sec]')
            ax1.set_ylabel('Moment [Nm]')
            ax2 = plt.subplot(312)
            # ax2.plot(self.t_curr_plot, My_external, 'b', label='$My_{external}$')
            ax2.plot(self.t_curr_plot, My_filtered, 'g', label='$My_{filtered}$')
            ax2.plot(self.t_curr_plot, My_wrench, 'r', label='$My_{wrench}$')
            ax2.plot(self.t_curr_plot, My_wrench_applied, linestyle='--', color='orange', label='$My_{applied}$')
            ax2.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(My_wrench), max(My_wrench)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax2.legend(loc='upper left')
            ax2.grid()
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('Moment [Nm]')
            ax3 = plt.subplot(313)
            # ax3.plot(self.t_curr_plot, Mz_external, 'b', label='$Mz_{external}$')
            ax3.plot(self.t_curr_plot, Mz_filtered, 'g', label='$Mz_{filtered}$')
            ax3.plot(self.t_curr_plot, Mz_wrench, 'r', label='$Mz_{wrench}$')
            ax3.plot(self.t_curr_plot, Mz_wrench_applied, linestyle='--', color='orange', label='$Mz_{applied}$')
            ax3.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Mz_wrench), max(Mz_wrench)],
                     linestyle='--',
                     label='insertion mode', color='black')
            ax3.legend(loc='upper left')
            ax3.grid()
            ax3.set_xlabel('Time [sec]')
            ax3.set_ylabel('Moment [Nm]')
            plt.tight_layout()

        if self.plot_lin_vel:
            [Vx_real, Vy_real, Vz_real] = [self.vel_real_plot[:, 0], self.vel_real_plot[:, 1],
                                           self.vel_real_plot[:, 2]]
            [Vx_ref, Vy_ref, Vz_ref] = [self.vel_ref_plot[:, 0], self.vel_ref_plot[:, 1],
                                        self.vel_ref_plot[:, 2]]
            [Vx_mod, Vy_mod, Vz_mod] = [self.vel_mod_plot[:, 0], self.vel_mod_plot[:, 1],
                                        self.vel_mod_plot[:, 2]]

            plt.figure('Linear velocity')
            plt.title('Linear velocity')

            ax1 = plt.subplot(311)
            ax1.plot(self.t_curr_plot, Vx_real, 'b', label='$v_x^{real}$')
            ax1.plot(self.t_curr_plot, Vx_ref, '--', color='orange', label='$v_x^{ref}$')
            ax1.plot(self.t_curr_plot, Vx_mod, 'g', label='$v_x^{mod}$')
            ax1.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Vx_real), max(Vx_real)],
                     'k--', label='insertion mode')
            ax1.legend(loc='upper left')
            ax1.grid()
            ax1.set_xlabel('Time [sec]')
            ax1.set_ylabel('velocity [m/s]')
            ax1.set_title(r'$v_x$')

            ax2 = plt.subplot(312)
            ax2.plot(self.t_curr_plot, Vy_real, 'b', label='$v^y_{real}$')
            ax2.plot(self.t_curr_plot, Vy_ref, '--', color='orange', label='$v^y_{ref}$')
            ax2.plot(self.t_curr_plot, Vy_mod, 'g', label='$v^y_{mod}$')
            ax2.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Vy_real), max(Vy_real)],
                     'k--', label='insertion mode')
            ax2.legend(loc='upper left')
            ax2.grid()
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('velocity [m/s]')
            ax2.set_title(r'$v_y$')

            ax3 = plt.subplot(313)
            ax3.plot(self.t_curr_plot, Vz_real, 'b', label='$v^z_{real}$')
            ax3.plot(self.t_curr_plot, Vz_ref, '--', color='orange', label='$v^z_{ref}$')
            ax3.plot(self.t_curr_plot, Vz_mod, 'g', label='$v^z_{mod}$')
            ax3.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Vz_real), max(Vz_real)],
                     'k--', label='insertion mode')
            ax3.legend(loc='upper left')
            ax3.grid()
            ax3.set_xlabel('Time [sec]')
            ax3.set_ylabel('velocity [m/s]')
            ax3.set_title(r'$v_z$')

            plt.tight_layout()

        if self.plot_ang_vel:
            [Wx_real, Wy_real, Wz_real] = [self.vel_real_plot[:, 3], self.vel_real_plot[:, 4],
                                           self.vel_real_plot[:, 5]]
            [Wx_ref, Wy_ref, Wz_ref] = [self.vel_ref_plot[:, 3], self.vel_ref_plot[:, 4],
                                        self.vel_ref_plot[:, 5]]
            [Wx_mod, Wy_mod, Wz_mod] = [self.vel_mod_plot[:, 3], self.vel_mod_plot[:, 4],
                                        self.vel_mod_plot[:, 5]]

            plt.figure('Angular velocity')
            plt.title('Angular velocity')

            ax1 = plt.subplot(311)
            ax1.plot(self.t_curr_plot, Wx_real, 'b', label='$\omega_x^{real}$')
            ax1.plot(self.t_curr_plot, Wx_ref, '--', color='orange', label='$\omega_x^{ref}$')
            ax1.plot(self.t_curr_plot, Wx_mod, 'g', label='$\omega_x^{mod}$')
            ax1.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Wx_real), max(Wx_real)],
                     'k--', label='insertion mode')
            ax1.legend(loc='upper left')
            ax1.grid()
            ax1.set_xlabel('Time [sec]')
            ax1.set_ylabel('angular velocity [rad/s]')
            ax1.set_title(r'$\omega_x$')

            ax2 = plt.subplot(312)
            ax2.plot(self.t_curr_plot, Wy_real, 'b', label='$\omega^y_{real}$')
            ax2.plot(self.t_curr_plot, Wy_ref, '--', color='orange', label='$\omega^y_{ref}$')
            ax2.plot(self.t_curr_plot, Wy_mod, 'g', label='$\omega^y_{mod}$')
            ax2.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Wy_real), max(Wy_real)],
                     'k--', label='insertion mode')
            ax2.legend(loc='upper left')
            ax2.grid()
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('angular velocity [rad/s]')
            ax2.set_title(r'$\omega_y$')

            ax3 = plt.subplot(313)
            ax3.plot(self.t_curr_plot, Wz_real, 'b', label='$\omega^z_{real}$')
            ax3.plot(self.t_curr_plot, Wz_ref, '--', color='orange', label='$\omega^z_{ref}$')
            ax3.plot(self.t_curr_plot, Wz_mod, 'g', label='$\omega^z_{mod}$')
            ax3.plot([self.switch_to_impedance_time, self.switch_to_impedance_time], [min(Wz_real), max(Wz_real)],
                     'k--', label='insertion mode')
            ax3.legend(loc='upper left')
            ax3.grid()
            ax3.set_xlabel('Time [sec]')
            ax3.set_ylabel('angular velocity [rad/s]')
            ax3.set_title(r'$\omega_z$')

            plt.tight_layout()

        print('delta_t =', self.t_curr_plot[3] - self.t_curr_plot[2])
        #print('delta_t =', self.t_curr_plot[30] - self.t_curr_plot[29])
        # Show all plots simultaneously
        plt.show()

    @property
    def name(self):
        return 'decreasing_Vrot_controller'
