import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm
# from Control_param_lab_vertical_0deg import Control_param_lab_vertical_0deg
# from Control_param_polygon_horizontal_0deg import Control_param_polygon_horizontal_0deg
# from Adeline_param_vertical_0deg import Control_param_Adeline
# from take_messuraments_param import Control_param_messuraments
# from Control_param_3mm_cable import Control_param_3mm_cable
# from Control_param_4mm_cable import Control_param_4mm_cable
# from Control_param_08mm_cable import Control_param_08mm_cable

from Control_Params.Control_param_20mm_circular_peg  import Control_param_20mm_circular_peg
from Control_Params.Learned_Control_params import Learned_Params
#from stationary_point_params import Stationary_point_params

class Control:
    def __init__(self, which_param):
        # Set the safety limits for the wrench commands:
        self.wrench_safety_limits = dict(Fxy=20, Fz=15, Mxy=4, Mz=3)  #dict(Fxy=20, Fz=15, Mxy=4, Mz=3)  #np.array([15, 15, 15, 3, 3, 3])

        self.which_param = which_param

        discrete_method = 'expm'
        if which_param=='learned_params':
            Control_param = Learned_Params()
        elif which_param=='20mm_circular_peg':
            Control_param = Control_param_20mm_circular_peg()

        elif which_param=='lab_vertical_0deg':
            Control_param = Control_param_lab_vertical_0deg()
        elif which_param=='polygon_horizontal_0deg':
            Control_param = Control_param_polygon_horizontal_0deg()
        elif which_param == 'take_messuraments_param':
            Control_param = Control_param_messuraments()
        elif which_param=='Adeline_param_vertical_0deg':
            Control_param = Control_param_Adeline()
            discrete_method = 'forward_differential'
        elif which_param=='Control_param_3mm_cable':
            Control_param = Control_param_3mm_cable()
        elif which_param=='Control_param_4mm_cable':
            Control_param = Control_param_4mm_cable()
        elif which_param == 'Control_param_08mm_cable':
            Control_param = Control_param_08mm_cable()
        elif which_param == 'stationary_point_params':
            Control_param = Stationary_point_params()

        else:
            raise RuntimeError(
                f'!!Error in "Control" initialization: "which_param"={which_param} is not a name of a known paramateres file!')

        # - - - - - - PD controller parameters - - - - - - -
        self.Kf_pd = np.copy(Control_param.Kf_pd)
        self.Cf_pd = np.copy(Control_param.Cf_pd)
        self.Km_pd = np.copy(Control_param.Km_pd)
        self.Cm_pd = np.copy(Control_param.Cm_pd)
        #print('!!!!!!PD_params =',self.Kf_pd,self.Cf_pd)

        # - - - - - - Impedance parameters - - - - - - - - - -
        self.M_imp = np.copy(Control_param.M_imp)
        self.K_imp = np.copy(Control_param.K_imp)
        self.C_imp = np.copy(Control_param.C_imp)
        # self.Lxf = np.copy(Control_param.Lxf)
        # self.Lxm = np.copy(Control_param.Lxm)
        # self.LRf = np.copy(Control_param.LRf)
        # self.LRm = np.copy(Control_param.LRm)

        # - - - - - - Matrices for Impedance equation - - - - - - - - - -
        self.discrete_method = discrete_method
        self.A_imp = np.copy(Control_param.A_imp)
        self.B_imp = np.copy(Control_param.B_imp)

        if not(self.discrete_method == 'forward_differential'):
            self.A_imp_inv = np.copy(Control_param.A_imp_inv)

        # - - - - - - Force Controller Parameters - - - - - - - - - - - - -
        self.P_fc = np.copy(Control_param.P_fc)

    def __str__(self):
        return (f'A control object with Admittance+PD control and Force control functions. the set of parameters belogs to{self.which_param}')
    
    
    def circular_wrench_limiter(self, wrench_cmd):
        # Limit the wrench in TOOL frame in a circular way. meaning that Fxy and Mxy consider as a vector with limited radius
        limited_wrench = wrench_cmd.copy()
        Fxy, Fz, Mxy, Mz = wrench_cmd[:2], wrench_cmd[2], wrench_cmd[3:5], wrench_cmd[5]
        Fxy_size, Mxy_size = LA.norm(Fxy), LA.norm(Mxy)


        if Fxy_size > self.wrench_safety_limits['Fxy']:
            Fxy_direction = Fxy / Fxy_size
            limited_wrench[:2] = self.wrench_safety_limits['Fxy'] * Fxy_direction
        if Fz < -self.wrench_safety_limits['Fz'] or Fz > self.wrench_safety_limits['Fz']:
            limited_wrench[2] = np.sign(Fz) * self.wrench_safety_limits['Fz']
        if Mxy_size > self.wrench_safety_limits['Mxy']:
            Mxy_direction = Mxy / Mxy_size
            limited_wrench[3:5] = self.wrench_safety_limits['Mxy'] * Mxy_direction
        if Mz < -self.wrench_safety_limits['Mz'] or Mz > self.wrench_safety_limits['Mz']:
            limited_wrench[5] = np.sign(Mz) * self.wrench_safety_limits['Mz']

        if np.inf in wrench_cmd:
            print('\n!!!! inf wrench !!!!\n')

        return limited_wrench
    
    def PD_controler(self,pose_real,pose_des,vel_real,vel_des,compensation_forces):
        # The compensation_forces could be the current force reading or the initial force reading of the internal sensor
        # Input Dimensions: all inputs should be numpy arrays with shape = (6,) or a lists with a length of 6
        # Output: wrench_command is numpy array with shape = (6,0) that should be sent to the robot when using force mode

        delta_pose = np.append(pose_des[:3]-np.array(pose_real[:3]),np.array(pose_real[3:])-pose_des[3:])
        delta_vel = np.append(vel_des[:3]-np.array(vel_real[:3]),np.array(vel_real[3:])-vel_des[3:])

        force_command = np.dot(self.Kf_pd,delta_pose[:3])+np.dot(self.Cf_pd,delta_vel[:3])
        moment_command = np.dot(self.Km_pd,delta_pose[3:])+np.dot(self.Cm_pd,delta_vel[3:])
        wrench_command = np.append(force_command,moment_command) - compensation_forces

        return wrench_command

    def Impedance_equation(self,pose_mod,vel_mod,pose_ref,vel_ref,F_int,F0,dt):
        # F_int is the interaction forces that should be provided be some external sensors
        # F0 are some desired F/T that one can set. notice that F0 are the applied F/T and not the measured F/T
        # Input Dimensions: all inputs should be numpy arrays with shape = (6,)
        # Output: pose_mod and vel_mod compose the modified trajectory. they are numpy arrays with shape = (6,0)

        # Convert A,B matrices to discrete time state space matrices Ad,Bd
        if self.discrete_method == 'forward_differential':
            Ad = dt*self.A_imp+np.eye(self.A_imp.shape[0])
            Bd = dt*self.B_imp
            print('++++++++++ Ad and Bd +++++++++++++++++++++\n',Ad,'\nBd=',Bd)
        else:
            Ad = expm(dt*self.A_imp)
            Bd = self.A_imp_inv@(Ad-np.eye(12))@self.B_imp

        X = np.append(pose_mod, vel_mod)
        # Notice - when using the Vrot method for the rotation the moments should take with negative sign:
        # U = np.block([self.Lxf @ F_int[:3] + self.Lxm @ F_int[3:] + F0[:3], -self.LRf @ F_int[:3] - self.LRm @ F_int[3:] - F0[3:],
        #               np.array(pose_ref), np.array(vel_ref)])
        #U = np.block([F_int[:3] + F0[:3], -(F_int[3:] + F0[3:]),np.array(pose_ref), np.array(vel_ref)])
        U = np.block([F_int - F0, np.array(pose_ref), np.array(vel_ref)])
        X_next = np.dot(Ad, X)+np.dot(Bd, U)
        #print(f'X = {X}\nAd*X = {np.dot(Ad, X)}\nU = {U}\nBd*U = {np.dot(Bd, U)}\nX_next = {X_next}')

        return X_next

    def Force_controler(self,F_external,F_internal,F_internal_init,F0):
        # "F_int" is the interaction forces that should be provided be some external sensors
        # The "compensation_forces" could be the current force reading or the initial force reading of the internal sensor
        # Input Dimensions: all inputs should be numpy arrays with shape = (6,) or a lists with a length of 6
        # Output: "wrench_command" is numpy array with shape = (6,0) that should be sent to the robot when using force mode

        force_x = self.P_fc[0, 0] * F_internal[0] + self.P_fc[0, 4] * F_external[4] -F_internal_init[0]  # Should use F_external[0]
        force_y = self.P_fc[1, 1] * F_external[1] + self.P_fc[1, 3] * F_external[3] -F_internal_init[1]
        force_z = self.P_fc[2, 2] * F_external[2] - F_internal_init[2]  # Using Fz of the impedance controller is a fine option
        moments_commands = np.array([self.P_fc[3, 3] * F_external[3] + self.P_fc[3, 1] * F_external[1],
                                     self.P_fc[4, 4] * F_external[4] + self.P_fc[4, 0] * F_external[0], 0]) - (
                                       F_internal[3:] + F_internal_init[3:])  # - force_reading[3:]

        wrench_command = np.block([force_x, force_y, force_z, moments_commands]) + F0
        return wrench_command

    def idial_impedance(self,pose_real,pose_ref,pose_error):
        xy_des = pose_ref[:2]-pose_error[:2]
        delta_yx = pose_real[[1,0]] - xy_des[[1,0]]
        RxRy_des = (0.5*np.pi/180)*delta_yx/(np.array([0.002,-0.0012])) # where x is the deviation from goal 'delta_yx'. when full error accur theta=8[deg]
        delta_RxRy = pose_real[[3,4]] - RxRy_des
        delta_Rz = pose_real[5]-pose_ref[5]
        delta_z = pose_real[2]-pose_ref[2]
        delta_pose = np.block([delta_yx[[1,0]],delta_z,delta_RxRy,delta_Rz])
        vel_mod = -(1/0.2)*delta_pose
        pose_des = pose_real-delta_pose
        pose_mod = pose_des
        return np.append(pose_mod, vel_mod)
