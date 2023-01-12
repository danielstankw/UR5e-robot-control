import numpy as np
import pandas as pd
from scipy.linalg import expm
from copy import deepcopy


class Controller(object):
    def __init__(self, control_dim):
        self.K = None
        self.C = None
        self.M = None
        # initialize K,C,M parameters
        self.control_dim = control_dim
        # TODO 26 dim
        action = np.loadtxt('action_1.csv', delimiter=',')
        self.set_control_param(action=action)

    def impedance_equation(self, pose_ref, vel_ref, pose_mod, vel_mod, f_int, f0, dt):
        """
        Impedance Eq: F_int-F0=K(x0-xm)+C(x0_d-xm_d)-Mxm_dd
        Solving the impedance equation for x(k+1)=Ax(k)+Bu(k) where
        x(k+1)=[Xm,thm,Xm_d,thm_d]
        Parameters:
            x0,x0_d,th0,th0_d - desired goal position/orientation and velocity
            F_int - measured force/moments in [N/Nm] (what the robot sense)
            F0 - desired applied force/moments (what the robot does)
            xm_pose - impedance model (updated in a loop) initialized at the initial pose of robot
            A_d, B_d - A and B matrices of x(k+1)=Ax(k)+Bu(k)
        Output:
            X_nex = x(k+1) = [Xm,thm,Xm_d,thm_d]
        """
        # state space formulation
        # X=[xm;thm;xm_d;thm_d] U=[F_int;M_int;x0;th0;x0d;th0d]
        A_1 = np.concatenate((np.zeros([6, 6], dtype=int), np.identity(6)), axis=1)
        A_2 = np.concatenate((np.dot(-np.linalg.pinv(self.M), self.K), np.dot(-np.linalg.pinv(self.M), self.C)), axis=1)
        A_temp = np.concatenate((A_1, A_2), axis=0)

        B_1 = np.zeros([6, 18], dtype=int)
        B_2 = np.concatenate((np.linalg.pinv(self.M), np.dot(np.linalg.pinv(self.M), self.K),
                              np.dot(np.linalg.pinv(self.M), self.C)), axis=1)
        B_temp = np.concatenate((B_1, B_2), axis=0)

        if np.isnan(A_temp).any() or np.isnan(B_temp).any():
            s = 1
        # discrete state space A, B matrices
        A_d = expm(A_temp * dt)
        B_d = np.dot(np.dot(np.linalg.pinv(A_temp), (A_d - np.identity(A_d.shape[0]))), B_temp)
        # reference
        x0 = pose_ref[:3].reshape(3, 1)
        th0 = pose_ref[3:].reshape(3, 1)
        x0_d = vel_ref[:3].reshape(3, 1)
        th0_d = vel_ref[3:].reshape(3, 1)

        # impedance model xm is initialized to initial position of the EEF and modified by force feedback
        xm = pose_mod[:3].reshape(3, 1)
        thm = pose_mod[3:].reshape(3, 1)
        xm_d = vel_mod[:3].reshape(3, 1)
        thm_d = vel_mod[3:].reshape(3, 1)

        # State Space vectors
        X = np.concatenate((xm, thm, xm_d, thm_d), axis=0)  # 12x1 column vector

        F = (-f0 + f_int).reshape(6, 1)
        U = np.concatenate((F, x0, th0, x0_d, th0_d), axis=0).reshape(18, 1)
        # discrete state solution X(k+1)=Ad*X(k)+Bd*U(k)
        X_nex = np.dot(A_d, X) + np.dot(B_d, U)
        # print(X_nex[9:12])
        return X_nex.reshape(12, )

    def set_control_param(self, action):

        if self.control_dim == 36:
            self.K = np.array([[action[0], 0, 0, 0, action[1], 0],
                               [0, action[2], 0, action[3], 0, 0],
                               [0, 0, action[4], 0, 0, 0],
                               [0, action[5], 0, action[6], 0, 0],
                               [action[7], 0, 0, 0, action[8], 0],
                               [0, 0, 0, 0, 0, action[9]]])

            self.C = np.array([[action[10], 0, 0, 0, action[11], 0],
                               [0, action[12], 0, action[13], 0, 0],
                               [0, 0, action[14], 0, 0, 0],
                               [0, action[15], 0, action[16], 0, 0],
                               [action[17], 0, 0, 0, action[18], 0],
                               [0, 0, 0, 0, 0, action[19]]])

            self.M = np.array([[action[20], 0, 0, 0, action[21], 0],
                               [0, action[22], 0, action[23], 0, 0],
                               [0, 0, action[24], 0, 0, 0],
                               [0, action[25], 0, action[26], 0, 0],
                               [action[27], 0, 0, 0, action[28], 0],
                               [0, 0, 0, 0, 0, action[29]]])

        if self.control_dim == 24:
            self.K = np.array([[action[0], 0, 0, 0, 0, 0],
                               [0, action[1], 0, 0, 0, 0],
                               [0, 0, action[2], 0, 0, 0],
                               [0, 0, 0, action[3], 0, 0],
                               [0, 0, 0, 0, action[4], 0],
                               [0, 0, 0, 0, 0, action[5]]])

            self.C = np.array([[action[6], 0, 0, 0, 0, 0],
                               [0, action[7], 0, 0, 0, 0],
                               [0, 0, action[8], 0, 0, 0],
                               [0, 0, 0, action[9], 0, 0],
                               [0, 0, 0, 0, action[10], 0],
                               [0, 0, 0, 0, 0, action[11]]])

            self.M = np.array([[action[12], 0, 0, 0, 0, 0],
                               [0, action[13], 0, 0, 0, 0],
                               [0, 0, action[14], 0, 0, 0],
                               [0, 0, 0, action[15], 0, 0],
                               [0, 0, 0, 0, action[16], 0],
                               [0, 0, 0, 0, 0, action[17]]])

        if self.control_dim == 26:
            self.K = np.array([[abs(action[0]), 0, 0, 0, action[1], 0],
                               [0, abs(action[2]), 0, action[3], 0, 0],
                               [0, 0, abs(action[4]), 0, 0, 0],
                               [0, action[5], 0, abs(action[6]), 0, 0],
                               [action[7], 0, 0, 0, abs(action[8]), 0],
                               [0, 0, 0, 0, 0, abs(action[9])]])

            self.C = np.array([[abs(action[10]), 0, 0, 0, action[11], 0],
                               [0, abs(action[12]), 0, action[13], 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, action[15], 0, abs(action[16]), 0, 0],
                               [action[17], 0, 0, 0, abs(action[18]), 0],
                               [0, 0, 0, 0, 0, abs(action[19])]])

            self.M = np.array([[abs(action[20]), 0, 0, 0, 0, 0],
                               [0, abs(action[21]), 0, 0, 0, 0],
                               [0, 0, abs(action[22]), 0, 0, 0],
                               [0, 0, 0, abs(action[23]), 0, 0],
                               [0, 0, 0, 0, abs(action[24]), 0],
                               [0, 0, 0, 0, 0, abs(action[25])]])

        if self.control_dim == 18:
            self.K = np.array([[abs(action[0]), 0, 0, 0, 0, 0],
                               [0, abs(action[1]), 0, 0, 0, 0],
                               [0, 0, abs(action[2]), 0, 0, 0],
                               [0, 0, 0, abs(action[3]), 0, 0],
                               [0, 0, 0, 0, abs(action[4]), 0],
                               [0, 0, 0, 0, 0, abs(action[5])]])

            self.C = np.array([[abs(action[6]), 0, 0, 0, 0, 0],
                               [0, abs(action[7]), 0, 0, 0, 0],
                               [0, 0, abs(action[8]), 0, 0, 0],
                               [0, 0, 0, abs(action[9]), 0, 0],
                               [0, 0, 0, 0, abs(action[10]), 0],
                               [0, 0, 0, 0, 0, abs(action[11])]])

            self.M = np.array([[abs(action[12]), 0, 0, 0, 0, 0],
                               [0, abs(action[13]), 0, 0, 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, 0, 0, abs(action[15]), 0, 0],
                               [0, 0, 0, 0, abs(action[16]), 0],
                               [0, 0, 0, 0, 0, abs(action[17])]])

        if self.control_dim == 30:
            self.K = np.array([[abs(action[0]), 0, 0, 0, action[1], 0],
                               [0, abs(action[2]), 0, action[3], 0, 0],
                               [0, 0, abs(action[4]), 0, 0, 0],
                               [0, action[5], 0, abs(action[6]), 0, 0],
                               [action[7], 0, 0, 0, abs(action[8]), 0],
                               [0, 0, 0, 0, 0, abs(action[9])]])

            self.C = np.array([[abs(action[10]), 0, 0, 0, action[11], 0],
                               [0, abs(action[12]), 0, action[13], 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, action[15], 0, abs(action[16]), 0, 0],
                               [action[17], 0, 0, 0, abs(action[18]), 0],
                               [0, 0, 0, 0, 0, abs(action[19])]])

            self.M = np.array([[abs(action[20]), 0, 0, 0, action[21], 0],
                               [0, abs(action[22]), 0, action[23], 0, 0],
                               [0, 0, abs(action[24]), 0, 0, 0],
                               [0, action[25], 0, abs(action[26]), 0, 0],
                               [action[27], 0, 0, 0, abs(action[28]), 0],
                               [0, 0, 0, 0, 0, abs(action[29])]])

        if self.control_dim == 20:
            self.K = np.array([[abs(action[0]), 0, 0, 0, action[1], 0],
                               [0, abs(action[2]), 0, action[3], 0, 0],
                               [0, 0, abs(action[4]), 0, 0, 0],
                               [0, action[5], 0, abs(action[6]), 0, 0],
                               [action[7], 0, 0, 0, abs(action[8]), 0],
                               [0, 0, 0, 0, 0, abs(action[9])]])

            self.C = np.array([[abs(action[10]), 0, 0, 0, action[11], 0],
                               [0, abs(action[12]), 0, action[13], 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, action[15], 0, abs(action[16]), 0, 0],
                               [action[17], 0, 0, 0, abs(action[18]), 0],
                               [0, 0, 0, 0, 0, abs(action[19])]])

            self.M = np.array([[44.93156433, 0., 0., 0., 0., 0.],
                               [0., 53.0908432, 0., 0., 0., 0.],
                               [0., 0., 5.83020163, 0., 0., 0.],
                               [0., 0., 0., 131.73561096, 0., 0.],
                               [0., 0., 0., 0., 7.31005669, 0.],
                               [0., 0., 0., 0., 0., 71.55409241]])


