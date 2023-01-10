"""
Minimum jerk trajectory for 6DOF robot
Latest update: 18.08.2021
Written by Elad Newman and Daniel Stankowski
"""
import numpy as np
from matplotlib import pyplot as plt
import angle_transformation as at
from scipy.spatial.transform import Rotation as R


class PathPlan(object):
    """
    """

    def __init__(self, initial_pose, target_pose, total_time, orientation_method = 'decreasing Vrot'):
        # Notice that the orientation inputs (i.e. initial_pose[3:] and target_pose[3:]) should be in axis-angles
        # representation!!!

        # Compute initial and target orientation according to the chosen "orientation_method" value:
        if orientation_method=='decreasing Vrot':
            self.initial_orientation = at.RotationVector(initial_pose[3:], target_pose[3:])
            self.target_orientation = np.zeros(3)
        elif orientation_method=='increasing Vrot':
            self.initial_orientation = np.zeros(3)
            self.target_orientation = at.RotationVector(initial_pose[3:], target_pose[3:])
        elif orientation_method=='euler':
            self.initial_orientation = R.from_rotvec(initial_pose[3:]).as_euler("zyx", degrees=False)
            self.target_orientation = R.from_rotvec(target_pose[3:]).as_euler("zyx", degrees=False)
            # insure shortest path (i.e. rotate not more than 180 degrees):
            indexes_for_correction = np.abs(self.target_orientation-self.initial_orientation) > np.pi
            correction = np.sign(self.target_orientation)*(2*np.pi)*indexes_for_correction
            self.target_orientation = self.target_orientation-correction
        else:
            raise ValueError('the argument "orientation_method" should take one of the following values: "decreasing Vrot", "increasing Vrot" or "euler"')

        self.initial_position = initial_pose[:3]
        self.target_position = target_pose[:3]
        self.tfinal = total_time
        self.ori_method = orientation_method

        # Compute the polynomial coefficients
        # position coefficients:
        delta_div_t3 = (self.target_position - self.initial_position)/ (self.tfinal ** 3)
        self.a5 = 6*delta_div_t3 / (self.tfinal ** 2)
        self.a4 = -15*delta_div_t3 / self.tfinal
        self.a3 = 10*delta_div_t3
        self.a0 = self.initial_position
        # orientation coefficients:
        delta_ori_div_t3 = (self.target_orientation - self.initial_orientation)/ (self.tfinal ** 3)
        self.a5_ori = 6*delta_ori_div_t3 / (self.tfinal ** 2)
        self.a4_ori = -15*delta_ori_div_t3 / self.tfinal
        self.a3_ori = 10*delta_ori_div_t3
        self.a0_ori = self.initial_orientation


    def trajectory_planning(self, t):
        position = self.a5*t**5 + self.a4*t**4 + self.a3*t**3 + self.a0
        lin_vel = 5*self.a5*t**4 + 4*self.a4*t**3 + 3*self.a3*t**2
        orientation = self.a5_ori*t**5 + self.a4_ori*t**4 + self.a3_ori*t**3 + self.a0_ori
        ang_vel = 5*self.a5_ori*t**4 + 4*self.a4_ori*t**3 + 3*self.a3_ori*t**2

        return position, orientation, lin_vel, ang_vel


# ******* Main Script (plot the min_jerk trajectory of some initial and target pose) **********
if __name__ == "__main__":
    initial_pose = np.array([-0.1779, -0.4952, 0.2933, -1.0342, -2.9657,  0.0176])
    target_pose = np.array([0.1845, -0.04, 0.1353, 2.5303,  -2.5, -0.3])

    tfinal = 5
    orientation_method = 'euler' #'decreasing Vrot' #'increasing Vrot' #'euler'
    trajectory = PathPlan(initial_pose, target_pose, tfinal,orientation_method)

    time_vec = []
    posx = []
    posy = []
    posz = []
    ori_x = []
    ori_y = []
    ori_z = []
    lin_vel_plot = np.empty([0,3])
    ang_vel_plot = np.empty([0,3])



    for t in np.linspace(0,tfinal,100):
        #t = (i / 100) * tfinal
        position, orientation, lin_vel, ang_vel = trajectory.trajectory_planning(t)

        posx.append(position[0])
        posy.append(position[1])
        posz.append(position[2])
        ori_x.append(orientation[0])
        ori_y.append(orientation[1])
        ori_z.append(orientation[2])
        lin_vel_plot = np.vstack((lin_vel_plot,lin_vel))
        ang_vel_plot = np.vstack((ang_vel_plot,ang_vel))
        time_vec.append(t)

    # plot position:
    plt.figure('position')
    plt.title('position')
    plt.plot(0*np.zeros(3), initial_pose[:3], 'ko')
    plt.plot(tfinal*np.ones(3), target_pose[:3], 'ko')
    plt.plot(time_vec, posx, label='X position')
    plt.plot(time_vec, posy, label='Y position')
    plt.plot(time_vec, posz, label='Z position')
    plt.legend()
    plt.grid()
    plt.ylabel('Position [m]')
    plt.xlabel('Time [s]')

    # plot orientations:
    if orientation_method == 'euler':
        initial_pose[3:] = R.from_rotvec(initial_pose[3:]).as_euler("zyx", degrees=False)
        target_pose[3:] = R.from_rotvec(target_pose[3:]).as_euler("zyx", degrees=False)
    else:
        initial_pose[3:6] = np.array([ori_x[0],ori_y[0],ori_z[0]])
        target_pose[3:6] = np.array([ori_x[-1],ori_y[-1],ori_z[-1]])

    plt.figure('orientation')
    plt.title(f'orientation in "{orientation_method}" representation')
    plt.plot(0*np.zeros(3), initial_pose[3:], 'ko')
    plt.plot(tfinal, target_pose[3], marker='o', markerfacecolor='black', markeredgecolor='r', linestyle='None')
    plt.plot(tfinal, target_pose[4], marker='o', markerfacecolor='black', markeredgecolor='g', linestyle='None')
    plt.plot(tfinal, target_pose[5], marker='o', markerfacecolor='black', markeredgecolor='b', linestyle='None')
    plt.plot(time_vec, ori_x, 'r',label=r'$\theta_x$ rotation')
    plt.plot(time_vec, ori_y, 'g', label=r'$\theta_y$ rotation')
    plt.plot(time_vec, ori_z, 'b',label=r'$\theta_z$ rotation')
    plt.legend()
    plt.grid()
    plt.ylabel('Rotation [rad]')
    plt.xlabel('Time [s]')

    # plot linear velocity:
    plt.figure('linear velocity')
    plt.title('linear velocity')
    plt.plot(0*np.zeros(3), 0*np.zeros(3), 'ko')
    plt.plot(tfinal*np.ones(3), 0*np.zeros(3), 'ko')
    plt.plot(time_vec, lin_vel_plot[:,0], label=r'$V_x$ velocity')
    plt.plot(time_vec, lin_vel_plot[:,1], label=r'$V_y$ velocity')
    plt.plot(time_vec, lin_vel_plot[:,2], label=r'$V_z$ velocity')
    plt.legend()
    plt.grid()
    plt.ylabel(r'velocity $[m/sec]$')
    plt.xlabel('Time [s]')

    # plot angular velocity:
    plt.figure('angular velocity')
    plt.title('angular velocity')
    plt.plot(0*np.zeros(3), 0*np.zeros(3), 'ko')
    plt.plot(tfinal*np.ones(3), 0*np.zeros(3), 'ko')
    plt.plot(time_vec, ang_vel_plot[:,0], label=r'$\omega_x$ velocity')
    plt.plot(time_vec, ang_vel_plot[:,1], label=r'$\omega_y$ velocity')
    plt.plot(time_vec, ang_vel_plot[:,2], label=r'$\omega_z$ velocity')
    plt.legend()
    plt.grid()
    plt.ylabel(r'velocity $[m/sec]$')
    plt.xlabel('Time [s]')


    plt.show()
