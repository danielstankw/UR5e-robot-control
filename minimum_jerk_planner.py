import numpy as np
from matplotlib import pyplot as plt
import angle_transformation as at
from scipy.spatial.transform import Rotation as R


class PathPlan(object):
    """
    """

    def __init__(self, initial_pose, target_pose, total_time):
        # Orientation inputs (i.e. initial_pose[3:] and target_pose[3:]) should be in axis-angles representation!!!

        # our strategy is minimizing magnitude! (start = theta -> end=0)
        self.initial_orientation = at.AxisAngle_To_RotationVector(target_pose[3:], initial_pose[3:])
        self.target_orientation = np.zeros(3)
        self.t_final = total_time

        self.X_init = initial_pose[0]
        self.Y_init = initial_pose[1]
        self.Z_init = initial_pose[2]

        self.X_final = target_pose[0]
        self.Y_final = target_pose[1]
        self.Z_final = target_pose[2]

    def trajectory_planning(self, t):

        x_traj = (self.X_final - self.X_init) / (self.t_final ** 3) * (
                6 * (t ** 5) / (self.t_final ** 2) - 15 * (t ** 4) / self.t_final + 10 * (t ** 3)) + self.X_init
        y_traj = (self.Y_final - self.Y_init) / (self.t_final ** 3) * (
                6 * (t ** 5) / (self.t_final ** 2) - 15 * (t ** 4) / self.t_final + 10 * (t ** 3)) + self.Y_init
        z_traj = (self.Z_final - self.Z_init) / (self.t_final ** 3) * (
                6 * (t ** 5) / (self.t_final ** 2) - 15 * (t ** 4) / self.t_final + 10 * (t ** 3)) + self.Z_init
        position = np.array([x_traj, y_traj, z_traj])

        # velocities
        vx = (self.X_final - self.X_init) / (self.t_final ** 3) * (
                30 * (t ** 4) / (self.t_final ** 2) - 60 * (t ** 3) / self.t_final + 30 * (t ** 2))
        vy = (self.Y_final - self.Y_init) / (self.t_final ** 3) * (
                30 * (t ** 4) / (self.t_final ** 2) - 60 * (t ** 3) / self.t_final + 30 * (t ** 2))
        vz = (self.Z_final - self.Z_init) / (self.t_final ** 3) * (
                30 * (t ** 4) / (self.t_final ** 2) - 60 * (t ** 3) / self.t_final + 30 * (t ** 2))
        velocity = np.array([vx, vy, vz])

        # acceleration
        ax = (self.X_final - self.X_init) / (self.t_final ** 3) * (
                120 * (t ** 3) / (self.t_final ** 2) - 180 * (t ** 2) / self.t_final + 60 * t)
        ay = (self.Y_final - self.Y_init) / (self.t_final ** 3) * (
                120 * (t ** 3) / (self.t_final ** 2) - 180 * (t ** 2) / self.t_final + 60 * t)
        az = (self.Z_final - self.Z_init) / (self.t_final ** 3) * (
                120 * (t ** 3) / (self.t_final ** 2) - 180 * (t ** 2) / self.t_final + 60 * t)
        acceleration = np.array([ax, ay, az])

        # orientation
        Vrot = at.AxisAngle_To_RotationVector(self.initial_orientation, self.target_orientation)

        upper_bound = 1e-6
        if np.linalg.norm(Vrot) < upper_bound:
            magnitude_traj = 0.0
            magnitude_vel_traj = 0.0
            direction = np.array([0.0, 0.0, 0.0])
        else:
            magnitude, direction = at.Axis2Vector(Vrot)
            # Daniel: this is Shir convention that is why she multiplied everything by (-1)
            # we use decrease magnitude strategy t_start: theta -> t_end = 0
            magnitude_traj = magnitude / (self.t_final ** 3) * (
                    6 * (t ** 5) / (self.t_final ** 2) - 15 * (t ** 4) / self.t_final + 10 * (t ** 3)) - magnitude
            magnitude_vel_traj = magnitude / (self.t_final ** 3) * (
                    30 * (t ** 4) / (self.t_final ** 2) - 60 * (t ** 3) / self.t_final + 30 * (t ** 2))

        orientation = magnitude_traj * direction
        ang_vel = magnitude_vel_traj * direction

        return position, orientation, velocity, ang_vel


if __name__ == "__main__":
    # # no rotation
    # init_pose = np.array([-0.2554, -0.3408, 0.2068, 0.1136, -3.1317, -0.0571])
    # goal_pose = np.array([-1.2360, -1.5677, -1.4024, 0.1136, -3.1317, -0.0571])
    # small rotation
    init_pose = np.array([-0.2554, -0.3408, 0.2068, 0.1136, -3.1317, -0.0571])
    goal_pose = np.array([-0.1028, -0.4503, 0.2597, 0.9577, 2.9857, 0.0577])

    total_time = 10
    trajectory = PathPlan(init_pose, goal_pose, total_time)

    time_vec = []
    pos_x = []
    pos_y = []
    pos_z = []
    ori_x = []
    ori_y = []
    
    ori_z = []
    vel_x = []
    vel_y = []
    vel_z = []
    ang_vel_x = []
    ang_vel_y = []
    ang_vel_z = []

    for t in np.linspace(0, total_time, 100):
        # t = (i / 100) * tfinal
        position, orientation, lin_vel, ang_vel = trajectory.trajectory_planning(t)

        time_vec.append(t)
        pos_x.append(position[0])
        pos_y.append(position[1])
        pos_z.append(position[2])
        ori_x.append(orientation[0])
        ori_y.append(orientation[1])
        ori_z.append(orientation[2])
        vel_x.append(lin_vel[0])
        vel_y.append(lin_vel[1])
        vel_z.append(lin_vel[2])
        ang_vel_x.append(ang_vel[0])
        ang_vel_y.append(ang_vel[1])
        ang_vel_z.append(ang_vel[2])


    # plots:

    plt.figure('Position')
    plt.title('Position')
    plt.plot(time_vec, pos_x, label=f"X position {init_pose[0]}->{goal_pose[0]}")
    plt.plot(time_vec, pos_y, label=f"Y position {init_pose[1]}->{goal_pose[1]}")
    plt.plot(time_vec, pos_z, label=f"Z position {init_pose[2]}->{goal_pose[2]}")
    plt.legend()
    plt.grid()
    plt.ylabel('Position [m]')
    plt.xlabel('Time [s]')

    plt.figure('Orientation')
    plt.title("Orientation")
    plt.plot(time_vec, ori_x, 'r', label=r'$\theta_x$ rotation')
    plt.plot(time_vec, ori_y, 'g', label=r'$\theta_y$ rotation')
    plt.plot(time_vec, ori_z, 'b', label=r'$\theta_z$ rotation')
    plt.legend()
    plt.grid()
    plt.ylabel('Rotation [rad]')
    plt.xlabel('Time [s]')

    # plot linear velocity:
    plt.figure('linear velocity')
    plt.title('linear velocity')
    plt.plot(time_vec, vel_x, label=r'$V_x$ velocity')
    plt.plot(time_vec, vel_y, label=r'$V_y$ velocity')
    plt.plot(time_vec, vel_z, label=r'$V_z$ velocity')
    plt.legend()
    plt.grid()
    plt.ylabel(r'velocity $[m/sec]$')
    plt.xlabel('Time [s]')

    # plot angular velocity:
    plt.figure('angular velocity')
    plt.title('angular velocity')
    plt.plot(time_vec, ang_vel_x, label=r'$\omega_x$ velocity')
    plt.plot(time_vec, ang_vel_y, label=r'$\omega_y$ velocity')
    plt.plot(time_vec, ang_vel_z, label=r'$\omega_z$ velocity')
    plt.legend()
    plt.grid()
    plt.ylabel(r'velocity $[m/sec]$')
    plt.xlabel('Time [s]')

    plt.show()
