"""
UR robot uses an axis-angle convention to describe the orientation Rx,Ry,Rz i.e rotation vector
To implement a minimum-jerk trajectory we need to convert the angle to Euler angles
Notion used is "RPY" roll-pitch-yaw convention i.e. XYZ Euler representation
http://web.mit.edu/2.05/www/Handout/HO2.PDF
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
Last modified: 02.08.2020
Daniel Stankowski and Elad Newman
"""

from scipy.spatial.transform import Rotation as R
import numpy as np


def Robot2Euler(orientation):
    """
    Convert axis-angle to euler xyz convention
    :param orient_array: np.array([Rx,Ry,Rz]) from the robot pose
    :return: euler angles in [rad]
    """

    temp = R.from_rotvec(orientation)
    euler = temp.as_euler("xyz", degrees=False)
    return np.array(euler)


def Euler2Robot(euler_angles):
    """
    Convert euler zyx angle to axis-angle
    :param: array of euler angles in xyz convention
    :return:  np.array([Rx,Ry,Rz])
    """
    temp2 = R.from_euler('xyz', euler_angles, degrees=False)
    axis_angles = temp2.as_rotvec()
    return np.array(axis_angles)


def Axis2Vector(axis_angles):
    """
    Convert axis-angle representation to the rotation vector form
    :param axis_angles: [Rx,Ry,Rz]
    :return: rot = [theta*ux,theta*uy,theta*uz] where:
    size is "theta"
    direction [ux,uy,uz] is a rotation vector
    """
    # axis_deg = np.rad2deg(axis_angles)
    size = np.linalg.norm(axis_angles)  # np.linalg.norm(axis_deg)
    if size > 1e-8:
        direction = axis_angles / size  # axis_deg/size
    else:
        direction = np.array([0,0,0])
    return size, direction


def Rot_matrix(angle, axis):
    if axis == 'x':
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    raise RuntimeError('!!Error in Rot_matrix(angle,axis): axis must take the values: "x","y",or "z" as characters!!')
    return 'Error in Rot_matrix(angle,axis)'


def AxisAngle_To_RotationVector(angles_current, angels_desired):
    """
    Outputs rotation vector mapping EEF orientation to the orientation in the base frame
    """

    R01 = R.from_rotvec(angles_current).as_matrix()  # rotation from 0 to 1 in (0) world frame
    R10 = R01.T
    R02 = R.from_rotvec(angels_desired).as_matrix() # rotation from 0 to 1 in (0) world frame
    R12 = np.dot(R10, R02) # rotation from 0 to 1 in (0) world frame

    Vrot1 = R.from_matrix(R12).as_rotvec() # rot vec from 1 to 2 in (1) current frame
    Vrot0 = np.dot(R01, Vrot1)  # rot vec from 1 to 2 in (0) base frame
    return Vrot0


def Rot_marix_to_axis_angles(Rot_matrix):
    Rotvec = R.from_matrix(Rot_matrix).as_rotvec()
    return Rotvec

def Gripper2Base_matrix(axis_angles_reading):
    R0t = R.from_rotvec(axis_angles_reading).as_matrix()
    return R0t

def Base2Gripper_matrix(axis_angles_reading):
    Rt0 = R.from_rotvec(axis_angles_reading).as_matrix().T
    return Rt0

def Tool2Base_vec(axis_angles_reading,vector):
    # "vector" is the vector that one would like to transform from Tool coordinate sys to the Base coordinate sys
    R0t = R.from_rotvec(axis_angles_reading).as_matrix()
    return R0t@vector


def Base2Tool_vec(axis_angles_reading,vector):
    # "vector" is the vector that one would like to transform from Base coordinate sys to the Tool coordinate sys.
    Rt0 = (R.from_rotvec(axis_angles_reading).as_matrix()).T
    return Rt0@vector


def Tool2Base_multiple_vectors(axis_angles_reading,matrix):
    # "matrix" is matrix with all the vectors the one want to translate from Tool sys to Base sys.
    # matrix.shape should be nX3, when n is any real number of vectors.
    R0t = R.from_rotvec(axis_angles_reading).as_matrix()
    return (R0t@(matrix.T)).T


def Base2Tool_multiple_vectors(axis_angles_reading,matrix):
    # "matrix" is matrix with all the vectors the one want to translate from Base sys to Tool sys.
    # matrix.shape should be nX3, when n is any real number of vectors
    Rt0 = (R.from_rotvec(axis_angles_reading).as_matrix()).T
    return (Rt0@(matrix.T)).T

def Base2Tool_sys_converting(coordinate_sys,pose_real,pose_ref,vel_real,vel_ref,F_internal,F_external,force_reading):
    # "coordinate_sys" is the axis_angle vector which represent the rotation vector between Base sys to Tool sys.

    REAL_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys, np.block(
        [[pose_real[:3]], [pose_real[3:]], [vel_real[:3]], [vel_real[3:]]]))
    [pose_real[:3], pose_real[3:], vel_real[:3], vel_real[3:]] = [REAL_DATA_tool[0], REAL_DATA_tool[1],
                                                                  REAL_DATA_tool[2], REAL_DATA_tool[3]]
    REF_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys, np.block(
        [[pose_ref[:3]], [pose_ref[3:]], [vel_ref[:3]], [vel_ref[3:]]]))
    [pose_ref[:3], pose_ref[3:], vel_ref[:3], vel_ref[3:]] = [REF_DATA_tool[0], REF_DATA_tool[1], REF_DATA_tool[2], REF_DATA_tool[3]]

    FORCE_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys,
                                                    np.block([[force_reading[:3]], [F_internal[:3]], [F_external[:3]]]))
    [force_reading[:3], F_internal[:3], F_external[:3]] = [FORCE_DATA_tool[0], FORCE_DATA_tool[1], FORCE_DATA_tool[2]]

    MOMENT_DATA_tool = Base2Tool_multiple_vectors(coordinate_sys, np.block(
        [[force_reading[3:]], [F_internal[3:]], [F_external[3:]]]))
    [force_reading[3:], F_internal[3:], F_external[3:]] = [MOMENT_DATA_tool[0], MOMENT_DATA_tool[1],
                                                           MOMENT_DATA_tool[2]]
    return pose_real,pose_ref,vel_real,vel_ref,F_internal,F_external,force_reading
