import numpy as np


action = np.loadtxt('/home/danieln7/Desktop/RobotCode2023/daniel_learning_runs/run3/action/scaled_params.csv',delimiter=',')

K = np.array([[action[0], 0, 0, 0, action[1], 0],
                                   [0, action[2], 0, action[3], 0, 0],
                                   [0, 0, action[4], 0, 0, 0],
                                   [0, action[5], 0, action[6], 0, 0],
                                   [action[7], 0, 0, 0, action[8], 0],
                                   [0, 0, 0, 0, 0, action[9]]])

C = np.array([[action[10], 0, 0, 0, action[11], 0],
                                   [0, action[12], 0, action[13], 0, 0],
                                   [0, 0, action[14], 0, 0, 0],
                                   [0, action[15], 0, action[16], 0, 0],
                                   [action[17], 0, 0, 0, action[18], 0],
                                   [0, 0, 0, 0, 0, action[19]]])

M = np.array([[action[20], 0, 0, 0, 0, 0],
                                   [0, action[21], 0, 0, 0, 0],
                                   [0, 0, action[22], 0, 0, 0],
                                   [0, 0, 0, action[23], 0, 0],
                                   [0, 0, 0, 0, action[24], 0],
                                   [0, 0, 0, 0, 0, action[25]]])

print(K)
print(C)
print(M)
print('---------------------')
K_load = np.loadtxt('/home/danieln7/Desktop/RobotCode2023/daniel_learning_runs/run3/action/K.csv',delimiter=',')
C_load = np.loadtxt('/home/danieln7/Desktop/RobotCode2023/daniel_learning_runs/run3/action/C.csv',delimiter=',')
M_load = np.loadtxt('/home/danieln7/Desktop/RobotCode2023/daniel_learning_runs/run3/action/M.csv',delimiter=',')
print(K_load)
print(C_load)
print(M_load)