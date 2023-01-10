import numpy as np
from numpy import linalg as LA


class Control_param_tube:
    def __init__(self):

        # - - - - - - PD controller parameters - - - - - - -
        zeta_pd = 0.707
        kp_f = 1100  # 2500
        kd_f = 2 * zeta_pd * np.sqrt(kp_f)
        kp_m = 45  # 10 #
        kd_m = 2 * zeta_pd * np.sqrt(kp_m)

        self.Kf_pd = np.identity(3) * kp_f
        self.Cf_pd = np.identity(3) * kd_f
        self.Km_pd = np.identity(3) * kp_m
        self.Cm_pd = np.identity(3) * kd_m

        self.Kf_pd[0, 0] = 0.7*500 #1300 #600 #1100  # 800
        self.Cf_pd[0, 0] = 2 * zeta_pd * np.sqrt(self.Kf_pd[0, 0])
        self.Kf_pd[1,1] = 0.7*500 #1300 #1300 #900 #800
        self.Cf_pd[1,1] = 2 * zeta_pd * np.sqrt(self.Kf_pd[1,1])
        self.Kf_pd[2, 2] = 500 #500 #600 #900  # 800
        self.Cf_pd[2, 2] = 5*2 * zeta_pd * np.sqrt(self.Kf_pd[2, 2])

        self.Km_pd[0, 0] = 180 #200 #70 #150
        self.Cm_pd[0, 0] = 0.5*2 * zeta_pd * np.sqrt(self.Km_pd[0, 0])
        self.Km_pd[1, 1] = 180  #200 #70 #85
        self.Cm_pd[1, 1] = 0.5*2 * zeta_pd * np.sqrt(self.Km_pd[1, 1])
        self.Km_pd[2, 2] = 180 #200 #70
        self.Cm_pd[2, 2] = 0.5*2 * zeta_pd * np.sqrt(self.Km_pd[2, 2])

        # - - - - - - Impedance parameters - - - - - - - - - -
        zeta_imp = 0.707
        wn = 15  # 15 #30  # 40
        mm = 20  # 5
        kk = mm * (wn ** 2)
        cc = zeta_imp * 2 * np.sqrt(kk * mm)

        self.M_imp = np.identity(6) * mm
        self.K_imp = np.identity(6) * kk
        self.C_imp = np.identity(6) * cc

        # Parameters that i found without simulation in 7.2021 (wnxy=8, wnRxy=8,kRx_imp = 140, kRy_imp = 1440)
        # self.Lxf = np.diag([10, 10, 1]) #np.diag([6, 5, 1]) #np.diag([7.5, 6, 1])
        # self.Lxm = np.array([[0, -4, 0], [4, 0, 0], [0, 0, 0]]) #np.array([[0, 1, 0], [-2.5, 0, 0], [0, 0, 0]])
        # self.LRm = np.diag([1,1,1])
        # self.LRf = np.array([[0, -10, 0], [4, 0, 0], [0, 0, 0]]) #np.array([[0, -1, 0], [0.5, 0, 0], [0, 0, 0]])

        # Parameters that i found in the simulation in 12.9.2021 (wnxy=26, wnRxy=18,kRx_imp = 140, kRy_imp = 140)
        self.Lxf = -1*1.1*np.diag([1.5, 1.5, 1.5])  #-0.6
        self.Lxm = -1*-1.1*190/310 * np.array([[0, 10, 0], [-10, 0, 0], [0, 0, 0]])  #-0.6
        self.LRm = -2.3*190/310 * np.diag([1,1,1])
        self.LRf = 2.2*-1.5*np.array([[0, 0.5, 0], [-0.5, 0, 0], [0, 0, 0]])

        self.L_imp = np.block([[self.Lxf,self.Lxm],[self.LRf,self.LRm]])*np.array([[1,1,1,-1,-1,-1]]).T
        # self.F0_imp = np.array([0,0,0,0,0,0])    # F0 are some desired F/T that one can set. notice that F0 are the applied F/T and not the measured F/T

        # mmx = 10
        wnx = 20 #26 #8 #15
        zeta_imp_x = 3 #4*1  # 1
        kx_imp = 350
        self.M_imp[0, 0] = kx_imp / wnx ** 2
        self.K_imp[0, 0] = kx_imp
        self.C_imp[0, 0] = zeta_imp_x * 2 * np.sqrt(self.K_imp[0, 0] * self.M_imp[0, 0])

        #mmy = 2 #0.2
        wny = 20 #26 #8 #15 #10 #21
        zeta_imp_y = 3 #4*1 #1
        ky_imp = 350
        self.M_imp[1, 1] = ky_imp/wny**2
        self.K_imp[1, 1] = ky_imp
        self.C_imp[1, 1] = zeta_imp_y * 2 * np.sqrt(self.K_imp[1, 1] * self.M_imp[1, 1])

        wnz = 20 #15 #25
        zeta_imp_z = 1  # 1
        kz_imp = 7000 #2400
        self.M_imp[2, 2] = kz_imp / wnz ** 2
        self.K_imp[2, 2] = kz_imp
        self.C_imp[2, 2] = zeta_imp_z * 2 * np.sqrt(self.K_imp[2, 2] * self.M_imp[2, 2])

        wnRx = 10 #18 #8 #5 #5 #10  # 21 #17 #13
        zeta_imp_Rx = 3 #3*1  # 1
        kRx_imp = 140
        self.M_imp[3, 3] = kRx_imp / wnRx ** 2
        self.K_imp[3, 3] = kRx_imp
        self.C_imp[3, 3] = zeta_imp_Rx * 2 * np.sqrt(self.K_imp[3, 3] * self.M_imp[3, 3])

        wnRy = 10 #18 #8 #5 #10  # 21 #17 #13
        zeta_imp_Ry = 3 #3*1  # 1
        kRy_imp = 140 # i used 1440 instead of 140 by mistake. try to use 140 and see if it works
        self.M_imp[4, 4] = kRy_imp / wnRy ** 2
        self.K_imp[4, 4] = kRy_imp
        self.C_imp[4, 4] = zeta_imp_Ry * 2 * np.sqrt(self.K_imp[4, 4] * self.M_imp[4, 4])

        wnRz = 20 #18 #25  # 21 #17 #13
        zeta_imp_Rz = 3 #3 #2  # 1
        kRz_imp = 140 #1
        self.M_imp[5, 5] = kRz_imp / wnRz ** 2
        self.K_imp[5, 5] = kRz_imp
        self.C_imp[5, 5] = zeta_imp_Rz * 2 * np.sqrt(self.K_imp[5, 5] * self.M_imp[5, 5])

        # Convert decoupled (with L matrix) represantation to coupled M,D,K represantation
        Limp_inv = LA.inv(self.L_imp)  #*np.array([[1,1,1,-1,-1,-1]]).T
        self.K_imp = Limp_inv@self.K_imp
        self.C_imp = Limp_inv@self.C_imp
        self.M_imp = Limp_inv@self.M_imp

        # - - - - - - Matrices for Impedance equation - - - - - - - - - -
        M_imp_inv = LA.inv(self.M_imp)
        self.A_imp = np.block([[np.zeros([6, 6]), np.eye(6)], [-M_imp_inv @ self.K_imp, -M_imp_inv @ self.C_imp]])
        self.B_imp = np.block([[np.zeros([6, 18])], [M_imp_inv, M_imp_inv @ self.K_imp, M_imp_inv @ self.C_imp]])
        self.A_imp_inv = LA.inv(self.A_imp)

        # - - - - - - Force Controller Parameters - - - - - - - - - - - - -
        self.P_fc = np.zeros([6,6])
        self.P_fc[0,0],self.P_fc[0,4]  = 7.5, 11.5
        self.P_fc[1,1], self.P_fc[1,3] = 6, -2.5
        self.P_fc[2,2] = 0.5
        self.P_fc[3,1], self.P_fc[3,3] = -0.5, 0.5
        self.P_fc[4,0], self.P_fc[4,4] = 0.6, 0.5
        self.P_fc[5,5] = 0

if __name__ == '__main__':
    params = Control_param_tube()
    K_eigvals, K_eigvecs = LA.eig(params.K_imp)
    C_eigvals, C_eigvecs = LA.eig(params.C_imp)
    M_eigvals, M_eigvecs = LA.eig(params.M_imp)
    A_eigvals, A_eigvecs = LA.eig(params.A_imp)
    Kinv_eigvals, Kinv_eigvecs = LA.eig(LA.inv(params.K_imp))

    print('finish eig_vals')
    # print(K_eigvecs[:,-1]@params.K_imp@K_eigvecs[:,-1]/(LA.norm(K_eigvecs[:,-1])**2))
    # print(K_eigvecs[:,3]@params.K_imp@K_eigvecs[:,3]/(LA.norm(K_eigvecs[:,3])**2))
    # print('A_eigvals =\n',A_eigvals.reshape([len(A_eigvals),1]))
    # print('X')

    # To save parameters uncomment the lines below:
    # np.save('Saved Parameters\\20mm peg params\K_imp', params.K_imp)
    # np.save('Saved Parameters\\20mm peg params\C_imp', params.C_imp)
    # np.save('Saved Parameters\\20mm peg params\M_imp', params.M_imp)
    # np.save('Saved Parameters\\20mm peg params\Kf_pd', params.Kf_pd)
    # np.save('Saved Parameters\\20mm peg params\Km_pd', params.Km_pd)
    # np.save('Saved Parameters\\20mm peg params\Cf_pd', params.Cf_pd)
    # np.save('Saved Parameters\\20mm peg params\Cm_pd', params.Cm_pd)
    # np.save('Saved Parameters\\20mm peg params\F0', np.array([0,0,-5,0,0,0]))


