import numpy as np
from numpy import linalg as LA
import pickle

class Control_param_20mm_circular_peg:
    def __init__(self):

        is_print_params = True  #False

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

        # Shir's Project PD params:
        self.Kf_pd[0, 0] = 700  # 1300  # 600 #1100  # 800
        self.Cf_pd[0, 0] = 2 * zeta_pd * np.sqrt(self.Kf_pd[0, 0])
        self.Kf_pd[1, 1] = 700  # 1300  # 1300 #900 #800
        self.Cf_pd[1, 1] = 2 * zeta_pd * np.sqrt(self.Kf_pd[1, 1])
        self.Kf_pd[2, 2] = 200  # 500 #600 #900  # 800
        self.Cf_pd[2, 2] = 2 * zeta_pd * np.sqrt(self.Kf_pd[2, 2])  # * 5

        self.Km_pd[0, 0] = 50  # 70 #150
        self.Cm_pd[0, 0] = 2 * zeta_pd * np.sqrt(self.Km_pd[0, 0])
        self.Km_pd[1, 1] = 50  # 70 #85
        self.Cm_pd[1, 1] = 2 * zeta_pd * np.sqrt(self.Km_pd[1, 1])
        self.Km_pd[2, 2] = 50  # 70
        self.Cm_pd[2, 2] = 2 * zeta_pd * np.sqrt(self.Km_pd[2, 2])

        # # Elad PPO (not learned params):
        # self.Kf_pd = np.diag([2500, 2500, 2500])  #delete!!! #np.diag([2500, 2500, 2500])
        # self.Cf_pd = np.sqrt(2) * np.sqrt(self.Kf_pd)
        # self.Km_pd = np.diag([500, 500, 500])
        # self.Cm_pd = np.sqrt(2) * np.sqrt(self.Km_pd)

        # - - - - - - Impedance parameters - - - - - - - - - -

        # #* * * * * * *  Hand Tuned Params * * * * * * * :
        # zeta_imp = 0.707
        # wn = 15  # 15 #30  # 40
        # mm = 20  # 5
        # kk = mm * (wn ** 2)
        # cc = zeta_imp * 2 * np.sqrt(kk * mm)
        #
        # # self.M_imp = np.identity(6) * mm
        # self.K_imp = np.identity(6) * kk
        # self.C_imp = np.identity(6) * cc
        #
        # # Parameters that i found without simulation in 7.2021 (wnxy=8, wnRxy=8,kRx_imp = 140, kRy_imp = 1440)
        # # self.Lxf = np.diag([10, 10, 1]) #np.diag([6, 5, 1]) #np.diag([7.5, 6, 1])
        # # self.Lxm = np.array([[0, -4, 0], [4, 0, 0], [0, 0, 0]]) #np.array([[0, 1, 0], [-2.5, 0, 0], [0, 0, 0]])
        # # self.LRm = np.diag([1,1,1])
        # # self.LRf = np.array([[0, -10, 0], [4, 0, 0], [0, 0, 0]]) #np.array([[0, -1, 0], [0.5, 0, 0], [0, 0, 0]])
        #
        # # Parameters that i found in the simulation in 12.9.2021 (wnxy=26, wnRxy=18,kRx_imp = 140, kRy_imp = 140)
        # self.Lxf = 1.1 * np.diag([1.5, 1.5, 1.5])
        # self.Lxm = -1.1 * 190 / 310 * np.array([[0, 10, 0], [-10, 0, 0], [0, 0, 0]])
        # self.LRm = 1 * 190 / 310 * np.diag([1, 1, 1])
        # self.LRf = -1.5 * np.array([[0, 0.5, 0], [-0.5, 0, 0], [0, 0, 0]])
        #
        # self.L_imp = np.block([[self.Lxf, self.Lxm], [self.LRf, self.LRm]]) * np.array([[1, 1, 1, -1, -1, -1]]).T
        # # self.F0_imp = np.array([0,0,0,0,0,0])    # F0 are some desired F/T that one can set. notice that F0 are the applied F/T and not the measured F/T
        #
        # # mmx = 10
        # wnx = 26  # 8 #15
        # zeta_imp_x = 4 * 1  # 1
        # kx_imp = 350
        # self.M_imp[0, 0] = kx_imp / wnx ** 2
        # self.K_imp[0, 0] = kx_imp
        # self.C_imp[0, 0] = zeta_imp_x * 2 * np.sqrt(self.K_imp[0, 0] * self.M_imp[0, 0])
        #
        # # mmy = 2 #0.2
        # wny = 26  # 8 #15 #10 #21
        # zeta_imp_y = 4 * 1  # 1
        # ky_imp = 350
        # self.M_imp[1, 1] = ky_imp / wny ** 2
        # self.K_imp[1, 1] = ky_imp
        # self.C_imp[1, 1] = zeta_imp_y * 2 * np.sqrt(self.K_imp[1, 1] * self.M_imp[1, 1])
        #
        # wnz = 20  # 15 #25
        # zeta_imp_z = 1  # 1
        # kz_imp = 7000  # 2400
        # self.M_imp[2, 2] = kz_imp / wnz ** 2
        # self.K_imp[2, 2] = kz_imp
        # self.C_imp[2, 2] = zeta_imp_z * 2 * np.sqrt(self.K_imp[2, 2] * self.M_imp[2, 2])
        #
        # wnRx = 18  # 8 #5 #5 #10  # 21 #17 #13
        # zeta_imp_Rx = 3 * 1  # 1
        # kRx_imp = 140
        # self.M_imp[3, 3] = kRx_imp / wnRx ** 2
        # self.K_imp[3, 3] = kRx_imp
        # self.C_imp[3, 3] = zeta_imp_Rx * 2 * np.sqrt(self.K_imp[3, 3] * self.M_imp[3, 3])
        #
        # wnRy = 18  # 8 #5 #10  # 21 #17 #13
        # zeta_imp_Ry = 3 * 1  # 1
        # kRy_imp = 140  # i used 1440 instead of 140 by mistake. try to use 140 and see if it works
        # self.M_imp[4, 4] = kRy_imp / wnRy ** 2
        # self.K_imp[4, 4] = kRy_imp
        # self.C_imp[4, 4] = zeta_imp_Ry * 2 * np.sqrt(self.K_imp[4, 4] * self.M_imp[4, 4])
        #
        # wnRz = 18  # 25  # 21 #17 #13
        # zeta_imp_Rz = 3  # 2  # 1
        # kRz_imp = 140  # 1
        # self.M_imp[5, 5] = kRz_imp / wnRz ** 2
        # self.K_imp[5, 5] = kRz_imp
        # self.C_imp[5, 5] = zeta_imp_Rz * 2 * np.sqrt(self.K_imp[5, 5] * self.M_imp[5, 5])
        #
        # # Convert decoupled (with L matrix) represantation to coupled M,D,K represantation
        # Limp_inv = LA.inv(self.L_imp)  # *np.array([[1,1,1,-1,-1,-1]]).T
        # self.K_imp = Limp_inv @ self.K_imp
        # self.C_imp = Limp_inv @ self.C_imp
        # self.M_imp = Limp_inv @ self.M_imp


        # # Shirs Paper params:
        # self.K_imp = np.array([[9.90660954, 0., 0., 0., 46.63195038, 0.],
        #                        [0., 33.75495148, 0., 157.51246643, 0., 0.],
        #                        [0., 0., 18.89282036, 0., 0., 0.],
        #                        [-38.88735199, 0., 0., 0., 32.1312713, 0.],
        #                        [0., -17.89356422, 0., 79.34197998, 0., 0.],
        #                        [0., 0., 0., 0., 0., 46.04693604]])
        # self.C_imp = np.array([[6.16129827, 0., 0., 0., -1.56223333, 0.],
        #                        [0., 114.63842773, 0., 30.10368919, 0., 0.],
        #                        [0., 0., 2.75284362, 0., 0., 0.],
        #                        [0., -12.54157734, 0., 69.30596924, 0., 0.],
        #                        [-42.15202713, 0., 0., 0., 75.14640808, 0.],
        #                        [0., 0., 0., 0., 0., 26.27482986]])
        # self.M_imp = np.array([[28.00367928, 0., 0., 0., 34.70161819, 0.],
        #                        [0., 71.05580902, 0., 37.04052734, 0., 0.],
        #                        [0., 0., 48.4661026, 0., 0., 0.],
        #                        [0., 39.43505096, 0., 63.75473022, 0., 0.],
        #                        [-44.1451416, 0., 0., 0., 7.56819868, 0.],
        #                        [0., 0., 0., 0., 0., 10.84090614]])

        # Shirs Paper params with correction of K matrix:
        self.K_imp = np.array([[9.90660954, 0., 0., 0., 46.63195038, 0.],
                               [0., 33.75495148, 0., 157.51246643, 0., 0.],
                               [0., 0., 18.89282036, 0., 0., 0.],
                               [0., -17.89356422, 0., 79.34197998, 0., 0.],
                               [-38.88735199, 0., 0., 0., 32.1312713, 0.],
                               [0., 0., 0., 0., 0., 46.04693604]])
        self.C_imp = np.array([[6.16129827, 0., 0., 0., -1.56223333, 0.],
                               [0., 114.63842773, 0., 30.10368919, 0., 0.],
                               [0., 0., 2.75284362, 0., 0., 0.],
                               [0., -12.54157734, 0., 69.30596924, 0., 0.],
                               [-42.15202713, 0., 0., 0., 75.14640808, 0.],
                               [0., 0., 0., 0., 0., 26.27482986]])
        self.M_imp = np.array([[28.00367928, 0., 0., 0., 34.70161819, 0.],
                               [0., 71.05580902, 0., 37.04052734, 0., 0.],
                               [0., 0., 48.4661026, 0., 0., 0.],
                               [0., 39.43505096, 0., 63.75473022, 0., 0.],
                               [-44.1451416, 0., 0., 0., 7.56819868, 0.],
                               [0., 0., 0., 0., 0., 10.84090614]])

        # Elad PPO params for 10mm Hole 8mm Peg and [15,15,15,3,3,3] wrench limits, Kp = [2500,2500,2500,500,500,500] and 6 sec for minimum jerk:
        # # After 1600 epochs:
        # self.K_imp = np.array([[126.26, 0, 0, 0, 375.23, 0],
        #                        [0, 126.26, 0, -375.23, 0, 0],
        #                        [0, 0, 1659.75, 0, 0, 0],
        #                        [0, 235.81, 0, 175.94, 0, 0],
        #                        [-235.81, 0, 0, 0, 175.94, 0],
        #                        [0, 0, 0, 0, 0, 363.14]])
        #
        # self.C_imp = np.array([[127.81, 0, 0, 0, -72, 0],
        #                        [0, 127.81, 0, 72, 0, 0],
        #                        [0, 0, 376.66, 0, 0, 0],
        #                        [0, 108.62, 0, 112.87, 0, 0],
        #                        [-108.62, 0, 0, 0, 112.87, 0],
        #                        [0, 0, 0, 0, 0, 116.99]])
        #
        # self.M_imp = np.array([[0.655, 0, 0, 0, -0.117, 0],
        #                        [0, 0.655, 0, 0.117, 0, 0],
        #                        [0, 0, 2.196, 0, 0, 0],
        #                        [0, 1.705, 0, 0.93, 0, 0],
        #                        [-1.705, 0, 0, 0, 0.93, 0],
        #                        [0, 0, 0, 0, 0, 1.807]])
        #
        # # Full training (1750 epochs):
        # self.K_imp = np.array([[124.5, 0., 0., 0., 380.1,  0.],
        #               [0., 124.5,  0., - 380.1,  0., 0.],
        #               [0., 0., 1691.93, 0., 0., 0.],
        #               [0., 213.93, 0., 181.41, 0.,   0.],
        #               [-213.93, 0., 0., 0., 181.41, 0.],
        #               [0., 0., 0., 0., 0., 362.03]])
        #
        # self.C_imp = np.array([[124.42, 0., 0., 0., - 71.71, 0.],
        #                [0., 124.42, 0.,  71.71, 0., 0.],
        #                [0., 0., 354.94, 0., 0., 0.],
        #                [0., 117.63, 0., 116.83, 0., 0.],
        #                [-117.63, 0., 0., 0., 116.83, 0.],
        #                [0.,   0.,   0.,   0.,   0., 115.24]])
        #
        # self.M_imp = np.array([[0.638, 0., 0., 0., - 0.122, 0.],
        #               [0.,  0.638,  0.,  0.122,  0.,  0.],
        #               [0., 0., 1.902, 0., 0., 0.],
        #               [0., 1.739, 0., 0.931, 0., 0.],
        #               [-1.739, 0., 0., 0., 0.931, 0.],
        #               [0.,  0.,  0.,  0.,  0.,  1.836]])

        # Load saved params:
        # signature = 'FMxynorm_F15M3limits' #'7alpha8beta' #'4alpha7beta' #'F_normalized'  #'F20M5limits'
        # with open(f'Control_Params/Learned_params_dict_{signature}' + '.pkl', 'rb') as f:
        #     KMC_dict = pickle.load(f)
        #     self.K_imp = KMC_dict['K']
        #     self.C_imp = KMC_dict['C']
        #     self.M_imp = KMC_dict['M']

            # # Delete2 KCM_norm_factor
            # KMC_norm_factor = 1  #3
            # self.K_imp = KMC_dict['K']/KMC_norm_factor
            # self.C_imp = KMC_dict['C']/KMC_norm_factor
            # self.M_imp = KMC_dict['M']/KMC_norm_factor
            # self.K_imp[2, 2] = KMC_dict['K'][2, 2]
            # self.C_imp[2, 2] = KMC_dict['C'][2, 2]
            # self.M_imp[2, 2] = KMC_dict['M'][2, 2]

        # # # Delete2!!! try negative signs for off-diagonal:
        # negative_off_diag = np.array([[1,0,0,0,-1,0],
        #                               [0,1,0,-1,0,0],
        #                               [0,0,1,0,0,0],
        #                               [0,-1,0,1,0,0],
        #                               [-1,0,0,0,1,0],
        #                               [0,0,0,0,0,1]])
        # # negative_off_diag = np.array([[1, 0, 0, 0, -1, 0],
        # #                               [0, 1, 0, -1, 0, 0],
        # #                               [0, 0, 1, 0, 0, 0],
        # #                               [0, 1, 0, 1, 0, 0],
        # #                               [1, 0, 0, 0, 1, 0],
        # #                               [0, 0, 0, 0, 0, 1]])
        # self.K_imp = negative_off_diag * self.K_imp
        # self.C_imp = negative_off_diag * self.C_imp
        # self.M_imp = negative_off_diag * self.M_imp


        # # Improvement for real robot:
        # self.K_imp[2, 2] = 2500 #1500 #5000
        # self.C_imp[2,2] = 2500
        # self.M_imp[2,2] = 5 #1


        # Print params if is_print_params=True:
        if is_print_params:
            print('\n')
            print(f'K_imp = {np.round(self.K_imp, 2)}')
            print(f'C_imp = {np.round(self.C_imp, 2)}')
            print(f'M_imp = {np.round(self.M_imp, 3)}')
            print('\n')

        # - - - - - - Matrices for Impedance equation - - - - - - - - - -
        M_imp_inv = LA.pinv(self.M_imp)   #LA.inv(self.M_imp)  #Delete pinv? use inv?
        self.A_imp = np.block([[np.zeros([6, 6]), np.eye(6)], [-M_imp_inv @ self.K_imp, -M_imp_inv @ self.C_imp]])
        self.B_imp = np.block([[np.zeros([6, 18])], [M_imp_inv, M_imp_inv @ self.K_imp, M_imp_inv @ self.C_imp]])
        self.A_imp_inv = LA.pinv(self.A_imp)  #LA.inv(self.A_imp)   #Delete pinv? use inv?

        # - - - - - - Force Controller Parameters - - - - - - - - - - - - -
        self.P_fc = np.zeros([6, 6])
        self.P_fc[0, 0], self.P_fc[0, 4] = 7.5, 11.5
        self.P_fc[1, 1], self.P_fc[1, 3] = 6, -2.5
        self.P_fc[2, 2] = 0.5
        self.P_fc[3, 1], self.P_fc[3, 3] = -0.5, 0.5
        self.P_fc[4, 0], self.P_fc[4, 4] = 0.6, 0.5
        self.P_fc[5, 5] = 0


if __name__ == '__main__':
    params = Control_param_20mm_circular_peg()
    K_eigvals, K_eigvecs = LA.eig(params.K_imp)
    C_eigvals, C_eigvecs = LA.eig(params.C_imp)
    M_eigvals, M_eigvecs = LA.eig(params.M_imp)
    A_eigvals, A_eigvecs = LA.eig(params.A_imp)
    Kinv_eigvals, Kinv_eigvecs = LA.eig(LA.inv(params.K_imp))

    # print(K_eigvecs[:,-1]@params.K_imp@K_eigvecs[:,-1]/(LA.norm(K_eigvecs[:,-1])**2))
    # print(K_eigvecs[:,3]@params.K_imp@K_eigvecs[:,3]/(LA.norm(K_eigvecs[:,3])**2))
    # print('A_eigvals =\n',A_eigvals.reshape([len(A_eigvals),1]))
    # print('X')

    # To save parameters uncomment the lines below:
    np.save('Saved Parameters\\20mm peg params\K_imp', params.K_imp)
    np.save('Saved Parameters\\20mm peg params\C_imp', params.C_imp)
    np.save('Saved Parameters\\20mm peg params\M_imp', params.M_imp)
    np.save('Saved Parameters\\20mm peg params\Kf_pd', params.Kf_pd)
    np.save('Saved Parameters\\20mm peg params\Km_pd', params.Km_pd)
    np.save('Saved Parameters\\20mm peg params\Cf_pd', params.Cf_pd)
    np.save('Saved Parameters\\20mm peg params\Cm_pd', params.Cm_pd)
    #np.save('Saved Parameters\\20mm peg params\F0', np.array([0, 0, -5, 0, 0, 0]))
