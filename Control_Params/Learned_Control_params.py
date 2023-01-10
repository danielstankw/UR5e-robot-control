import numpy as np
from numpy import linalg as LA
import pickle

class Learned_Params:
    def __init__(self):
        # Load saved params:
        signature = '100succ_10traj_KXsymCMKpKdDiagsym_181122'  #'100succ_10traj_KMCKpKdDiagsym_201122'  #'100succ_10traj_KXsymCMKpKdDiagsym_181122'  #'080real_best_0955sim_FMxynorm_10traj_KCXsymMKpKdDiagsym_221122'  #'050succ_best_KXsymCMKpKdDiagsym_171122'  #'070succ_KXsymCMKpKdDiagsym_171122'  #'100succ_best_FMxynorm_10traj_KCXsymMKpKdDiagsym'  #'100succ_10traj_KXsymCMKpKdDiagsym_181122'  #'best_KXsymCMKpKdDiagsym_171122'  #'best_10traj_KXsymCMKpKdDiagsym_181122'   #'10traj_KMCKpKdDiagsym_201122'  #'best_FMxynorm_KCXsymMKpKdDiagsym_161122'  #'FMxynorm_KCXsymMKpKdDiagsym_161122' #'KXsymCMKpKdDiagsym_171122'  #'FMxynorm_F15M3limits' #'7alpha8beta' #'4alpha7beta' #'F_normalized'  #'F20M5limits'
        path = f'Control_Params/learned_params_21_11_2022/{signature}' + '.pkl'
        with open(path , 'rb') as f:
            params_dict = pickle.load(f)
            self.K_imp = params_dict['K_imp']
            self.C_imp = params_dict['C_imp']
            self.M_imp = params_dict['M_imp']
            Kp = params_dict['Kp']
            Kd = params_dict['Kd']
            self.Kf_pd = Kp[:3,:3]
            self.Cf_pd = Kd[:3,:3]
            self.Km_pd = Kp[3:,3:]
            self.Cm_pd = Kd[3:,3:]

            # Optional - print the loaded parameters:
            is_print_params = True #False
            if is_print_params:
                print('\n')
                print(f'K_imp = \n{np.round(self.K_imp, 2)}')
                print(f'C_imp = \n{np.round(self.C_imp, 2)}')
                print(f'M_imp = \n{np.round(self.M_imp, 3)}')
                print(f'Kp = \n{np.round(Kp,2)}')
                print(f'Kf_pd = \n{np.round(self.Kf_pd,2)}')
                print(f'Km_pd = \n{np.round(self.Km_pd, 2)}')
                print(f'Kd = \n{np.round(Kd, 2)}')
                print(f'Cf_pd = \n{np.round(self.Cf_pd, 2)}')
                print(f'Cm_pd = \n{np.round(self.Cm_pd, 2)}')
                print('\n')

            # - - - - - - Matrices for Impedance equation - - - - - - - - - -
            M_imp_inv = LA.pinv(self.M_imp)  # LA.inv(self.M_imp)  #Delete pinv? use inv?
            self.A_imp = np.block(
                [[np.zeros([6, 6]), np.eye(6)], [-M_imp_inv @ self.K_imp, -M_imp_inv @ self.C_imp]])
            self.B_imp = np.block(
                [[np.zeros([6, 18])], [M_imp_inv, M_imp_inv @ self.K_imp, M_imp_inv @ self.C_imp]])
            self.A_imp_inv = LA.pinv(self.A_imp)  # LA.inv(self.A_imp)   #Delete pinv? use inv?

            # - - - - - - Force Controller Parameters - - - - - - - - - - - - -
            self.P_fc = np.zeros([6, 6])
            self.P_fc[0, 0], self.P_fc[0, 4] = 7.5, 11.5
            self.P_fc[1, 1], self.P_fc[1, 3] = 6, -2.5
            self.P_fc[2, 2] = 0.5
            self.P_fc[3, 1], self.P_fc[3, 3] = -0.5, 0.5
            self.P_fc[4, 0], self.P_fc[4, 4] = 0.6, 0.5
            self.P_fc[5, 5] = 0

        # else:
        #     raise ValueError(f'Failed to load params from path: {path}. Check that the file you are willing to load exist at this location.')


