import robosuite as suite
import numpy as np
import math


from ActorCritic import ActorCritic_model, load_model, random_pos_err_list

# from ActorCritic_const_actor import ActorCritic_model, load_model

import time
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # # Load and plot Return vs episode and Success vs epoch:
    # with open('AC_avg_return_list' + '.pkl', 'rb') as f:
    #     avg_returns = pickle.load(f)
    # with open('AC_best_return_list' + '.pkl', 'rb') as f:
    #     best_returns = pickle.load(f)
    # with open('AC_success_rate_list' + '.pkl', 'rb') as f:
    #     success_rate_list = pickle.load(f)
    #
    # episodes_x_vec = np.arange(0, 10 * avg_returns.__len__(), 10)
    # plt.figure()
    # plt.plot(episodes_x_vec, best_returns, color='orange', label='best_returns')
    # plt.plot(episodes_x_vec, avg_returns, color='blue', label='avg_returns')
    # plt.xlabel('episode')
    # plt.ylabel('return')
    # plt.legend()
    # plt.grid()
    # plt.figure()
    # plt.plot(success_rate_list, color='orange')
    # plt.xlabel('epoch')
    # plt.ylabel('success rate')
    # plt.grid()
    #
    # # Plot avaerage success rate:
    # epochs_for_avg = 10
    # plt.figure(f'Avg success rate (for each {epochs_for_avg} epochs)')
    # avg_success_rate = np.array(success_rate_list).reshape([-1, epochs_for_avg])
    # avg_success_rate = np.sum(avg_success_rate, axis=1) / epochs_for_avg
    # epochs_x_vec = np.linspace(0, epochs_for_avg*avg_success_rate.__len__(), avg_success_rate.__len__())
    # plt.plot(epochs_x_vec, avg_success_rate, color='green')
    # plt.xlabel('epoch')
    # plt.ylabel('success rate')
    # plt.grid()
    #
    # plt.show()

    # Load a trained model:
    is_save_params = True  # False  #True #False
    is_print_params = True  # False
    which_policy = 'mu_normal'  #'mu_normal'  #'best_mu_normal'
    AC_model = load_model('my_AC_model')

    # ------------ Enviroment settings and initialization -------------:
    time_for_simulation = 17+4  #17 #15
    control_freq = 10
    trials = 200  #50 #100 #100
    num_of_steps = round(control_freq * time_for_simulation)
    print(f'num_of_steps = horizon = {num_of_steps}')
    display = False  # True
    action_method_dict = AC_model.action_method_dict
    print(action_method_dict)
    adaptive_xy_ref = False  # True
    random_err = True  #False
    fixed_err = 0.002 #0.0006#0.002
    pos_err_list = [np.array([-fixed_err, 0, 0]), np.array([fixed_err, 0, 0]), np.array([0, -fixed_err, 0]),np.array([0, fixed_err, 0])]  #[np.array([0.0,0.01,0])]
    if random_err:
        pos_err_list = random_pos_err_list(size=trials, radius_range=(0.0015,0.0025))  #radius_range=(0.0004,0.0008)) #radius_range=(0.00021,0.0008))

    controller_configs = dict(type='decreasing_Vrot_controler', input_max=1, input_min=-1, output_max=0.1,   #type='quaternions_delta_controler'  #type='decreasing_Vrot_controler'  #type='quaternions_controler'
                              output_min=-0.1, torque_limits=None, interpolation=None, ramp_ratio=0.2,
                              action_method_dict=action_method_dict, adaptive_xy_ref=adaptive_xy_ref)
    # env_name='PegInHole','Lift''NutAssemblyRound'
    env_configs = dict(  # env_initializer_func(
        env_name='PegInHole',
        robots='UR5e',
        controller_configs=controller_configs,
        has_renderer=False,  # False, # True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        horizon=num_of_steps,  # (steps_per_action + steps_per_rest) * num_test_steps,
        control_freq=control_freq,
        pos_err_list=pos_err_list,
        # control_dim=6,
        # initialization_noise=None,
    )  # control_freq=20,


    #display = True #False
    env_configs['has_renderer'] = display #False #True
    env = suite.make(**env_configs)

    print(f'\n* * * display info and evaluate for the policy="{which_policy}" * * *\n')

    if is_save_params or is_print_params:
        # Construct params dict:
        params_dict = env.robots[0].controller.params_dict   # Here the params_dict is set with the default values
        some_state = np.array([0, 0])
        action = AC_model.get_scaled_mu(state=some_state, which_policy=which_policy)
        AC_model.ActionConvertor.update_params_values(action_values=action)
        AC_model.ActionConvertor.params_placement_for_outer_source(outer_params_dict=params_dict)

    if is_print_params:
        print(f'std = {AC_model.covariance_mat.diagonal() ** 0.5}')

        K_imp = params_dict['K_imp']  # AC_model.ActionConvertor.params_dict['K_imp'].as_params
        params_source = 'learned' if action_method_dict['K_imp'] != None else 'default'
        print(f'K_imp ({params_source}) = {K_imp.round(2)}')

        C_imp = params_dict['C_imp']  # AC_model.ActionConvertor.params_dict['C_imp'].as_params
        params_source = 'learned' if action_method_dict['C_imp'] != None else 'default'
        print(f'C_imp ({params_source}) = {C_imp.round(2)}')

        M_imp = params_dict['M_imp']  # AC_model.ActionConvertor.params_dict['M_imp'].as_params
        params_source = 'learned' if action_method_dict['M_imp'] != None else 'default'
        print(f'M_imp ({params_source}) = {M_imp.round(2)}')

        Kp = params_dict['Kp']  # AC_model.ActionConvertor.params_dict['Kp'].as_params
        params_source = 'learned' if action_method_dict['Kp'] != None else 'default'
        print(f'Kp ({params_source}) = {Kp.round(2)}')

        Kd = params_dict['Kd']  # AC_model.ActionConvertor.params_dict['Kd'].as_params
        params_source = 'learned' if action_method_dict['Kd'] != None else 'default'
        print(f'Kd ({params_source}) = {Kd.round(2)}')

        spiral_p = params_dict['spiral_p']   #AC_model.ActionConvertor.params_dict['spiral_p'].as_params
        params_source = 'learned' if action_method_dict['spiral_p'] != None else 'default'
        print(f'spiral_p ({params_source}) = {round(1000 * spiral_p, 3)}[mm]')

        spiral_v = params_dict['spiral_v']   #AC_model.ActionConvertor.params_dict['spiral_v'].as_params
        params_source = 'learned' if action_method_dict['spiral_v'] != None else 'default'
        print(f'spiral_v ({params_source}) = {round(1000 * spiral_v, 3)}[mm/sec]')

    if is_save_params:
        # Save constant params. If you are using NN as actor you shall save the AC_model.actor which is an NN_model object.
        Learned_params_dict = {'K_imp': K_imp, 'C_imp': C_imp, 'M_imp': M_imp, 'Kp': Kp, 'Kd': Kd, 'spiral_p': spiral_p,
                               'spiral_v': spiral_v}
        try:
            if which_policy=='mu_normal':
                add_policy_str = ''
            elif which_policy=='best_mu_normal':
                add_policy_str = '_best'
            path = f'{AC_model.path}/Learned_params_dict{add_policy_str}'
            with open(path + '.pkl', 'wb') as f:
                pickle.dump(Learned_params_dict, f)
            print('\nYour params have been saved.\n')
        except Exception as e:
            print(f'Saving parameters in path = {path} failed raising the error below:\n{e}')
            path = 'Learned_params_dict'  # f'{AC_model.path}/Learned_params_dict'
            with open(path + '.pkl', 'wb') as f:
                pickle.dump(Learned_params_dict, f)
            print(
                '\nInstead: Your params have been saved in the default folder (where the current program file is located).\n')

    #RL_PG_model2.run_policy(env, episodes=2, horizon=num_of_steps, reward_func=reward_function, randomization=True, display=True)
    data = AC_model.run_policy(env, episodes_num=trials, horizon=num_of_steps, reward_func=None, which_policy=which_policy, gamma_return=1, display=display)

    successes = sum(data['successes'])
    trials = data['successes'].__len__()
    print(f'successes = {successes}')
    print(f'\nSuccess rate = {successes}/{trials}')
    # Plot simulation:
    robot = env.robots[0]
    env.close()
    robot.controller.plot_simulation()
