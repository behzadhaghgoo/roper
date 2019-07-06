import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ray.tune import run, Trainable, sample_from

from dqn import DQN
from loss import TDLoss, StableTDLoss
from pbuffer import PrioritizedBuffer


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

beta_start = 0.4
beta_frames = 1000
BETA_BY_FRAME = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
EPSILON_BY_FRAME = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


class MyTrainable(Trainable):

    def _setup(self):
        self.env, self.val_env = get_env(env_id)

    def _train(self):

        hardcoded_reward = self.train_helper(True) 
        non_hardcoded_reward = self.train_helper(False) 

        return non_hardcoded_reward - hardcoded_reward


    def train_helper(self, hardcoded):

        config = self.config 

        losses = []
        all_rewards = []
        standard_val_rewards = []
        noisy_val_rewards = []
        states_count_ratios = []
        episode_reward = 0

        # Initialize state
        noisyGame = False
        state = self.env.reset()
        state = np.append(state, float(noisyGame)) 
        meta_state = (state, float(noisyGame))

        # Initialize replay buffer, model, TD loss, and optimizers
        # result_df = pd.DataFrame()
        power = config['theta']
        all_standard_val_rewards = []
        all_proportions = []
        std_weights = []
        noisy_weights = []
        std_buffer_example_count = []
        noisy_buffer_example_count = []

        for t in range(num_trials):

            self.training_iteration += 1 

            if cnn:
                current_model = CnnDQN(self.env.observation_space.shape, self.env.action_space.n)
                target_model  = CnnDQN(self.env.observation_space.shape, self.env.action_space.n)
            else:
                current_model = DQN(self.env.observation_space.shape[0] + 1, self.env.action_space.n) 
                target_model  = DQN(self.env.observation_space.shape[0] + 1, self.env.action_space.n)

            if config['method'] == 'ours':
                td_loss = StableTDLoss()
            else:
                td_loss = TDLoss()

            optimizer = optim.Adam(current_model.parameters(), lr=config['lr'])


            if USE_CUDA:
                current_model = current_model.cuda()
                target_model  = target_model.cuda()


            replay_buffer = PrioritizedBuffer(int(1e6))

            print("trial number: {}".format(t))

            for frame_idx in range(1, num_frames + 1):
                epsilon = EPSILON_BY_FRAME(frame_idx)
                original_action = current_model.act(state, epsilon)

                # If in noisy environment, make action random with probability eps
                if noisyGame and random.uniform(0,1) < config['decision_eps']:
                    if config['invert_actions']:
                        actual_action = 1 - original_action # invert
                    else:
                        actual_action = original_action
                else:
                    actual_action = original_action

                next_state, reward, done, _ = self.env.step(actual_action)

                # If in noisy environment, make reward completely random
                if noisyGame:
                    reward *= np.random.normal(config['mean'], config['var'])

                if not config['cnn']:
                    next_state = np.append(next_state, float(noisyGame))
                meta_next_state = (next_state, float(noisyGame))
                
                # store q values and hidden states in buffer
                replay_buffer.push(meta_state, original_action, reward, meta_next_state, done)

                meta_state = meta_next_state
                episode_reward += reward

                if done:
                    noisyGame = 1-noisyGame
                    state = self.env.reset()
                    state = np.append(state, float(noisyGame))
                    meta_state = (state, float(noisyGame))
                    all_rewards.append(episode_reward)
                    episode_reward = 0

                if len(replay_buffer) > config['batch_size'] and frame_idx % 4 == 0:
                    beta = BETA_BY_FRAME(frame_idx)
                    loss = td_loss.compute_td_loss(current_model, target_model, beta, replay_buffer, optimizer)
                    losses.append(loss.data.tolist())

                if frame_idx % 200 == 0:
                    all_standard_val_rewards.append(self._test(False, config['eps'], config['num_val_trials'], current_model))
                    all_proportions.append(float(replay_buffer.states_count[1]) / (float(replay_buffer.states_count[1])  + float(replay_buffer.states_count[0])))
                    weight_dict = replay_buffer.get_average_weight_by_env()
                    std_weights.append(weight_dictconfig['std_avg'])
                    noisy_weights.append(weight_dictconfig['noisy_avg'])
                    std_buffer_example_count.append(weight_dictconfig['std_count'])
                    noisy_buffer_example_count.append(weight_dictconfig['noisy_count'])
                    

                if frame_idx % 1000 == 0:
                    print("Frame {}".format(frame_idx))
                    update_target(current_model, target_model)

        #return tune.TrainingResult(timesteps_this_iter=n, mean_loss=validation_loss)
        return np.mean(all_standard_val_rewards)


    def _test(noisyGame, eps, num_val_trials, current_model):

        rewards = []

        for i in range(num_val_trials):
            epsilon = 0
            episode_reward = 0
            state = self.val_env.reset()
            state = np.append(state, float(noisyGame))

            with torch.no_grad():
                while True:
                    original_action = current_model.act(state, epsilon)

                    if original_action != int(original_action):
                        original_action = original_action.numpy()[0]

                    actual_action = original_action
                    next_state, reward, done, _ = self.val_env.step(actual_action)
                    next_state = np.append(next_state, float(noisyGame))

                    state = next_state
                    episode_reward += reward

                    if done:
                        rewards.append(episode_reward)
                        break

        return np.mean(rewards)



# def train(self):

#     config = self.config 

#     '''
#     hyperparams 

#     method - 'average_over_batch', 'PER'
#     var
#     mean
#     decision_eps,
#     alpha, beta,
#     hardcoded, cnn,
#     invert_actions = False, 
#     num_frames = 30000, 
#     num_val_trials = 10, 
#     batch_size = 32, 
#     gamma = 0.99,
#     num_trials = 5, 
#     USE_CUDA = False, 
#     device = "", 
#     eps = 1., 
#     avg_stored=False
#     '''
    
#     if USE_CUDA:
#         device = torch.device("cuda")

#     """Args:"""
#     losses = []
#     all_rewards = []
#     standard_val_rewards = []
#     noisy_val_rewards = []
#     states_count_ratios = []
#     episode_reward = 0

#     # Initialize state
#     noisyGame = False
#     state = config['env'].reset()
#     state = np.append(state, float(noisyGame)) 
#     meta_state = (state, float(noisyGame))

#     # Initialize replay buffer, model, TD loss, and optimizers

#     result_df = pd.DataFrame()
#     theta = 1.
#     power = config['theta']
#     all_standard_val_rewards = []
#     all_proportions = []
#     std_weights = []
#     noisy_weights = []
#     std_buffer_example_count = []
#     noisy_buffer_example_count = []

#     for t in range(num_trials):

#         if cnn:
#             current_model = CnnDQN(env.observation_space.shape, env.action_space.n)
#             target_model  = CnnDQN(env.observation_space.shape, env.action_space.n)
#         else:
#             current_model = DQN(env.observation_space.shape[0] + 1, env.action_space.n) 
#             target_model  = DQN(env.observation_space.shape[0] + 1, env.action_space.n)
        
#         td_loss = TDLoss(method=config['method'])

#         optimizer = optim.Adam(current_model.parameters())


# #     # Single GPU Code
#         if USE_CUDA:
#             current_model = current_model.cuda()
#             target_model  = target_model.cuda()


#         if config['method']=='average_over_buffer':
#             replay_buffer = AugmentedPrioritizedBuffer(int(1e6))
#         else:
#             replay_buffer = PrioritizedBuffer(int(1e6))

#         print("trial number: {}".format(t))

#         for frame_idx in range(1, config['num_frames'] + 1):
#             epsilon = EPSILON_BY_FRAME(frame_idx)
#             original_action = current_model.act(state, epsilon)

#             # If in noisy environment, make action random with probability eps
#             if noisyGame and random.uniform(0,1) < config['decision_eps']:
#                 if invert_actions:
#                     actual_action = 1 - original_action # invert
#                 else:
#                     actual_action = original_action
#             else:
#                 actual_action = original_action

#             next_state, reward, done, _ = config['env'].step(actual_action)

#             # If in noisy environment, make reward completely random
#             if noisyGame:
#                 reward *= np.random.normal(config['mean'], var)

#             if not cnn:
#                 next_state = np.append(next_state, float(noisyGame))
#             meta_next_state = (next_state, float(noisyGame))
            
#             # store q values and hidden states in buffer
#             if config['method']=='average_over_buffer':
#                 state_var = Variable(torch.FloatTensor(np.float32(state)))
#                 with torch.no_grad():
#                     q_values, hiddens = current_model.forward(state_var, config['return_latent'] = "last")
#                 replay_buffer.push(meta_state, original_action, reward, meta_next_state, done, hiddens, q_values)
#             else:
#                 replay_buffer.push(meta_state, original_action, reward, meta_next_state, done)

#             meta_state = meta_next_state
#             episode_reward += reward

#             if done:
#                 noisyGame = 1-noisyGame
#                 state = env.reset()
#                 state = np.append(state, float(noisyGame))
#                 meta_state = (state, float(noisyGame))
#                 all_rewards.append(episode_reward)
#                 episode_reward = 0

#             if len(replay_buffer) > batch_size and frame_idx % 4 == 0:
#                 beta = BETA_BY_FRAME(frame_idx)
#                 loss = td_loss.compute_td_loss(current_model, target_model, beta, replay_buffer, optimizer)
#                 losses.append(loss.data.tolist())

#             if frame_idx % 200 == 0:
#                 all_standard_val_rewards.append(test(val_env, False, eps, num_val_trials, current_model))
#                 all_proportions.append(float(replay_buffer.states_count[1]) / (float(replay_buffer.states_count[1])  + float(replay_buffer.states_count[0])))
#                 weight_dict = replay_buffer.get_average_weight_by_env()
#                 std_weights.append(weight_dictconfig['std_avg'])
#                 noisy_weights.append(weight_dictconfig['noisy_avg'])
#                 std_buffer_example_count.append(weight_dictconfig['std_count'])
#                 noisy_buffer_example_count.append(weight_dictconfig['noisy_count'])
                
                
#             #         plot(frame_idx, all_rewards, losses, standard_val_rewards, noisy_val_rewards, states_count_ratios)

#             if frame_idx % 1000 == 0:
#                 print("Frame {}".format(frame_idx))
#                 update_target(current_model, target_model)

#     print(len(all_proportions))
    
#     result_dfconfig['frame'] = 200*np.arange(len(all_proportions)) % num_frames
#     result_dfconfig['trial_num'] = np.floor(200 *np.arange(len(all_proportions)) / num_frames)
#     result_dfconfig['val_reward'] = all_standard_val_rewards
#     result_dfconfig['proportion'] = all_proportions
#     result_dfconfig['std_weights'] = std_weights
#     result_dfconfig['noisy_weights'] = noisy_weights
#     result_dfconfig['std_buffer_example_count'] = std_buffer_example_count
#     result_dfconfig['noisy_buffer_example_count'] = noisy_buffer_example_count
#     return result_df

