import numpy as np
'''
Contains PrioritizedBuffer class 
'''

class PrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha= 0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.states_count = {0:0, 1:0, 'frac': 0}
        self.states_loss = {0:0, 1:0, 'frac': 0}
        self.sample_counts = np.zeros(capacity)

    def push(self, meta_state, action, reward, meta_next_state, done):
        state = meta_state[0]
        state_env = meta_state[1]
        next_state = meta_next_state[0]

        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, int(state_env)))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done, int(state_env))

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        self.sample_counts[indices] += 1

        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= (weights.max()) # FIXIT: What if the max is zero?
        weights  = np.array(weights, dtype=np.float32)

        batch       = list(zip(*samples))
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]
        state_envs = list(batch[5])

        # increment states count
        self.states_count[0] += len(state_envs) - sum(state_envs) # states.shape[0] - np.sum(states[:,-1])
        self.states_count[1] += sum(state_envs)  # np.sum(states[:,-1])

        return states, actions, rewards, next_states, dones, indices, weights, state_envs

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
            
    def get_average_weight_by_env(self):
        # may be worth switching to np array buffer for efficiency
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        noisy_weight_sum = 0
        std_weight_sum = 0
        noisy_count = 1e-5
        for i in range(len(self.buffer)):
            state = self.buffer[i][0]
            is_noisy_state = (state.squeeze()[-1] == 1)
            if is_noisy_state:
                noisy_weight_sum += probs[i]
                noisy_count += 1
            else:
                std_weight_sum += probs[i]
                
        return {'std_avg': std_weight_sum / (len(self.buffer) - noisy_count),
                'noisy_avg': noisy_weight_sum / noisy_count,
                'noisy_count': noisy_count,
                'std_count': len(self.buffer) - noisy_count}
            

    def __len__(self):
        return len(self.buffer)

