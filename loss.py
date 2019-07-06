import numpy as np
import torch


# TODO: Add average over batch / Average over buffer
class TDLoss():
    """Compute Baseline TD Loss."""
    def __init__(self, batch_size = 32, stored_aug_size=1000, theta = 1, mode = "dot", exp = False, meg_norm = False, average_q_values = False, method='PER'):
        """Args:
         mode: "dot" or "euc", the distance function for averaging
         theta: power of weights (see paper) """
        super(TDLoss, self).__init__()
        self.batch_size = batch_size
        self.theta = theta
        self.mode = mode
        self.exp = exp
        self.meg_norm = meg_norm
        self.method = method
        self.gamma = 0.99
        self.stored_aug_size = stored_aug_size

    def compute(self, cur_model, tar_model, beta, replay_buffer, optimizer):
        state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta)

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action     = Variable(torch.LongTensor(action))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(done))
        weights    = Variable(torch.FloatTensor(weights))

        q_values      = current_model(state)
        next_q_values = target_model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss  = (q_value - expected_q_value.detach()).pow(2) * weights
        priorities = loss + 1e-5 # Avoiding zero loss.
        loss  = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        replay_buffer.update_priorities(indices, priorities.data.cpu().numpy())
        optimizer.step()

        return loss

class StableTDLoss():
    def __init__(self, batch_size = 32, stored_aug_size=1000, theta = 1, mode = "dot", exp = False, meg_norm = False, average_q_values = False, method='PER'):
        """Args:
         mode: "dot" or "euc", the distance function for averaging
         theta: power of weights (see paper) """
        super(TDLoss, self).__init__()
        self.batch_size = batch_size
        self.theta = theta
        self.mode = mode
        self.exp = exp
        self.meg_norm = meg_norm
        self.hidden = "hidden"
        # self.method = method
        self.gamma = 0.99
        # self.stored_aug_size = stored_aug_size

    def hidden_weights(h, power, batch_size = 32, meg_norm = False):
        if meg_norm:
            h = h / torch.reshape(torch.norm(torch.mul(h, h), dim = 1, p = 2), (-1,1))
    # #       w[i,j] = h[i].h[j]/|h[i]||h[j]|

        weights = torch.mm(h,torch.transpose(h, 0, 1))
        weights = torch.abs(weights)**power
        return weights


    def hidden_mean(h, tensor, batch_size, power, mode = "dot", exp = False):
        tensor = torch.reshape(tensor,(-1,1))
        #TODO: what was the better metric that I (Behzad) saw?
        if mode == "dot":
            weights = hidden_weights(h, power, batch_size)
        elif mode == "euc":
            raise NotImplementedError

        output = torch.mm(weights, tensor)
        output = output.squeeze(1)

        return output * batch_size

    def compute(self, cur_model, tar_model, beta, replay_buffer, optimizer):

        state, action, reward, next_state, done, indices, weights, state_envs = replay_buffer.sample(batch_size, beta)

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action     = Variable(torch.LongTensor(action))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(done))
        weights    = Variable(torch.FloatTensor(weights))

        q_values, hiddens = cur_model.forward(state, return_hidden = True)
        next_q_values = tar_model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss  = (q_value - expected_q_value.detach()).pow(2) * weights

        # Averaging
        q_value = hidden_mean(hiddens, q_value, batch_size, power)
        expected_q_value = hidden_mean(hiddens, expected_q_value, batch_size, power)

        priorities = (q_value - expected_q_value.detach()).pow(2) * weights + 1e-5
        loss  = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        replay_buffer.update_priorities(indices, priorities.data.cpu().numpy())
        optimizer.step()

        return loss
