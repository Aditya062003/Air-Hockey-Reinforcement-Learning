import numpy as np
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
from Normalizer import Normalizer

device = "cpu"


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        Function = nn.Tanh
        
        
        policy_layers = []

        policy_layers.append(nn.Linear(18,128))
        policy_layers.append(Function())
        for i in range(1):
            policy_layers.append(nn.Linear(128,128))
            policy_layers.append(Function())
        policy_layers.append(nn.Linear(128,8))
        torch.nn.init.orthogonal_(policy_layers[-1].weight, gain=0.01)
        self.actor = nn.Sequential(*policy_layers)

        
        value_layers = []

        value_layers.append(nn.Linear(18, 256))
        value_layers.append(Function())

        for i in range(3):
            value_layers.append(nn.Linear(256,256))
            value_layers.append(Function())

        value_layers.append(nn.Linear(4, 1))

        torch.nn.init.orthogonal_(value_layers[-1].weight,gain=0.001)
        self.critic = nn.Sequential(*value_layers)

        
        for l in (policy_layers[:-2] + value_layers[:-1]):
            if isinstance(l, nn.Linear):
                torch.nn.init.orthogonal_(l.weight)
        
        self._action_std_log = torch.nn.Parameter(np.log(0.5) * torch.ones(4, dtype=torch.float32, device=device))
        self._evaluation_std = (0.0000001 * torch.ones((4,), dtype=torch.float32, device=device))

    def forward(self, state, evaluation):
        network_output = self.actor(state)
        action_mean = network_output[:, 0:4]
        if evaluation:
            action_std = self._evaluation_std.detach()
        else:
            action_std = torch.exp(self._action_std_log).expand_as(action_mean)
            action_std = torch.clamp(action_std, 0.01, 15)
        cov_mat = torch.diag_embed(action_std**2)
        return MultivariateNormal(action_mean, cov_mat)

    def act(self, state, memory=None):
        evaluation = memory is None
        dist = self(state, evaluation)
        action = dist.sample()
        if evaluation:
            torch.no_grad()
        else:
            action_logprob = dist.log_prob(action)
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        return action.cpu().numpy()

    def evaluate(self, state, action):
        dist = self(state, False)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self):

        self.input_normalizer = Normalizer(18,10)


        self.policy = ActorCritic().to(device)

        self.policy_old = ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=0.0003,weight_decay=0.0,betas=(0.9, 0.999),eps=1e-8)

        self._mse_loss = nn.MSELoss()

    def act(self, state, memory=None):
        state = self.input_normalizer.add_and_normalize(state)
        state = np.asarray(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return np.tanh(self.policy_old.act(state, memory).flatten())

    def update(self, memory):
        
        old_states = torch.squeeze(torch.stack(memory.states).to(device),1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device),1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs),1).to(device).detach()

        for i in range(10):
            advantages, returns = (self.compute_advantages_and_returns(memory))

            for states_, actions_, old_logprobs_, returns_, advantages_ in self.generate_batch_iterations(old_states, old_actions,old_logprobs, returns,advantages):
                logprobs, state_values, dist_entropy = self.policy.evaluate(states_, actions_)
                ratios = torch.exp(logprobs - old_logprobs_)
                surr_policy_1 = ratios * advantages_
                surr_policy_2 = torch.clamp(ratios, 0.75, 1.25) * advantages_
                value_loss = self._mse_loss(state_values, returns_)
                loss = (-torch.min(surr_policy_1, surr_policy_2)+ 0.5*value_loss - 0.01*dist_entropy)
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def generate_batch_iterations(self, states, actions, log_probs, returns, advantages):
        batch_size = states.size(0)
        for _ in range(batch_size // self._mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self._mini_batch_size)
            yield (states[rand_ids, :], actions[rand_ids, :],log_probs[rand_ids], returns[rand_ids],advantages[rand_ids])

    def compute_advantages_and_returns(self, memory):
        states = torch.stack(memory.states).to(device)
        values = self.policy.critic(states).flatten().float().detach().cpu()
        values = values.numpy()
        final_states = torch.FloatTensor(memory.final_states).to(device)
        final_values = self.policy.critic(final_states).flatten().detach().cpu().numpy()

        returns = np.zeros(len(memory.actions), dtype=float)
        index = len(returns)-1

        step_reverse_counter = memory.lengths[-1]
        reverse_index = len(memory.lengths)-1
        nextValue = final_values[reverse_index]
        mask = np.concatenate((np.ones(step_reverse_counter), np.asarray([0])))
        gae = 0

        for step in reversed(range(len(memory.rewards))):
            if step_reverse_counter == 0:
                reverse_index -= 1
                nextValue = final_values[reverse_index]
                if reverse_index >= 0:
                    step_reverse_counter = memory.lengths[reverse_index]
                mask = np.concatenate((np.ones(step_reverse_counter),np.asarray([0])))
                gae = 0

            reward = memory.rewards[step]

            delta = (reward + (0.95 * nextValue*mask[step_reverse_counter]) - values[step])
            gae = delta + (0.95 * 0.95 * mask[step_reverse_counter] * gae)

            returns[index] = gae + values[step]
            index -= 1

            step_reverse_counter -= 1
            
            nextValue = values[step]

        advantages = returns - values

        advantages = torch.from_numpy(advantages).float().to(device)
        returns = torch.from_numpy(np.asarray(returns)).float().to(device)
        return advantages, returns
