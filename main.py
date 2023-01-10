import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6

mbpo_target_entropy_dict = {'Hopper-v2':-1, 'HalfCheetah-v2':-3, 'Walker2d-v2':-3, 'Ant-v2':-4, 'Humanoid-v2':-2}
mbpo_epoches = {'Hopper-v2':125, 'Walker2d-v2':300, 'Ant-v2':300, 'HalfCheetah-v2':400, 'Humanoid-v2':300}

def weights_init_(m):

    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr+1) % self.max_size
        
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    idxs=idxs)


class Mlp(nn.Module):
    
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        
        self.hidden_layers = nn.ModuleList()
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.apply(weights_init_)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output

class TanhNormal(Distribution):

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):

        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - \
               torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        
        z = (
            self.normal_mean +
            self.normal_std *
            Normal( 
                torch.zeros(self.normal_mean.size()),
                torch.ones(self.normal_std.size())
            ).sample()
        )
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

class TanhGaussianPolicy(Mlp):
   
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0
    ):
        super().__init__(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )
        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        
        self.action_limit = action_limit
        self.apply(weights_init_)

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=True,
    ):
        
        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        mean = self.last_fc_layer(h)

        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob = log_prob - torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        return (
            action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
        )

def soft_update_model1_with_model2(model1, model2, rou):
    
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(rou*model1_param.data + (1-rou)*model2_param.data)

def test_agent(agent, test_env, max_ep_len, logger, n_eval=1):
    
    ep_return_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            
            a = agent.get_test_action(o)
            o, r, d, _ = test_env.step(a)
            ep_ret = ep_ret + r
            ep_len = ep_len + 1
        ep_return_list[j] = ep_ret
        if logger is not None:
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    return ep_return_list


def get_redq_true_estimate_value(agent, logger, test_env, max_ep_len, n_eval=1):
    
    true_return_list = []
    estimate_return_list = []
    max_ep_len = 150

    for j in range(n_eval):
        o = test_env.reset()
        for j in range(n_eval):
            r_true, d_true, ep_ret_true, ep_len_true = 0, False, 0, 0
            reward_list = []
            while not (d_true or (ep_len_true == max_ep_len)):
                a = agent.get_test_action(o)
                obs_tensor = torch.Tensor(o).unsqueeze(0).to(agent.device)
                action_tensor = torch.Tensor(a).unsqueeze(0).to(agent.device)                
                q_prediction_list = []
                for q_i in range(agent.num_Q):
                    q_prediction = agent.q_net_list[q_i](torch.cat([obs_tensor, action_tensor], 1))
                    q_prediction_list.append(q_prediction)
                    q_prediction_list_mean = torch.cat(q_prediction_list, 1).mean(dim=1).reshape(-1, 1) 
                estimate_return_list.append(q_prediction_list_mean)

                o, r_true, d_true, _ = test_env.step(a)
                ep_ret_true = ep_ret_true + r_true * (agent.gamma ** ep_len_true) * 1 
                reward_list.append(r_true)
                ep_len_true = ep_len_true  + 1

            true_return_list = []
            true_return_list.append(ep_ret_true)
            for ii in range(len(reward_list)-1):
                tem_reward = np.true_divide(true_return_list[ii]-reward_list[ii],agent.gamma)
                true_return_list.append(tem_reward)


        estimate_return_list_array = torch.cat(estimate_return_list, 1).detach().cpu().numpy().reshape(-1)
        true_return_list_array = np.array(true_return_list)

        expected_true_value = abs(np.mean(true_return_list_array))
        exp_error = np.mean(estimate_return_list_array-true_return_list_array)
        exp_error = np.true_divide(exp_error, expected_true_value)
        std_error = np.std(estimate_return_list_array-true_return_list_array)
        std_error = np.true_divide(std_error, expected_true_value)

        if logger is not None:
            logger.store(ExpError=exp_error, StdError=std_error)

    return exp_error, std_error       