import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus

class MLPActorCritic(nn.Module):
    
    def __init__(self, actor_layer_sizes, critic_layer_sizes, initial_sd, num_actions, device, activation=nn.Tanh):
        super().__init__()
        
        # initialize with an initial standard deviation to stimulate exploration early on
        self.initial_sd = torch.as_tensor(initial_sd, dtype=torch.float32, device=device).detach()
        
        self.num_actions = num_actions
        
        self.actor = self._mlp(actor_layer_sizes, activation).to(device)
        
        with torch.no_grad():
            # force the weights of the last layer to be really small
            # this initializes the actions to be observation independent with a value of 0
            # which is a nice unbiased starting point for the policy network
            self.actor[-1].weight *= 1/100
        
        self.critic = self._mlp(critic_layer_sizes, activation).to(device)
        
    def _mlp(self, layer_sizes, activation):
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation())

        return nn.Sequential(*layers)
        
    def _get_policy(self, states):
        outputs = self.actor(states)
        sd_constant = self.initial_sd  + torch.log(1 - torch.exp(-self.initial_sd)) 

        dis = Normal(outputs[:,:self.num_actions], softplus(outputs[:,self.num_actions:] + sd_constant))
        
        return dis

    def get_action(self, states):
        pi = self._get_policy(states)
        
        # bound action into range -1 to 1
        action = torch.tanh(pi.sample()).squeeze()

        return  action.cpu().numpy()

    def get_log_prob(self, states, actions):
        pi = self._get_policy(states)
        
        return  pi.log_prob(torch.atanh(actions)).sum(-1)