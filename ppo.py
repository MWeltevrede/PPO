import numpy as np
import torch
from torch.optim import Adam
from utils import discount_cumsum, update, finalize
import math, os, pickle, shutil, time
from torch.utils.tensorboard import SummaryWriter

class PPO():
    """
    Description: 
        Proximal Policy Optimization (the clipping variant) implemented in PyTorch.
    """
    
    def __init__(self, env, actor_critic, buffer_size=4096, max_steps=int(1e6), gamma=0.99,
                 clip_ratio=0.2, lr=3e-4, epochs=10, batch_size=64,
                 lam=0.97, save_freq=10, save_path="models", log_path="tensorboard/ppo", device=torch.device('cpu'), 
                 input_normalization=True, time_feature=False, max_ep_len=1000):
        """
        Args:
            env: An environment instance that follows the OpenAI Gym API
            
            actor_critic: A PyTorch Actor-Critic instance with the following modules:
                ===========  ======================================
                Symbol       Description
                ===========  ======================================
                ``actor``    | PyTorch Module, containing
                             | the Actor (or Policy) network.
                ``critic``   | PyTorch Module, containing
                             | the critic (or Value) network.
                ===========  ======================================
                
                It also should contain the following functions:
                ================  =============================  =============================
                Symbol            Input                          Return
                ================  =============================  =============================
                ``get_action``    | Torch tensor corresponding   | Numpy array of actions for
                                  | to a batch of states.        | each state.                 
                ``get_log_prob``  | Torch tensor corresponding   | Torch tensor of log 
                                  | to a batch of states and     | probabilities for the batch
                                  | Torch tensor corresponding   | of actions.
                                  | to a batch of actions.       |
                ================  =============================  =============================
                
            device: PyTorch device to be used.
            
            buffer_size (int): Number of environment steps to buffer per training iteration.
                
            max_steps (int): Maximum number of environment steps to train for.
                
            gamma (float): Discount factor.
            
            clip_ratio (float): PPO hyperparameter for clipping policy objective.
                
            lr (float): Learning rate for the policy and value function optimization.
            
            epochs (int): Number of epochs to train the policy and value network for during each training iteration.
            
            batch_size (int): Mini-batch size used for gradient descent.
                
            lam (float): Generalized Advantage Estimation hyperparameter.

            save_freq (int): Frequency (epochs) with which to save the model.
            
            save_path (string): Directory in which to save the models.
            
            log_path (string): Directory in which to log the tensorboard scalars.
                               Note that it will overwrite any logs already in this directory.
            
            input_normalization (bool): Whether or not to use input normalization.
            
            time_feature (bool): Whether to add the time remaining untill the end of the episode (as defined by max_ep_len) to the observation.
            
            max_ep_len (int): Maximum length of an episode.
        """
        self.env = env
        self.actor_critic = actor_critic
        self.buffer_size = buffer_size
        self.max_steps = max_steps
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.lam = lam
        self.save_freq = save_freq
        self.save_path = save_path
        self.device = device
        self.max_ep_len = max_ep_len
        self.obs_dim = env.observation_space.shape[0]
        
        if not self.buffer_size % self.batch_size == 0 or not self.buffer_size / self.batch_size > 1:
            print("Warning! Make sure buffer size is a multiple of batch size.")
        
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        
        self.actor_optimizer = Adam(self.actor_critic.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.actor_critic.critic.parameters(), lr=lr)
        self.critic_loss = torch.nn.MSELoss()
        
        
        self.time_feature = time_feature
        if time_feature:
            self.obs_dim += 1
        
        self.input_norm = input_normalization
        if input_normalization:
            # aggragate used for input normalization
            self.input_aggregate = [(0, 0, 0) for _ in range(self.obs_dim)]
            
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        self.writer = SummaryWriter(log_dir=log_path)
        
    def _policy_loss(self, states, actions, advantage, logp_old):
        """
            Calculate the clipped policy loss for Proximal Policy Optimization (PPO)
        """
        logp = self.actor_critic.get_log_prob(states, actions)
        ratio = torch.exp(logp - logp_old)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss = -(torch.min(ratio * advantage, clipped_ratio * advantage)).mean()

        kl_estimate = (logp_old - logp).mean().item()
        return loss, kl_estimate
    
    def _normalize_input(self, states):
        """
            Normalize the input per dimension based on the entire 
            history of inputs seen so far
        """
        for i in range(self.obs_dim):
            mean, var = finalize(self.input_aggregate[i])
            states[:,i] = (states[:,i] - mean) / max(math.sqrt(var), 1e-6)
        return states
    
    def _add_time_feature(self, obs, t):
        """
            Add remaining time before the end of an episode (as defined by maximum episode length) 
            to the observations.
        """
        new_obs = np.append(obs, np.array([self.max_ep_len - t], dtype=obs.dtype))
        return new_obs
    
    def _update_aggregate(self, state):
        """
            Update the aggregate used for input normalization
        """
        self.input_aggregate = [update(self.input_aggregate[i], state[i]) for i in range(self.obs_dim)]

    def _train_epoch(self):
        """
            Train for single epoch
        """
        batch_states = []
        batch_actions = []
        rewards = []
        batch_returns = []
        batch_lengths = []
        batch_rtg = []
        batch_GAE = []
        values = []

        # Collect experience
        total_steps = 0
        step = 0
        state = self.env.reset()
        if self.time_feature:
            state = self._add_time_feature(state, step)
            
        while True:
            step += 1
            total_steps += 1
            
            if self.input_norm:
                self._update_aggregate(state)
                    
            state_tensor = torch.as_tensor(np.array(state), dtype=torch.float32, device=self.device)
            # add batch dimension
            state_tensor = state_tensor.unsqueeze(0)
            
            if self.input_norm:
                state_tensor = self._normalize_input(state_tensor)
            
            action = self.actor_critic.get_action(state_tensor)

            obs, reward, done, _ = self.env.step(action)

            if self.time_feature:
                obs = self._add_time_feature(obs, step)

            batch_states.append(state.copy())
            batch_actions.append(action)
            
            values.append(self.actor_critic.critic(state_tensor).item())
            rewards.append(reward)
            
            state = obs
            
            timeout = step == self.max_ep_len
            terminal = timeout or done
            
            # finish the episode
            if terminal:
                if timeout:
                    obs_tensor = torch.as_tensor(np.array(obs), dtype=torch.float32, device=self.device)
                    # add batch dimension
                    obs_tensor = obs_tensor.unsqueeze(0)

                    if self.input_norm:
                        obs_tensor = self._normalize_input(obs_tensor)

                    last_val = self.actor_critic.critic(obs_tensor).item()
                elif done:
                    last_val = 0
                    
                # allow for value/reward bootstrapping if episode is prematurely terminated
                r = np.append(np.array(rewards), last_val)
                v = np.append(np.array(values), last_val)
                
                # calculate TD delta = reward + GAE_gamma * V(obs) − V(state)
                deltas = r[:-1] + self.gamma * v[1:] - v[:-1]
                
                # log unbootstrapped returns and lengths
                ret, ep_length = sum(rewards), len(rewards)
                batch_returns.append(ret)
                batch_lengths.append(ep_length)
                
                # calculate Rewards-to-go for every step in this episode
                batch_rtg = batch_rtg + list(discount_cumsum(r, self.gamma)[:-1])

                # calculate GAE for every step in this episode
                batch_GAE = batch_GAE + list(discount_cumsum(deltas, self.gamma*self.lam))

                state = self.env.reset()
                if self.time_feature:
                    state = self._add_time_feature(state, step)
                    
                step = 0
                rewards = []
                values = []

                if total_steps > self.buffer_size:
                    break

        # PPO update
        states = torch.as_tensor(np.array(batch_states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(batch_actions), dtype=torch.float32, device=self.device)

        if self.input_norm:
            states = self._normalize_input(states)

        logp_old = self.actor_critic.get_log_prob(states, actions).detach()
        
        # train policy and value function for several epochs
        losses = []
        kls = []
        for epoch in range(self.epochs):
            # shuffle individual transitions:
            idxs = np.random.permutation(total_steps)

            # perform mini-batch updates
            i = 0
            while i < total_steps:
                mbatch_idxs = idxs[i:i+self.batch_size]
                
                # update policy
                states = torch.as_tensor(np.array(batch_states), dtype=torch.float32, device=self.device)[mbatch_idxs]
                actions = torch.as_tensor(np.array(batch_actions), dtype=torch.float32, device=self.device)[mbatch_idxs]

                if self.input_norm:
                    states = self._normalize_input(states)

                GAE = torch.as_tensor(np.array(batch_GAE), dtype=torch.float32, device=self.device)[mbatch_idxs]
                std, mean = torch.std_mean(GAE)
                GAE = (GAE - mean) / std

                self.actor_optimizer.zero_grad()
                loss, kl = self._policy_loss(states, actions, GAE, logp_old[mbatch_idxs])
                    
                losses.append(loss.item())
                kls.append(kl)
            
                loss.backward()
                self.actor_optimizer.step()
                
                # update value function
                states = torch.as_tensor(np.array(batch_states), dtype=torch.float32, device=self.device)[mbatch_idxs]
                rtg = torch.as_tensor(np.array(batch_rtg), dtype=torch.float32, device=self.device)[mbatch_idxs]

                if self.input_norm:
                    states = self._normalize_input(states)

                self.critic_optimizer.zero_grad()
                vf_loss = self.critic_loss(torch.squeeze(self.actor_critic.critic(states), -1), rtg)
                vf_loss.backward()
                self.critic_optimizer.step()
                

                i += self.batch_size


        return total_steps, np.array(losses).mean(), np.array(kls).mean(), batch_returns, batch_lengths

    def train(self, print_freq=10):
        """
            Train the agent using Proximal Policy Optimization
        """
        avg_return = 0
        avg_length = 0
        avg_loss = 0
        avg_kl = 0
        
        step_id = 0
        iteration = 0
        start = time.time()
        
        
        while step_id < self.max_steps:
            iteration += 1
            
            steps, mean_loss, mean_kl, returns, lengths = self._train_epoch()
            
            step_id += steps
            avg_return += np.mean(returns) / print_freq
            avg_length += np.mean(lengths) / print_freq
            avg_loss += mean_loss / print_freq
            avg_kl += mean_kl / print_freq
            
            self.writer.add_scalar("return", np.mean(returns), step_id)
            self.writer.add_scalar("episode length", np.mean(lengths), step_id)
            self.writer.add_scalar("loss", mean_loss, step_id)
            self.writer.add_scalar("kl", mean_kl, step_id)
            time_elapsed = time.time() - start
            self.writer.add_scalar("speed", step_id/time_elapsed, step_id)
            
            if iteration % self.save_freq == 0 and not iteration == 0:
                print(f"Saved model at iteration: {iteration}")
                
                # save model
                filename = self.save_path + f"/iteration{int(iteration)}.pt"
                torch.save(self.actor_critic.actor.state_dict(), filename)
                
                # save input aggregate
                filename = self.save_path + f"/aggregate{int(iteration)}.p"
                with open(filename, 'wb') as filehandler:
                    pickle.dump(self.input_aggregate, filehandler)
            
            if iteration % print_freq == 0 and not iteration == 0:
                print(f"Iteration: {iteration}, steps: {step_id}, mean loss: {avg_loss}, mean KL divergence: {avg_kl} average return: {avg_return}, average episode length: {avg_length}")
                avg_return = 0
                avg_length = 0
                avg_loss = 0
                avg_kl = 0
                
        self.writer.close()