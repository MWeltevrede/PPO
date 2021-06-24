import numpy as np
import torch
import torch.multiprocessing as mp
from queue import Empty
from torch.optim import Adam
from utils import discount_cumsum, update_single, update_sets, finalize
import math, os, pickle, shutil, time, logging, traceback, random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class WorkerProcess(mp.Process):
    def __init__(self, env_constructor, model_constructor, experience_queue, parameters, seed, p_id, buffer_size, max_ep_len, gamma, lam, input_norm):
        super(WorkerProcess, self).__init__()
        
        self.env_constructor = env_constructor
        self.model_constructor = model_constructor
        self.seed = seed
        self.experience_queue = experience_queue
        self.parameters = parameters
        self.p_id = p_id
        self.buffer_size = buffer_size
        self.max_ep_len = max_ep_len
        self.gamma = gamma
        self.lam = lam
        self.input_norm = input_norm
        
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        
    def run(self):
        env = self.env_constructor()

        torch.manual_seed(self.seed + self.p_id)
        np.random.seed(self.seed + self.p_id)
        random.seed(self.seed + self.p_id)
        env.seed(self.seed + self.p_id)
        
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        actor_critic = self.model_constructor(torch.device('cpu'))
        actor_critic.eval()
        
        while True:
            # update model parameters
            while self.p_id not in self.parameters:
                time.sleep(0.0001)
            parameters, input_aggregate = self.parameters[self.p_id]
            del self.parameters[self.p_id]
            actor_critic.load_state_dict(parameters)
            acting_aggregate = input_aggregate.copy()
            
            experience, new_aggregate = self._collect_experience(env, actor_critic, acting_aggregate)
            self.experience_queue.put((self.p_id, (experience, new_aggregate)))
            
            
    def _normalize_input(self, acting_aggregate, states):
        """
            Normalize the input per dimension based on the entire 
            history of inputs seen so far
        """
        for i in range(len(acting_aggregate)):
            mean, var = finalize(acting_aggregate[i])
            states[:,i] = (states[:,i] - mean) / max(math.sqrt(var), 1e-6)
        return states
    
    def _update_aggregate(self, input_aggregate, state):
        """
            Update the aggregate used for input normalization
        """
        input_aggregate = [update_single(input_aggregate[i], state[i]) for i in range(len(input_aggregate))]
        return input_aggregate
            
        
    def _collect_experience(self, env, actor_critic, acting_aggregate):
        batch_states = []
        batch_actions = []
        rewards = []
        batch_returns = []
        batch_lengths = []
        batch_wins = []
        batch_rtg = []
        batch_GAE = []
        values = []

        new_aggregate = [(0,0,0) for _ in range(len(acting_aggregate))]

        # Collect experience
        total_steps = 0
        step = 0
        state = env.reset()
        
        total_time_waiting = 0

        while True:
            step += 1
            total_steps += 1
            
            if self.input_norm:
                acting_aggregate = self._update_aggregate(acting_aggregate, state)
                new_aggregate = self._update_aggregate(new_aggregate, state)
            
            with torch.inference_mode():
                state_tensor = torch.as_tensor(np.array(state), dtype=torch.float32, device='cpu')
    #           # add batch dimension
                state_tensor = state_tensor.unsqueeze(0)
        
                if self.input_norm:
                    state_tensor = self._normalize_input(acting_aggregate, state_tensor)

                action = actor_critic.get_action(state_tensor)
                value = actor_critic.critic(state_tensor).item()

            obs, reward, done, _ = env.step(action)

            batch_states.append(state.copy())
            batch_actions.append(action)

            values.append(value)
            rewards.append(reward)

            state = obs

            timeout = (step == self.max_ep_len or total_steps == self.buffer_size)
            terminal = timeout or done

            # finish the episode
            if terminal:
                if timeout:
                    with torch.inference_mode():
                        obs_tensor = torch.as_tensor(np.array(obs), dtype=torch.float32, device='cpu')
                        # add batch dimension
                        obs_tensor = obs_tensor.unsqueeze(0)
                        
                        if self.input_norm:
                            obs_tensor = self._normalize_input(acting_aggregate, obs_tensor)
                        
                        last_val = actor_critic.critic(obs_tensor).item()
                elif done:
                    last_val = 0
                    if reward <= -100:
                        # I lost
                        batch_wins.append(0)
                    else:
                        # I won
                        batch_wins.append(1)

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

                state = env.reset()

                step = 0
                rewards = []
                values = []

                if total_steps == self.buffer_size:
                    break
                    
        return {"s":batch_states, "a":batch_actions, "r":batch_returns, "l":batch_lengths, "rtg":batch_rtg, "gae":batch_GAE, "wins":batch_wins}, new_aggregate


class PPO():
    """
    Description: 
        Proximal Policy Optimization (the clipping variant) implemented in PyTorch.
    """
    
    def __init__(self, env_constructor, ac_constructor, buffer_size=4096, max_steps=int(1e6), gamma=0.99,
                 clip_ratio=0.2, lr=3e-4, epochs=10, batch_size=64,
                 lam=0.97, save_freq=10, save_path="models", log_path="tensorboard/ppo", device=torch.device('cpu'), 
                 input_normalization=True, time_feature=False, max_ep_len=1000, num_workers=1, seed=0):
        """
        Args:
            env_constructor: A constructor function for an environment that follows the OpenAI Gym API
            
            ac_constructor: A constructor function that takes the following arguments:
                ===========        ======================================
                Symbol             Description
                ===========        ======================================
                ``device``         | Torch.device() to run this model on
                ===========        ======================================
            
                It should create a PyTorch Actor-Critic instance with the following modules:
                ===========  ======================================
                Symbol       Description
                ===========  ======================================
                ``actor``    | PyTorch Module, containing
                             | the Actor (or Policy) network.
                ``critic``   | PyTorch Module, containing
                             | the critic (or Value) network.
                ===========  ======================================
                
                That also contains the following functions:
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
            
            num_workers (int): Number of parallel cpu workers that collect experience.
            
            seed (int): Random number generator seed used by the workers.
        """
        self.env_constructor = env_constructor
        self.ac_constructor = ac_constructor
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
        temp_env = env_constructor()
        self.obs_dim = temp_env.observation_space.shape
        self.num_workers = num_workers
        
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        logger = mp.log_to_stderr()
        logger.setLevel(logging.WARNING)
        n_cpus = mp.cpu_count()
        if num_workers > n_cpus:
            print(f"Warnning! You specified {num_workers} workers with {n_cpus} cpus.")
            
        assert buffer_size % num_workers == 0, "Error! Buffer size is not divisible by number of workers."
        
        
        if not self.buffer_size % self.batch_size == 0 or not self.buffer_size / self.batch_size > 1:
            print("Warning! Make sure buffer size is a multiple of batch size.")
        
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
            
        self.actor_critic = ac_constructor(self.device)
        
        self.actor_optimizer = Adam(self.actor_critic.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.actor_critic.critic.parameters(), lr=lr)
        self.critic_loss = torch.nn.MSELoss()
        
        
        self.time_feature = time_feature
        if time_feature:
            self.obs_dim[0] += 1
        
        self.input_norm = input_normalization
        if input_normalization:
            # aggragate used for input normalization
            self.input_aggregate = [(0, 0, 0) for _ in range(self.obs_dim[0])]
            
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
        for i in range(self.obs_dim[0]):
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
    
    def _update_aggregate(self, new_aggregate):
        """
            Update the aggregate used for input normalization
        """
        self.input_aggregate = [update_sets(self.input_aggregate[i], new_aggregate[i]) for i in range(self.obs_dim[0])]
        

    def _train_epoch(self, step_id):
        """
            Train for single epoch
        """
        worker_states = [None]*self.num_workers
        worker_actions = [None]*self.num_workers
        worker_returns = [None]*self.num_workers
        worker_lengths = [None]*self.num_workers
        worker_rtg = [None]*self.num_workers
        worker_GAE = [None]*self.num_workers
        worker_wins = [None]*self.num_workers
        worker_aggregates = [None]*self.num_workers
        for _ in range(self.num_workers):
            # this blocks until experience is available
            p_id, output = self.experience_queue.get()
            exp, new_aggregate = output
                    
            worker_aggregates[p_id] = new_aggregate
            worker_states[p_id] = exp['s']
            worker_actions[p_id] = exp['a']
            worker_returns[p_id] = exp['r']
            worker_lengths[p_id] = exp['l']
            worker_rtg[p_id] = exp['rtg']
            worker_GAE[p_id] = exp['gae']
            worker_wins[p_id] = exp['wins']
            
        batch_states = []
        batch_actions = []
        batch_returns = []
        batch_lengths = []
        batch_rtg = []
        batch_GAE = []
        batch_wins = []
        for i in range(self.num_workers):
            if self.input_norm:
                self._update_aggregate(worker_aggregates[i])
            batch_states = batch_states + worker_states[i]
            batch_actions = batch_actions + worker_actions[i]
            batch_returns = batch_returns + worker_returns[i]
            batch_lengths = batch_lengths + worker_lengths[i]
            batch_rtg = batch_rtg + worker_rtg[i]
            batch_GAE = batch_GAE + worker_GAE[i]
            batch_wins = batch_wins + worker_wins[i]
        
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
            idxs = np.random.permutation(self.buffer_size)

            # perform mini-batch updates
            i = 0
            while i < self.buffer_size:
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

        self._update_worker_networks()
        return self.buffer_size, np.array(losses).mean(), np.array(kls).mean(), batch_returns, batch_lengths, batch_wins
    
    def _update_worker_networks(self):
        if self.device == torch.device('cuda'):
            self.actor_critic.cpu()
            
        if self.input_norm:
            aggregate = self.input_aggregate
        else:
            aggregate = [(0, 0, 0) for _ in range(self.obs_dim[0])]
            
        state_dict = self.actor_critic.state_dict()
        for p_id in self.p_ids:
            self.parameters[p_id] = (state_dict, aggregate)
            
        if self.device == torch.device('cuda'):
            self.actor_critic.cuda()

    def train(self, print_freq=100):
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
        
        try:
            self.p_ids = range(self.num_workers)
            self.manager = mp.Manager()
            self.experience_queue = self.manager.Queue(self.num_workers)
            self.parameters = self.manager.dict()
            self.processes = [WorkerProcess(self.env_constructor, 
                                            self.ac_constructor, 
                                            self.experience_queue, 
                                            self.parameters, 
                                            self.seed, 
                                            p_id, 
                                            self.buffer_size // self.num_workers, 
                                            self.max_ep_len, 
                                            self.gamma, 
                                            self.lam, 
                                            self.input_norm)
                             for p_id in self.p_ids]
            
            for i in range(self.num_workers):
                self.processes[i].start()
        
            self._update_worker_networks()
            
            while step_id < self.max_steps:
                iteration += 1

                steps, mean_loss, mean_kl, returns, lengths, wins = self._train_epoch(step_id)

                step_id += steps
                avg_return += np.mean(returns) / print_freq
                avg_length += np.mean(lengths) / print_freq
                avg_loss += mean_loss / print_freq
                avg_kl += mean_kl / print_freq

                if len(wins) > 0:
                    self.writer.add_scalar("win ratio", sum(wins) / len(wins), step_id)
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
                    if self.input_norm:
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
            
            for i in range(self.num_workers):
                self.processes[i].terminate()
            for i in range(self.num_workers):
                self.processes[i].join()
            
        except Exception as e:
            print(f"Exception caught: {e}")
            traceback.print_exc()
            print(f"Safely closing all processes")
            self.writer.close()
            for i in range(self.num_workers):
                self.processes[i].terminate()
            for i in range(self.num_workers):
                self.processes[i].join()
                print(f"Worker {i} has closed")

