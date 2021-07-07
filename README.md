# PPO
Implementation of a multi-processed Proximal Policy Optimization (PPO) with some "implementation tricks" from the article [What Matters In On-Policy Reinforcement Learning?](https://arxiv.org/abs/2006.05990)

It achieves a score of around 300 (considered solved) on the [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) environment from OpenAI:
![Bipedal Walker GIF](bipedal_iteration250.gif)


### Update July 2021
Added the following features:
* Multi-processed workers for experience collection (resulting in 2-3x faster wall clock time).
* Model loading to allow warmstarting or continuing from a checkpoint.
* Proper seeding resulting in fully reproducible runs.
* Tensorboard logging.
