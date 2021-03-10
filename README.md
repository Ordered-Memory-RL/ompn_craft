# Introduction
This is the codebase of the paper 
[Learning Task Decomposition with Order-Memory Policy Network](https://openreview.net/forum?id=vcopnwZ7bC)
It contains a version of the craft environment with gym wrappers.
Dependency
```
python3.6
torch==1.5.1
```
Install locally `pip install -e .` If you find the environment or the paper to be useful, please cite
```
@inproceedings{
lu2021learning,
title={Learning Task Decomposition with Order-Memory Policy Network},
author={Yuchen Lu and Yikang Shen and Siyuan Zhou and Aaron Courville and Joshua B. Tenenbaum and Chuang Gan},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=vcopnwZ7bC}
}
```
# Environments
```
import gym_psketch
print(gym_psketch.env_list)
```

# Test
Keyboard interactive mode. Use arrow keys to move and `u` to use. Other key triggers `done` action.
```
python scripts/enjoy.py -mode keyboard -env [ENV_NAME]
```

See rule-based bot.
```
python scripts/enjoy.py -mode demo -env [ENV_NAME]
```
More scripts see in `scripts`

# Train
Use ``main.py`` as main entry for both IL and RL
### Imitation
Generate demo
```
python main.py --mode demo \
	--envs <ENVS_NAME> \
	--demo_episodes 1500
```
Run OMPN on unsupervised task information
```
python main.py --mode IL --arch omstack \
	--flagfile ilflagfile \ 
	--nb_slots 3 \
	--cuda \
	--envs <ENVS_NAME> \
	--env_arch noenv
``` 
Run OMPN on unsupervised with sketch information
```
python main.py --mode IL --arch omstack \
	--flagfile ilflagfile \ 
	--nb_slots 3 \
	--cuda \
	--envs <ENVS_NAME> \
	--env_arch sketch
``` 
Visualize the learned expanding position by
```
python scripts/analysis.py --model_ckpt PATH_TO_PKL \
        --envs makebedfull-v0 --use_demo --episodes 20
```
