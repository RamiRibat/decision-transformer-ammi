# AMMI Bootcamp II,  Summer 2021

## Decision Transformer AMMI

This is a re-implementation of **Decision Transformer** as a part of AMMI Bootcamp II. I chose only to re-implemnt the gym code, due to time constrain and limited experience in Atari tasks. I took some codes and scripts from the original repo (mentioned where taken); because I want to focus only on the algorithmic part. (I'm running the experiement now, no reported results yet)

**Paper:** Decision Transformer: Reinforcement Learning via Sequence Modeling, [Arxiv](https://arxiv.org/abs/2106.01345)
**Original repo:** [Github](https://github.com/kzl/decision-transformer)  |  **Slides:** [Link](https://docs.google.com/presentation/d/1UC4lRa7Rp1DrWDjl-jJEHkFddBdCLfoQgxj2x7oqVkg/edit?usp=sharing)  |  **W&B:** [Link](https://wandb.ai/aimsammi/dt-ammi?workspace=user-rami-ahmed)

## Installation

### Ubuntu 20.04
Create a new cond aenvironment:
```
conda create -n dt-gym-ammi python=3.8
```

Install the following python packages, using:
```
pip install numpy==1.20.3 torch==1.8.1 transformers==4.5.1 wandb==0.9.1 gym==0.18.3
```

Install [MuJoCo](http://www.mujoco.org/) and [mujoco-py](https://github.com/openai/mujoco-py):
```
sudo apt-get install ffmpeg

pip install mujoco-py==2.0.2.13
```

### MacOS Big Sur
Create a new cond aenvironment:
```
conda create -n dt-gym-ammi python=3.8
```

Install the following python packages, using:
```
pip install numpy==1.20.3 torch==1.8.1 transformers==4.5.1 wandb==0.9.1 gym==0.18.3
```

Install [MuJoCo](http://www.mujoco.org/) and [mujoco-py](https://github.com/openai/mujoco-py):
```
brew install ffmpeg gcc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

pip install mujoco-py==2.0.2.13
```


## Run an Experiement
To download offline datasets, you should install [D4RL](https://github.com/rail-berkeley/d4rl) and then run:

```
python data/d4rl_dataset.py
```

Configure your setting in `config/`, and then run:
```
python experiment.py -cfg <config file name-.py>
```
for example:
```
python experiment.py -cfg dt_gym_halfcheetah
```

## Results
Due to time limit and limited compute resources, I chose a subset of tasks to evaluate and validate my re-implementation: HalfCheetah, Walker, and Hopper for medium offline datasets. Check [W&B](https://wandb.ai/aimsammi/dt-ammi?workspace=user-rami-ahmed)

| Dataset | Environement | DT (mine) | DT (paper) |
| ------------- | ------------- | ------------- | ------------- |
| Medium | HalfCheetah | ? ± ? | 42.6 ± 0.1 |
| Medium | Hopper | ? ± ? | 67.6 ± 1.0 |
| Medium | Walker2d | ? ± ? | 74.0 ± 1.4 |

Next, I'll run the following:

| Dataset | Environement | DT (mine) | DT (paper) |
| ------------- | ------------- | ------------- | ------------- |
| Medium-Expert | HalfCheetah | ? ± ? | 68.8 ± 1.3 |
| Medium-Expert | Hopper | ? ± ? | 107.6 ± 1.8 |
| Medium-Expert | Walker2d | ? ± ? | 108.1 ± 0.2 |


Note: The above results are normalized scores for those tasks, to calculate the normalized score from the final return:

<img src="https://render.githubusercontent.com/render/math?math=norm\_score = \frac{score - min\_score}{max\_score - min\_score} * 100">

where the score is the return from the plot, and the min-max scores for the environments are in the following table:

| Environement | Min | Max |
| ------------- | ------------- | ------------- |
| HalfCheetah | -280.178953 | 12135.0 |
| Hopper | -20.272305| 3234.3 |
| Walker2d | 1.629008 | 4592.3 |

