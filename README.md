[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"


### Introduction

Train an agent to navigate and collect bananas  

![Trained Agent][image1]

#### Goal: 

Collect as many yellow bananas as possible while avoiding blue bananas.

Reward:
 - +1: collecting a yellow banana
 - -1: collecting a blue banana
  

#### State Space
37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Instructions

To set up the environment please see [unity_instructions](./unity_instructions.md)

The main entry is `Navigation.ipynb`

### Contents

- [Navigation.ipynb](./Navigation.ipynb)
    - main development environment
- [trainer.py](./trainer.py)
    - Convenience wrapper to execute training of an agent in the environment
- [agent.py](./agent.py)
    - The convenience wrapper that uses the model (specified below) to learn and interact with the environment
- [model.py](./model.py)
    - The model used to predict actions from the environment
- [/params/best_params_15.pth](./params/best_params_15.pth)
    - model weights saved from a trained agent
- [scores.pkl](./scores.pkl)
    - log of scores over time during training run