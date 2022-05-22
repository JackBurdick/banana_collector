[//]: # (Image References)

[training_plot]: ./assets/training_plot.png "training_plot"
[smoothed_training_plot]: ./assets/smoothed_training_plot.png "training_plot"


# Banana collector

## Context

The goal environment information can be found in the main [readme.md](./README.md)


## Description

Below is an overview of:
 - loss plots
 - model
 - agent

### Loss Plots

The raw loss plot (below) is a little challenging to interpret.

![Reward over time][training_plot]

The smoothed version (below) shows a nice/desireable curve that shows rewards
increasing over time.

![Smoothed reward over time][smoothed_training_plot]

### Model

The model itself isn't terribly exciting, largely a standard DNN

The core model consists of a number of linear layers of pre-specified units in
this case `[64, 32, 16, 8]`

```python
# input and hidden
units = [state_size, *hidden_units]
layer_list = []
for i, u in enumerate(units):
    if i != 0:  # skip first
        layer_list.append(nn.Linear(units[i - 1], u))
self.hidden_layers = nn.ModuleList(layer_list)
```

Followed by corresponding branches for the advantage and state:

```python
# output
self.advantage_hidden = nn.Linear(units[-1], units[-1])
self.output_advantage_values = nn.Linear(units[-1], action_size)

# output state values scalar
self.state_values_hidden = nn.Linear(units[-1], units[-1])
self.output_state_values = nn.Linear(units[-1], 1)
```

The 'interesting' bit is how the output is determined. Rather than simply output
the values, both an action advantage (state-dependent) and a state value is
estimated -- this is detailed in [Dueling Network Architectures for Deep
Reinforcement Learning](https://arxiv.org/abs/1511.06581). As described in the
paper, this allows the model to learn which states are valuable (or not!),
without having to learn the effect of an action for each state.

When called the hidden layers are wrapped in a non-linearity (elu) and returned

```python
def forward(self, state):
    """Model inference"""
    x = state
    for layer in self.hidden_layers:
        x = F.elu(layer(x))

    # action value estimation
    av = F.elu(self.advantage_hidden(x))
    av = self.output_advantage_values(av)

    # state value estimation
    sv = F.elu(self.state_values_hidden(x))
    sv = self.output_state_values(sv)

    # originally formulated as a max, then converted to
    # mean for more stability. a softmax was also attempted
    # but yielded similar results to mean + was more complex
    out = sv + (av - av.mean())

    return out
```

### Agent

The agent consists of a couple components worth mentioning:
 - Prioritized Experience Replay
 - Local and Target network (and corresponding soft update)
 - beta
 - Epsilon-greedy action selection

#### beta

```python
self.beta = self.beta + (self.beta * 0.01)
self.beta = min(1.0, self.beta)
if w is not None:
    rbmv = 1 / len(self.replaybuff.memory)
    # sum to 1
    w = w / w.sum(axis=0, keepdims=1)
    w += 0.00001  # avoid divide by zero
    inv_w = 1 / w
    mult = rbmv * inv_w
    powered = np.power(mult, self.beta)
    powered = powered.astype(np.float32)
    loss = torch.from_numpy(powered) * loss
```


#### Prioritized Experience Replay

It's easiest to look at the `ReplayBuffer` class in [agent.py](./agent.py). But
a replay buffer is used. The replay of experiences is then prioritized by some
criteria ([Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)).
There are a couple considerations to how to select which experiences should be
prioritized, the idea being "replay important transitions more frequently, and
therefore learn more efficiently".  I took the following approach, and talk a
little more about this in the future directions section.

```python
def sample(self):
    """Sample a batch of experiences from memory.

    stack all entities in each var for batch training
    (e.g. rather than return a sample of tuples)
    """

    if random.random() < self.p_weighted:
        # select experiences from memory with either large or small differences between
        # target and expected

        # weights of each experience to bias sampling, weigh on difference between exp vs obs
        w = np.asarray([m.diff for m in self.memory if m is not None])

        # decide whether to bias sampling from 'surprising', 'unsurprising', or 'random' experiences
        if random.random() > self.p_surprise:
            # invert probability to be more likely to choose events with small differences
            # this is a similar idea to addressing the issue of never sampling 'unsurprising' experiences
            # (other solutions are adding a small value to differences near 0)
            w = w - np.max(w)
            w = np.abs(w)
        # else:
        #     # w is now weighted by positive rewards (will sample high reward actions)
        #     w = np.asarray([m.reward for m in self.memory if m is not None])
        #     # choose based on good/bad, reward
        #     if random.random() > self.p_good:
        #         # invert probability to be more likely to choose events with small differences
        #         # this is a similar idea to addressing the issue of never sampling 'unsurprising' experiences
        #         # (other solutions are adding a small value to differences near 0)
        #         w = w - np.max(w)
        #         w = np.abs(w)

        # return index and experiences, with weighted sampling
        # larger values will be sampled more frequently
        tups = random.choices(
            list(enumerate(self.memory)),
            weights=w,
            k=self.batch_size,
        )

        # the index of the experience and the experience named tuple
        indexes, experiences = zip(*tups)
    else:
        # uniform sample from all experiences
        tups = random.sample(list(enumerate(self.memory)), k=self.batch_size)
        indexes, experiences = zip(*tups)
        w = None

    # increase quadratically then clip to self.p_weighted_max
    self.p_weighted = min(
        self.p_weighted + (self.p_weighted * self.p_weighted_growth),
        self.p_weighted_max,
    )
```


#### Local and Target network (and corresponding soft update)

Select best action with one set of params and evaluate that action with a
different set of parameters, this is best described here: [Deep Reinforcement
Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

Update procedure in pytorch:
```python
# pair params for easy access
paired_params = zip(target_model.parameters(), local_model.parameters())

# tau: interpolation parameter

# iterate params and update target params with tau 
# regulated combination of local and target
for target_param, local_param in paired_params:
    target_param.data.copy_(
        tau * local_param.data + (1.0 - tau) * target_param.data
    )
```


#### Epsilon-greedy action selection

```python
# Epsilon-greedy action selection
if random.random() > eps:
    # exploit: select best
    action = np.argmax(action_values.cpu().data.numpy())
else:
    # explore: perform random choice
    action = random.choice(np.arange(self.action_size))
```

governed by:
```python
self.eps = max(self.eps_end, self.eps_decay * self.eps)
```

in this case:
```python
eps_init=1.0,
eps_end=0.01,
eps_decay=0.995,
```

## Future Work

A few ideas:
 - Improve the agent
 - learn from the environment directly (raw pixels)

### Improving the Agent

Model
> The model that takes input and produces outputs is very basic. Maybe this is
> ok, but maybe it would be worthwhile to spend some more time here. It's
> possible that we could build useful representations from the related inputs,
> as opposed to simply treating them all as independent/equally important
> observations. However, I didn't spend enough time with the data to make any
> strong suggestions here.

Experience replay buffer sampling
> I haven't read the literature yet, but I would imagine there are "better" ways
> of sampling experiences. I can imagine an approach where we "embed"
> experiences and sample from clusters according to some logic.



### Learn from the Environment

Learn from Raw Pixels
> rather than use the supplied values from the environment, it would be more
> interesting to learn from the environment (raw pixels) directly. This would
> involve creating (likely learned) models to supply information to the agent.
> This could be challenging in that we're now training multiple "models" at the
> same time. Additionally, when learning from the raw pixels, we're not exactly
> sure what features are most useful. I think in this case it's easy to say
> "blue" banana vs "yellow" banana, but in more advanced environments this could
> become even more challenging.