# floofyredpanda

a joyful little wrapper that lets you feed a PyTorch `.pt` model any kind of input, from numbers and text files to wavs and raw bytes—and get back exactly the output you asked for and none of the tantrums... hopefully

# Disclaimer 
### The package will be updated alot untill 1.0.0 
- this will be to fix errors, or add new things
- after 1.0.0 i will still continue to release updates just not on a daily bases maybe more weekly
- 

## Features

- infinite positional inputs: numbers, lists, file-paths, bytes, you name it  

- output converters for `int`, `float`, `str`, `binary` (and any custom type you fancy)  
- wav file support out of the box (requires `soundfile`)  
- easy to extend: register your own converters for images, midi, pickles
## Installation
# Linux
```bash
 pip install floofyredpanda
```
# Windows
###  should be the same
```cmd
pip install floofyredpanda
```
## Uses
# Simple numeric inference
```python
import floofyredpanda as frp

# model expects a couple of floats, returns a float
result = frp.infer(0.1, 0.2, 'models/simple.pt')
print("prediction:", result)   # e.g. [0.305]
```
# Lists → list output
```python
from pathlib import Path
import floofyredpanda as frp

# vector inputs, default output as Python list
vector1 = [1.0, 2.0, 3.0]
vector2 = [0.5, 0.4, 0.1]
model_path = Path('models/vector_model.pt')

out = frp.infer(vector1, vector2, str(model_path))
print("vector output:", out)    # e.g. [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

```
# File input & custom converter
```python
import floofyredpanda as frp
import soundfile as sf

# register a WAV writer so we can get a .wav back
def wav_writer(tensor):
    data = tensor.cpu().numpy()
    sf.write('out.wav', data, samplerate=22050)
    return 'out.wav'

frp.register_output_converter('wav', wav_writer)

# feed in a .wav, ask for .wav out
output_file = frp.infer('input.wav', 'models/audio.pt', output_type='wav')
print("saved processed audio at", output_file)

```
)


# Training a tiny net
```python
import torch
import floofyredpanda as frp

# toy data: learn y = 2x
inputs  = [1.0, 2.0, 3.0, 4.0]
outputs = [2.0, 4.0, 6.0, 8.0]

model = frp.train(
    raw_inputs  = inputs,
    raw_outputs = outputs,
    layers      = [8, 4],          # two hidden layers
    activation  = 'relu',
    lr          = 0.01,
    epochs      = 50,
    batch_size  = 2,
)

# save & test
torch.save(model, 'models/x2net.pt')
print("saved to models/x2net.pt")

# run inference on unseen x
print("predict 5→", frp.infer(5, 'models/x2net.pt'))

```
# RL
### the code below might not work (it dosnt)
```python

import numpy as np
from floofyredpanda import rl, DQNAgent

# a trivial custom environment: you get +1 reward for picking action 0, episode ends at step 20
class ToyEnv:
    def __init__(self):
        self.action_size = 2
        self.state = 0
        self.max_steps = 20

    def reset(self):
        self.state = 0
        return [self.state]

    def step(self, action):
        reward = 1 if action == 0 else 0
        self.state += 1
        done = (self.state >= self.max_steps)
        # next state is just the step count
        return [self.state], reward, done, {}

def evaluate(env, agent, episodes=10):
    total = 0
    for _ in range(episodes):
        st = np.array(env.reset(), dtype=np.float32).reshape(1, -1)
        done = False
        ep_reward = 0
        while not done:
            a = agent.act(st)               # exploit mode (epsilon may be > 0)
            nxt, r, done, _ = env.step(a)
            ep_reward += r
            st = np.array(nxt, dtype=np.float32).reshape(1, -1)
        total += ep_reward
    print(f"avg reward over {episodes} eval eps: {total/episodes:.2f}")

def main():
    env = ToyEnv()

    # 1) train a fresh agent for 200 episodes
    print("training new agent…")
    agent = rl(env, 200)

    # 2) save it
    agent.save("toy_agent.pth")
    print("saved agent to toy_agent.pth (epsilon now:", agent.epsilon, ")")

    # 3) load into a new object
    print("loading agent back in…")
    agent2 = DQNAgent(state_size=1, action_size=env.action_size)
    agent2.load("toy_agent.pth")
    print("loaded agent epsilon:", agent2.epsilon)

    # 4) continue training that same agent for another 100 episodes
    print("continuing training…")
    agent2 = rl(env, 100, agent2)
    print("continuation done (epsilon now:", agent2.epsilon, ")")

    # 5) quick evaluation to see how it fares
    evaluate(env, agent2, episodes=20)

if __name__ == "__main__":
    main()

```
# update 0.4.6
- added new methods in frp.rl

# Licence

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.

