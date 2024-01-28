## Learning Playing Piano with Bionic-Constrained Diffusion Policy for Anthropomorphic Hand


*Practical implementations of Bionic-Constrained Diffusion Policy (Bio-CDP) for online model-free RL.*

## Experiments

### Requirements
Installations of [PyTorch](https://pytorch.org/) and [MuJoCo](https://github.com/deepmind/mujoco) are needed. 
A suitable [conda](https://conda.io) environment named `BioCDP` can be created and activated with:
```.bash
conda create BioCDP
conda activate BioCDP
```
To get started, install the additionally required python packages into you environment.
```.bash
pip install -r requirements.txt
```

### Running
Running experiments based our code could be quite easy:

```.bash
python main.py --env_name RoboPianist-debug-CMajorChordProgressionTwoHands-v0 --num_steps 10000000 --n_timesteps 100 --cuda 0 --seed 0
```

