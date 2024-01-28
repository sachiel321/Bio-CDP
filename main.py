import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import torch
from pathlib import Path

import time
from dataclasses import dataclass, asdict
import wandb
from agent.BioCDP import BioCDP
from agent.replay_memory import ReplayMemory, DiffusionMemory
import specs
import tqdm
from action_wrapper import CanonicalSpecWrapperv2

from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy')
    parser.add_argument('--env_name', default="RoboPianist-repertoire-150-ClairDeLune-v0",
                        help='Pianist environment (default: RoboPianist-debug-CMajorChordProgressionTwoHands-v0)')
    parser.add_argument('--seed', type=int, default=43, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--frame_stack', type=int, default=1, metavar='N',
                        help='frame stack (default: 1)')
    parser.add_argument('--eval_episodes', type=int, default=1, metavar='N',
                        help='eval episodes (default: 1)')
    

    parser.add_argument('--num_steps', type=int, default=3000000, metavar='N',
                        help='env timesteps (default: 1000000)')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--update_actor_target_every', type=int, default=1, metavar='N',
                        help='update actor target per iteration (default: 1)')

    parser.add_argument("--policy_type", type=str, default="Diffusion", metavar='S',
                        help="Diffusion, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", metavar='S',
                        help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=100, metavar='N',
                        help='diffusion timesteps (default: 100)')
    parser.add_argument('--diffusion_lr', type=float, default=0.0003, metavar='G',
                        help='diffusion learning rate (default: 0.0003)')
    parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
                        help='critic learning rate (default: 0.0003)')
    parser.add_argument('--action_lr', type=float, default=0.03, metavar='G',
                        help='diffusion learning rate (default: 0.03)')
    parser.add_argument('--noise_ratio', type=float, default=1.0, metavar='G',
                        help='noise ratio in sample process (default: 1.0)')

    parser.add_argument('--action_gradient_steps', type=int, default=20, metavar='N',
                        help='action gradient steps (default: 20)')
    parser.add_argument('--ratio', type=float, default=0.1, metavar='G',
                        help='the ratio of action grad norm to action_dim (default: 0.1)')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, metavar='G',
                        help='actor and critic grad norm (default: 1.0)')

    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')
    
    parser.add_argument('--addition_info', default='biocdp-clipaction-ClairDeLune',
                        help='addition info (default: )')

    return parser.parse_args()

def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}

def get_env(args, record_dir = None):
    env = suite.load(
        environment_name=args.env_name,
        seed=args.seed,
        stretch=1.0,
        shift=0,
        task_kwargs=dict(
            n_steps_lookahead=10,
            trim_silence=False,
            gravity_compensation=True,
            reduced_action_space=False,
            control_timestep=0.05,
            wrong_press_termination=False,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
            primitive_fingertip_collisions=False,
            change_color_on_activation=True,
        ),
    )
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=1,
            camera_id="piano/back",
            height=480,
            width=640,
        )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=1
        )
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=1
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    # if args.action_reward_observation:
    #     env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
    # env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = CanonicalSpecWrapperv2(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    return env

def evaluate(env, agent, writer, steps):
    episodes = 10
    returns = np.zeros((episodes,), dtype=np.float32)

    for i in range(episodes):
        state = env.reset()
        episode_reward = 0.
        done = False
        while not done:
            action = agent.sample_action(state, eval=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        returns[i] = episode_reward

    mean_return = np.mean(returns)

    writer.add_scalar(
            'reward/test', mean_return, steps)
    print('-' * 60)
    print(f'Num steps: {steps:<5}  '
              f'reward: {mean_return:<5.1f}')
    print('-' * 60)


def main(args=None):
    if args is None:
        args = readParser()

    device = torch.device(args.cuda)

    dir = "record"
    # dir = "test"
    # log_dir = os.path.join(dir, f'{args.env_name}', f'policy_type={args.policy_type}', f'ratio={args.ratio}', f'seed={args.seed}')
    run_name = f"SAC-{args.env_name}-{args.seed}-{time.time()}--{args.addition_info}"
    wandb.init(
        project="robopianist",
        entity='sachiel',# "icccr24",
        tags=([]),
        notes=None,
        config=args,
        mode="online",
        name=run_name,
    )

    # Create experiment directory.
    experiment_dir = Path('experiment') / run_name
    experiment_dir.mkdir(parents=True)

    # Initial environment
    env = get_env(args)
    eval_env = get_env(args, record_dir=experiment_dir / "eval")
    spec = specs.EnvironmentSpec.make(env)
    state_size = 1136
    action_size = 45
    print(action_size)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    memory_size = 1e6
    num_steps = args.num_steps
    start_steps = 5000
    eval_interval = 10000
    updates_per_step = 1
    batch_size = args.batch_size
    
    memory = ReplayMemory(state_size, action_size, memory_size, device)
    diffusion_memory = DiffusionMemory(state_size, action_size, memory_size, device)

    agent = BioCDP(args, state_size, action_size, memory, diffusion_memory, device)

    steps = 0
    episodes = 0
    timestep = env.reset()
    state = timestep.observation
    while steps < num_steps:
        episode_reward = 0.
        episode_steps = 0
        done = False
        # Reset episode.
        if timestep.last():
            wandb.log(prefix_dict("train", env.get_statistics()), step=steps)
            timestep = env.reset()
            state = timestep.observation
        episodes += 1
        while not done:
            if start_steps > steps:
                 action = spec.sample_action(random_state=env.random_state)
            else:
                action = agent.sample_action(timestep.observation, eval=False)
            timestep = env.step(action)
            next_state = timestep.observation
            if timestep.reward is not None:
                reward = timestep.reward
            else:
                reward = 0.0
            if timestep.last():
                done = True
            else:
                done = False

            mask = 0.0 if done else args.gamma
            steps += 1
            episode_steps += 1
            episode_reward += reward

            agent.append_memory(state, action, reward, next_state, mask)

            if steps >= start_steps:
                agent.train(updates_per_step, batch_size=batch_size, log_writer=None)

            # if steps % eval_interval == 0:
            #     evaluate(env, agent, writer, steps)
            #     # self.save_models()
            #     done =True

            # Eval.
            if steps % eval_interval == 0:
                for _ in range(args.eval_episodes):
                    timestep = eval_env.reset()
                    while not timestep.last():
                        eval_action = agent.sample_action(timestep.observation, eval=True)
                        temp_action = eval_action
                        # temp_action = action_post_process(eval_action, index_map)

                        timestep = eval_env.step(temp_action)
                log_dict = prefix_dict("eval", eval_env.get_statistics())
                music_dict = prefix_dict("eval", eval_env.get_musical_metrics())
                wandb.log(log_dict | music_dict, step=steps)
                video = wandb.Video(str(eval_env.latest_filename), fps=4, format="mp4")
                wandb.log({"video": video, "global_step": steps})
                eval_env.latest_filename.unlink()

            state = next_state

        # if episodes % log_interval == 0:
        #     writer.add_scalar('reward/train', episode_reward, steps)

        print(f'episode: {episodes:<4}  '
            f'episode steps: {episode_steps:<4}  '
            f'reward: {episode_reward:<5.1f}')


if __name__ == "__main__":
    main()
