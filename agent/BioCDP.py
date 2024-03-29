import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from agent.model import MLP, Critic
from agent.diffusion import Diffusion
from agent.vae import VAE
from agent.helpers import EMA


class Actor(nn.Module):
    def __init__(self,args, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()

        self.actor_wrist = MLP(state_dim=state_dim, action_dim=8)
        self.wrist_embedding = nn.Linear(8, state_dim)
        self.actor_finger = Diffusion(state_dim=state_dim, action_dim=36, noise_ratio=args.noise_ratio,
                                   beta_schedule=args.beta_schedule, n_timesteps=args.n_timesteps)
        self.finger_embedding = nn.Linear(36, state_dim)
        self.actor_sustain = MLP(state_dim=state_dim, action_dim=1)
    
    def action_postprocessing(self, action_wrist, action_finger, action_sustain):
        if action_wrist.dim() == 1:
            action = torch.cat((action_wrist[:4], action_finger[:18], action_wrist[4:], action_finger[18:], action_sustain), dim=0)
        else:
            action = torch.cat((action_wrist[:,:4], action_finger[:,:18], action_wrist[:,4:], action_finger[:,18:], action_sustain), dim=1)
        return action

    def forward(self, state, eval=False):
        action_wrist = self.actor_wrist(state, eval)
        wrist_embedding = self.wrist_embedding(action_wrist.detach())
        action_finger = self.actor_finger(state + wrist_embedding, eval)
        finger_embedding = self.finger_embedding(action_finger.detach())
        action_sustain = self.actor_sustain(state + wrist_embedding + finger_embedding, eval)

        action = self.action_postprocessing(action_wrist, action_finger, action_sustain)
        return action

    def loss(self, action, state):
        action_wrist = self.actor_wrist(state, False)
        wrist_embedding = self.wrist_embedding(action_wrist.detach())
        action_finger = self.actor_finger(state + wrist_embedding, False)
        finger_embedding = self.finger_embedding(action_finger.detach())
        action_sustain = self.actor_sustain(state + wrist_embedding + finger_embedding, False)

        action_finger_data = torch.cat([action[:, 4:22], action[:, 26:-1]], dim=-1)
        loss_wrist = F.mse_loss(action_wrist, torch.cat([action[:, :4], action[:, 22:26]], dim=-1), reduction='mean')
        loss_finger = self.actor_finger.loss(action_finger_data, state + self.wrist_embedding(action_wrist))
        loss_sustain = F.mse_loss(action_sustain, action[:, -1], reduction='mean')
        return loss_wrist + loss_finger + loss_sustain

class BioCDP(object):
    def __init__(self,
                 args,
                 state_dim,
                 action_dim,
                 memory,
                 diffusion_memory,
                 device,
                 ):

        self.policy_type = args.policy_type
        if self.policy_type == 'BioCDP':
            self.actor = Actor(args,state_dim=state_dim, action_dim=action_dim).to(device)
        elif self.policy_type == 'Diffusion':
            self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, noise_ratio=args.noise_ratio,
                                   beta_schedule=args.beta_schedule, n_timesteps=args.n_timesteps).to(device)
        elif self.policy_type == 'VAE':
            self.actor = VAE(state_dim=state_dim, action_dim=action_dim, device=device).to(device)
        else:
            self.actor = MLP(state_dim=state_dim, action_dim=action_dim).to(device)

        # self.actor = torch.compile(self.actor)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.diffusion_lr, eps=1e-5)

        self.memory = memory
        self.diffusion_memory = diffusion_memory
        self.action_gradient_steps = args.action_gradient_steps

        self.action_grad_norm = action_dim * args.ratio
        self.ac_grad_norm = args.ac_grad_norm

        self.step = 0
        self.tau = args.tau
        self.actor_target = copy.deepcopy(self.actor)
        self.update_actor_target_every = args.update_actor_target_every

        self.critic = Critic(state_dim, action_dim).to(device)
        # self.critic = torch.compile(self.critic)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        self.action_dim = action_dim

        self.action_lr = args.action_lr

        self.device = device


        self.action_scale = 1.
        self.action_bias = 0.

    def append_memory(self, state, action, reward, next_state, mask):
        action = (action - self.action_bias) / self.action_scale
        
        self.memory.append(state, action, reward, next_state, mask)
        self.diffusion_memory.append(state, action)

    def sample_action(self, state, eval=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        action = self.actor(state, eval).cpu().data.numpy().flatten()

        action = action.clip(-1, 1)
        action = action * self.action_scale + self.action_bias
        return action

    def action_gradient(self, batch_size, log_writer):
        states, best_actions, idxs = self.diffusion_memory.sample(batch_size)

        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)


        for i in range(self.action_gradient_steps):
            best_actions.requires_grad_(True)
            q1, q2 = self.critic(states, best_actions)
            loss = -torch.min(q1, q2)

            actions_optim.zero_grad()

            loss.backward(torch.ones_like(loss))
            if self.action_grad_norm > 0:
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm, norm_type=2)

            actions_optim.step()

            best_actions.requires_grad_(False)
            best_actions.clamp_(-1., 1.)

        # if self.step % 10 == 0:
        #     log_writer.add_scalar('Action Grad Norm', actions_grad_norms.max().item(), self.step)

        best_actions = best_actions.detach()

        self.diffusion_memory.replace(idxs, best_actions.cpu().numpy())

        return states, best_actions

    def train(self, iterations, batch_size=256, log_writer=None):
        for _ in range(iterations):
            # Sample replay buffer / batch
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(states, actions)

            next_actions = self.actor_target(next_states, eval=False)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = (rewards + masks * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.ac_grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
                # if self.step % 10 == 0:
                #     log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            self.critic_optimizer.step()

            """ Policy Training """
            states, best_actions = self.action_gradient(batch_size, log_writer)

            actor_loss = self.actor.loss(best_actions, states)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.ac_grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
                # if self.step % 10 == 0:
                #     log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
            self.actor_optimizer.step()

            """ Step Target network """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.step % self.update_actor_target_every == 0:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))

