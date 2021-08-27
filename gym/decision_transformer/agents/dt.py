# Imports
## ML & RL
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import Embedding, Linear, LayerNorm, Sequential, Tanh, MSELoss
import transformers
## DT-AMMI
from .traj_gptx.gpt2 import GPT2Model


class DecisionTransformer(nn.Module):
    """
    Causal Decision Transformer: (R2Gt, st) --> (at)
        traj = (R2G0, s0, a0,, ..., R2Gt, st, at, ..., sT)
    """
    def __init__(self, state_dim, act_dim, Ni, Nt, dt_config):
        print('Initialize Decision Transformer!')
        super().__init__()
        self.dt_config = dt_config
        self.state_dim = state_dim
        self.act_dim = act_dim
        emb_dim = dt_config['emb_dim']
        self.emb_dim = emb_dim
        # self.device = self.config['experiment']['device']

        gpt_args = dict(n_layer = dt_config['nLayers'],
                        n_head = dt_config['nHeads'],
                        n_inner = 4*emb_dim,
                        n_positions = dt_config['positions'],
                        activation_function = dt_config['act_fn'],
                        resid_pdrop = dt_config['dropout'],
                        attn_pdrop = dt_config['dropout'])

        gpt_config = transformers.GPT2Config(vocab_size=1, n_embd=emb_dim, **gpt_args)

        # Causal Transformer
        self.transformer = GPT2Model(gpt_config)
        
        self.emb_t = nn.Embedding(1000, emb_dim)
        self.emb_R2G = Linear(1, emb_dim)
        self.emb_s = Linear(state_dim, emb_dim)
        self.emb_a = Linear(act_dim, emb_dim)

        self.emb_ln = LayerNorm(emb_dim)

        self.next_act = Sequential(*[Linear(emb_dim, act_dim), Tanh()])

        optimizer = 'th.optim.' + dt_config['optimizer']
        schedular = 'th.optim.lr_scheduler.' + dt_config['scheduler']
        self.optimizer = eval(optimizer)(
            self.parameters(), lr=dt_config['lr'], weight_decay=dt_config['weight_decay'])
        self.scheduler = eval(schedular)(
            self.optimizer, lambda t: min((t+1)/(Ni*Nt), 1))
        self.loss = MSELoss()


    def forward(self, T, R2G, S, A, att_mask=None):
        batch_size, seq_len = S.shape[0], S.shape[1]
        if att_mask is None: att_mask = th.ones((batch_size, seq_len), dtype=th.long)

        # print('T type: ', T.type())
        # print('S type: ', S.type())
        t_embs = self.emb_t(T)
        R2G_embs = self.emb_R2G(R2G) + t_embs
        s_embs = self.emb_s(S) + t_embs
        a_embs = self.emb_a(A) + t_embs

        stacked_ips = th.stack((R2G_embs, s_embs, a_embs),
                                dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_len, self.emb_dim)
        stacked_ips = self.emb_ln(stacked_ips)
        stacked_att_mask = th.stack((att_mask, att_mask, att_mask),
                                    dim=1).permute(0, 2, 1).reshape(batch_size, 3*seq_len)

        transformer_op = self.transformer(
            inputs_embeds=stacked_ips,
            attention_mask=stacked_att_mask)
        z = transformer_op['last_hidden_state'].reshape(batch_size, seq_len, 3, self.emb_dim).permute(0, 2, 1, 3)
        
        return self.next_act(z[:, 1])


    def predict_action(self, T, R2G, S, A):
        T = T.reshape(1, -1)
        R2G = R2G.reshape(1, -1, 1)
        S = S.reshape(1, -1, self.state_dim)
        A = A.reshape(1, -1, self.act_dim)
        
        K = self.dt_config['K']
        if K:
            T = T[:, -K:]
            R2G = R2G[:, -K:]
            S = S[:, -K:]
            A = A[:, -K:]

            att_mask = th.cat([th.zeros(K-S.shape[1]), th.ones(S.shape[1])]).reshape(1, -1).to(dtype=th.long, device=S.device)
            R2G = th.cat([th.zeros((R2G.shape[0], K-R2G.shape[1], 1), device=R2G.device), R2G], dim=1).to(dtype=th.float32)
            S = th.cat([th.zeros((S.shape[0], K-S.shape[1], self.state_dim), device=S.device), S], dim=1).to(dtype=th.float32)
            A = th.cat([th.zeros((A.shape[0], K-A.shape[1], self.act_dim), device=A.device), A], dim=1).to(dtype=th.float32)
            T = th.cat([th.zeros((T.shape[0], K-T.shape[1]), device=T.device), T], dim=1).to(dtype=th.long)
        else:
            att_mask = None

        return self(T, R2G, S, A, att_mask=att_mask)[0, -1]


    def train_model(self, data, batch_size):
        S, A, _, _, R2G, T, mask = data.sample_batch()
        a_target = th.clone(A)
        a_preds = self(T, R2G[:, :-1], S, A, att_mask=mask)

        act_dim = a_preds.shape[2]
        a_target = a_target.reshape(-1, act_dim)[mask.reshape(-1) > 0]
        a_preds = a_preds.reshape(-1, act_dim)[mask.reshape(-1) > 0]
        loss = self.loss(a_target, a_preds)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), .25)
        self.optimizer.step()
        return loss.detach().cpu().item()


    # adapted from original code, DT/gym/decision_transformer/evaluation/evaluate_episodes.py (start)
    def evaluate_model(
        self,
        env,
        device,
        mode,
        scale,
        E,
        state_mean=0.,
        state_std=1.,
        target_return=None,):


        # self.agent.eval()
        # self.agent.to(device=device)

        state_mean = th.as_tensor(state_mean).to(device=device)
        state_std = th.as_tensor(state_std).to(device=device)

        state = env.reset()
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)

        states = th.as_tensor(state).reshape(1, self.state_dim).to(device=device, dtype=th.float32)
        actions = th.zeros((0, self.act_dim), device=device, dtype=th.float32)
        rewards = th.zeros(0, device=device, dtype=th.float32)

        ep_return = target_return
        target_return = th.tensor(ep_return, device=device, dtype=th.float32).reshape(1, 1)
        timesteps = th.tensor(0, device=device, dtype=th.long).reshape(1, 1)

        sim_states = []

        episode_return, episode_length = 0, 0
        for t in range(E):

            # add padding
            actions = th.cat([actions, th.zeros((1, self.act_dim), device=device)], dim=0)
            rewards = th.cat([rewards, th.zeros(1, device=device)])

            action = self.predict_action(
                timesteps.to(dtype=th.long),
                target_return.to(dtype=th.float32),
                (states.to(dtype=th.float32) - state_mean) / state_std,
                actions.to(dtype=th.float32),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)

            cur_state = th.as_tensor(state).to(device=device).reshape(1, self.state_dim)
            states = th.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            if mode != 'delayed':
                pred_return = target_return[0,-1] - (reward/scale)
            else:
                pred_return = target_return[0,-1]
            target_return = th.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = th.cat(
                [timesteps,
                th.ones((1, 1), device=device, dtype=th.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

        return episode_return, episode_length
    # adapted from original code, DT/gym/decision_transformer/evaluation/evaluate_episodes.py (end)
