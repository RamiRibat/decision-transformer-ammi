import pickle
import random

import numpy as np
from numpy.ma import concatenate
import torch as th



# adapted from original code, DT/gym/experiment.py (start)
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum
# adapted from original code, DT/gym/experiment.py (end)



class Data:
    def __init__(self, state_dim, act_dim, config):
        self.state_dim, self.act_dim = state_dim, act_dim
        self.config = config
        self.device = config['experiment']['device']

        data_path = f"offdata/{config['experiment']['env_name']}-{config['data']['data_type']}-v2.pkl"
        with open(data_path, 'rb') as f: self.Trajs = pickle.load(f)

        mode = config['experiment']['mode']
        Traj_len, S, R = [], [], []
        for path in self.Trajs:
            if mode == 'delayed':
                pass
            Traj_len.append(len(path['observations']))
            S.append(path['observations'])
            R.append(path['rewards'].sum())
        Traj_lens, R = np.array(Traj_len), np.array(R)
        S = np.concatenate(S, axis=0)
        self.state_mean, self.state_std = np.mean(S, axis=0), np.std(S, axis=0) + 1e-6
        numT = sum(Traj_len)

        # >>> adapted from original code, DT/gym/experiment.py (start)
        pct_traj = 1. # variant.get('pct_traj', 1.)

        # only train on top pct_traj trajectories (for %BC experiment)
        numT = max(int(pct_traj*numT), 1)
        sorted_inxs = np.argsort(R)  # lowest to highest
        nTrajs = 1
        T = Traj_lens[sorted_inxs[-1]]
        inx = len(self.Trajs) - 2
        while inx >= 0 and T + Traj_lens[sorted_inxs[inx]] < nTrajs:
            T += Traj_lens[sorted_inxs[inx]]
            nTrajs += 1
            inx -= 1
        sorted_inxs = sorted_inxs[-nTrajs:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        self.p_sample = Traj_lens[sorted_inxs] / sum(Traj_lens[sorted_inxs])
        # <<< adapted from original code, DT/gym/experiment.py (end)

        self.nTrajs = nTrajs
        self.sorted_inxs = sorted_inxs
        pass


    def sample_batch(self):
        device = self.device
        batch_size = self.config['data']['batch_size']
        scale = self.config['experiment']['scale']
        K = self.config['agent']['K']
        E = self.config['experiment']['max_env_len']

        idxs = np.random.choice(self.nTrajs, size=batch_size, replace=True, p=self.p_sample)

        S, A, R, D, R2G, T, mask = [], [], [], [], [], [], []

        # >>> adapted from original code, DT/gym/experiment.py (start)
        for i in range(batch_size):
            traj = self.Trajs[int(self.sorted_inxs[idxs[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            S.append(traj['observations'][si:si + K].reshape(1, -1, self.state_dim))
            A.append(traj['actions'][si:si + K].reshape(1, -1, self.act_dim))
            R.append(traj['rewards'][si:si + K].reshape(1, -1, 1))
            if 'terminals' in traj:
                D.append(traj['terminals'][si:si + K].reshape(1, -1))
            else:
                D.append(traj['dones'][si:si + K].reshape(1, -1))
            T.append(np.arange(si, si + S[-1].shape[1]).reshape(1, -1))
            T[-1][T[-1] >= E] = E-1  # padding cutoff
            R2G.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:S[-1].shape[1] + 1].reshape(1, -1, 1))
            if R2G[-1].shape[1] <= S[-1].shape[1]:
                R2G[-1] = np.concatenate([R2G[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = S[-1].shape[1]
            S[-1] = np.concatenate([np.zeros((1, K - tlen, self.state_dim)), S[-1]], axis=1)
            S[-1] = (S[-1] - self.state_mean) / self.state_std
            A[-1] = np.concatenate([np.ones((1, K - tlen, self.act_dim)) * -10., A[-1]], axis=1)
            R[-1] = np.concatenate([np.zeros((1, K - tlen, 1)), R[-1]], axis=1)
            D[-1] = np.concatenate([np.ones((1, K - tlen)) * 2, D[-1]], axis=1)
            R2G[-1] = np.concatenate([np.zeros((1, K - tlen, 1)), R2G[-1]], axis=1) / scale
            T[-1] = np.concatenate([np.zeros((1, K - tlen)), T[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, K - tlen)), np.ones((1, tlen))], axis=1))
        # <<< adapted from original code, DT/gym/experiment.py (end)

        S = th.as_tensor(np.concatenate(S, axis=0), dtype=th.float32).to(self.device)
        A = th.as_tensor(np.concatenate(A, axis=0), dtype=th.float32).to(self.device)
        R = th.as_tensor(np.concatenate(R, axis=0), dtype=th.float32).to(self.device)
        D = th.as_tensor(np.concatenate(D, axis=0), dtype=th.long).to(self.device)
        R2G = th.as_tensor(np.concatenate(R2G, axis=0), dtype=th.float32).to(self.device)
        T = th.as_tensor(np.concatenate(T, axis=0), dtype=th.long).to(self.device)
        mask = th.as_tensor(np.concatenate(mask, axis=0)).to(self.device)


        # S = th.from_numpy(np.concatenate(S, axis=0)).to(dtype=th.float32, device=self.device)
        # A = th.from_numpy(np.concatenate(A, axis=0)).to(dtype=th.float32, device=self.self.self.device)
        # R = th.from_numpy(np.concatenate(R, axis=0)).to(dtype=th.float32, device=self.self.device)
        # D = th.from_numpy(np.concatenate(D, axis=0)).to(dtype=th.float32, device=self.device)
        # R2G = th.from_numpy(np.concatenate(R2G, axis=0)).to(dtype=th.float32, device=self.device)
        # T = th.from_numpy(np.concatenate(T, axis=0)).to(dtype=th.float32, device=self.self.device)
        # mask = th.from_numpy(np.concatenate(mask, axis=0)).to(dtype=th.float32, device=self.device)

        return S, A, R, D, R2G, T, mask
