configurations = {

    'experiment': {
        'project_name': 'decision-transformer-gym',
        'env_name': 'hopper',
        'max_env_len': 1000,
        'scale': 1000,
        'env_targets': [3600, 1800],
        'mode': 'normal', # Sparse: 'delayed'
        'pct_traj': 1.,
        # 'device': 'cpu',
        'device': 'cuda:0',
        'WandB': True,
    },

    'data': {
        # 'data_type': 'expert', # [Avg ret: 3511.36, std: 328.59 | Max ret: 3759.08, min: 1645.28]
        # 'data_type': 'medium', # [Avg ret: 1422.06, std: 378.95 | Max ret: 3222.36, min: 315.87]
        'data_type': 'medium-replay', # [Avg ret: 467.30, std: 511.03 | Max ret: 3192.93, min: -1.44]
        'batch_size': 64,
    },

    'agent': {
        'type': 'dt',
        'K': 20,
        'emb_dim': 128,
        'nLayers': 3,
        'nHeads': 1,
        'positions': 1024,
        'act_fn': 'relu',
        'dropout': 0.1,
        'optimizer': "AdamW",
        'lr': 1e-4,
        'scheduler': "LambdaLR",
        'weight_decay': 1e-4,
    },

    'learning': {
        'nIter': 10,
        'niIter': 1, # Episodes
        'iter_steps': 10000, # Iterations/Episode
    },

    'evaluation': {
        'evaluate': True,
        'eval_episodes': 100,
        'render': None,
    }

}
