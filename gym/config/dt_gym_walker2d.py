configurations = {

    'experiment': {
        'project_name': 'decision-transformer-gym',
        'env_name': 'walker2d',
        'max_env_len': 1000,
        'scale': 1000,
        'env_targets': [5000, 2500],
        'mode': 'normal', # Sparse: 'delayed'
        'pct_traj': 1.,
        # 'device': 'cpu',
        'device': 'cuda:0',
        'WandB': True,
    },

    'data': {
        # 'data_type': 'expert', # [Avg ret: 4920.51, std: 136.39 | Max ret: 5011.69, min: 763.42]
        # 'data_type': 'medium', # [Avg ret: 2852.09, std: 1095.44 | Max ret: 4226.94, min: -6.61]
        'data_type': 'medium-replay', # [Avg ret: 682.70, std: 895.96 | Max ret: 4132.00, min: -50.20]
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
