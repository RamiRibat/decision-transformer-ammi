configurations = {

    'experiment': {
        'project_name': 'decision-transformer-gym',
        'env_name': 'halfcheetah',
        'max_env_len': 1000,
        'scale': 1000,
        'env_targets': [12000, 6000],
        'mode': 'normal', # Sparse: 'delayed'
        'pct_traj': 1.,
        # 'device': 'cpu',
        'device': 'cuda:0',
        'WandB': True,
    },

    'data': {
        # 'data_type': 'expert', # [Avg ret: 10656.43, std: 441.68 | Max ret: 11252.04, min: 2045.83]
        # 'data_type': 'medium', # [Avg ret: 4770.33, std: 355.75 | Max ret: 5309.38, min: -310.23]
        'data_type': 'medium-replay', # [Avg ret: 3093.29, std: 1680.69 | Max ret: 4985.14, min: -638.49]
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
