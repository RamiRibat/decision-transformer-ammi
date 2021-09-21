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
        # 'data_type': 'medium-exp', # []
        'data_type': 'medium', # [2179.8]
        # 'data_type': 'medium-rep', # []
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
