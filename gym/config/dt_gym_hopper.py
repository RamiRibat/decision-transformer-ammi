configurations = {

    'experiment': {
        'project_name': 'decision-transformer-gym',
        'env_name': 'hopper',
        'max_env_len': 1000,
        'env_targets': [3600, 1800],
        'mode': 'normal',
        'pct_traj': 1.,
        'device': 'cuda:0',
        'WandB': True,
    },

    'data': {
        'data_type': 'medium',
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
        'lr': 1e-4,
        'weight_decay': 1e-4,
    },

    'learning': {
        'epochs': 10,
        'epoch_steps': 10000, # Iterations/Episode
        'init_epochs': 1, # Episodes
    },

    'evaluation': {
        'evaluate': True,
        'eval_episodes': 100,
        'render': None,
    }

}