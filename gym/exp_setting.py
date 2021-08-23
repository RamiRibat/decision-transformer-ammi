settings = {

    'experiment': {
        'env_name': 'hopper',
        'env_targets': [3600, 1800],
        'mode': 'normal',
        'K': 20,
        'pct_traj': 1.,
        'device': 'cuda:0',
    },

    'data': {
        'type': 'medium',
        'batch_size': 64,
    },
    
    'agent': {
        'type': 'dt',
        'positions': 1024,
        'emb_dim': 128,
        'nLayers': 3,
        'nHead': 1,
        'activation': 'relu',
        'dropout': 0.1,
        'lr': 1e-4,
        'weight_decay': 1e-4,
    },

    'training': {
        'nEps': 10,
        'nInit': 1, # Episodes
        'nIter': 10000, # Iterations/Episode
    },

    'evaluation': {
        'nEps': 100,
        'render': True,
    }

}